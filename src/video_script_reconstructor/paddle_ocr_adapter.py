"""Production PP-OCRv5 adapter backed by an isolated local worker."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .errors import BlockedError, InputError, ValidationFailure
from .model_store import verify_model
from .ocr import OCRAdapter, OCRObservation, OCRToken, normalize_ocr_text

RESULT_PREFIX = "VSR_RESULT\t"


class _PersistentOCRWorker:
    """Bounded newline protocol around one long-lived isolated OCR process.

    Every response is attributed by echoing the request's ``request_id``; a
    payload that does not carry the exact outstanding identifier is a stale
    terminal leftover (a late result after a timeout, or an exit notice from
    a replaced process) and is never consumed as a later request's answer.
    """

    def __init__(self, adapter: PaddleOCRV5Adapter) -> None:
        self.adapter = adapter
        self.process: subprocess.Popen[str] | None = None
        self.responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self.reader: threading.Thread | None = None
        self.lock = threading.RLock()
        self.request_number = 0
        self.generation = 0

    def _read_stdout(self) -> None:
        process = self.process
        generation = self.generation
        if process is None or process.stdout is None:
            return
        for raw_line in process.stdout:
            if not raw_line.startswith(RESULT_PREFIX):
                continue
            try:
                payload = json.loads(raw_line[len(RESULT_PREFIX) :])
            except json.JSONDecodeError as exc:
                self.responses.put(
                    {
                        "ok": False,
                        "error": f"malformed worker JSON: {exc}",
                        "_generation": generation,
                    }
                )
                continue
            if isinstance(payload, dict):
                self.responses.put(payload)
        self.responses.put(
            {
                "ok": False,
                "error": "persistent PP-OCR worker exited before returning a result",
                "_generation": generation,
            }
        )

    def _ensure_started(self, request: dict[str, Any]) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        # Starting or restarting a worker opens a new response generation.
        # Drain leftovers from the previous generation up front; strict
        # request-id matching in ``request`` remains the authoritative filter
        # because the previous reader thread can still enqueue terminal
        # payloads after this drain runs.
        self.generation += 1
        while True:
            try:
                self.responses.get_nowait()
            except queue.Empty:
                break
        environment = self.adapter._worker_environment()
        self.process = subprocess.Popen(
            [
                str(self.adapter.worker_python),
                "-m",
                "video_script_reconstructor.paddle_ocr_worker",
                "--persistent",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        self.reader = threading.Thread(
            target=self._read_stdout,
            name="vsr-paddle-ocr-reader",
            daemon=True,
        )
        self.reader.start()

    def request(self, request: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        with self.lock:
            self._ensure_started(request)
            process = self.process
            if process is None or process.stdin is None:
                raise ValidationFailure("persistent PP-OCR worker did not expose stdin")
            self.request_number += 1
            expected_request_id = str(self.request_number)
            generation = self.generation
            request_with_id = dict(request)
            request_with_id["request_id"] = expected_request_id
            try:
                process.stdin.write(json.dumps(request_with_id, ensure_ascii=False) + "\n")
                process.stdin.flush()
                payload = self._await_response(
                    expected_request_id,
                    generation,
                    deadline=time.monotonic() + timeout_seconds,
                )
            except queue.Empty as exc:
                self.close()
                raise BlockedError(
                    f"PP-OCRv5 persistent worker timed out after {timeout_seconds:g}s"
                ) from exc
            if not payload.get("ok"):
                raise ValidationFailure(str(payload.get("error") or "persistent worker failed"))
            return payload

    def _await_response(
        self, expected_request_id: str, generation: int, *, deadline: float
    ) -> dict[str, Any]:
        """Return only the response attributable to this exact request.

        Responses are consumed in arrival order, so anything already queued
        when this request starts belongs to an earlier generation or an
        abandoned request. The worker echoes ``request_id`` on every real
        response; reader-thread diagnostics carry no id and are tagged with
        the generation that produced them. A terminal diagnostic from this
        generation means the request can never be answered and surfaces
        immediately; everything else stale is discarded. The overall wait
        stays bounded by the caller's original timeout.
        """

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            payload = self.responses.get(timeout=remaining)
            if payload.get("request_id") == expected_request_id:
                return payload
            if "request_id" not in payload and payload.get("_generation") == generation:
                return payload

    def close(self) -> None:
        with self.lock:
            process = self.process
            self.process = None
            if process is None:
                return
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            finally:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass


class PaddleOCRV5Adapter(OCRAdapter):
    """Use separately hash-verified PP-OCRv5 server detector/recognizer weights."""

    backend_name = "pp-ocrv5-server"
    is_production = True

    def __init__(
        self,
        *,
        worker_python: str | Path | None = None,
        model_root: str | Path | None = None,
        detector_name: str = "pp-ocrv5-server-det",
        recognizer_name: str = "pp-ocrv5-server-rec",
        device: str = "gpu:0",
        uncertainty_threshold: float = 0.8,
        timeout_seconds: float = 900.0,
    ) -> None:
        configured = worker_python or os.environ.get("VSR_PADDLE_OCR_PYTHON")
        if configured is None:
            raise BlockedError(
                "PP-OCRv5 requires an isolated worker Python; set VSR_PADDLE_OCR_PYTHON "
                "or pass worker_python"
            )
        self.worker_python = Path(configured).expanduser().resolve()
        if not self.worker_python.is_file():
            raise BlockedError(f"PP-OCRv5 worker Python is absent: {self.worker_python}")
        self.model_root = Path(model_root).expanduser().resolve() if model_root else None
        self.detector_name = detector_name
        self.recognizer_name = recognizer_name
        self.device = device
        self.uncertainty_threshold = uncertainty_threshold
        self.timeout_seconds = timeout_seconds
        self.cache_identity = (
            f"{detector_name}|{recognizer_name}|device={device}|"
            f"uncertainty_threshold={uncertainty_threshold}"
        )
        self._statuses: tuple[dict[str, Any], dict[str, Any]] | None = None
        self._persistent_worker: _PersistentOCRWorker | None = None
        self._persistent_disabled = False
        self.persistent_worker_used = False
        self.persistent_worker_fallback_count = 0

    def spawn_worker(self) -> PaddleOCRV5Adapter:
        """Create an independent adapter with the identical OCR contract.

        A persistent Paddle worker is stateful and cannot safely serve two
        requests concurrently.  Fan-out therefore uses independent adapter
        instances, each with its own subprocess/model lifecycle.  The caller
        owns the returned adapter and must close it after the bounded work is
        complete.
        """

        return type(self)(
            worker_python=self.worker_python,
            model_root=self.model_root,
            detector_name=self.detector_name,
            recognizer_name=self.recognizer_name,
            device=self.device,
            uncertainty_threshold=self.uncertainty_threshold,
            timeout_seconds=self.timeout_seconds,
        )

    def _verified_models(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._statuses is None:
            detector = verify_model(self.detector_name, self.model_root)
            recognizer = verify_model(self.recognizer_name, self.model_root)
            for name, status in (
                (self.detector_name, detector),
                (self.recognizer_name, recognizer),
            ):
                if not status.get("verified"):
                    reason = (
                        status.get("reason")
                        or status.get("missing_files")
                        or "integrity check failed"
                    )
                    raise BlockedError(f"Local OCR model {name!r} is not hash-verified: {reason}")
            self._statuses = detector, recognizer
        return self._statuses

    def available(self) -> bool:
        if not self.worker_python.is_file():
            return False
        try:
            self._verified_models()
        except (BlockedError, ValidationFailure):
            return False
        return True

    def _worker_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
                "PYTHONUTF8": "1",
            }
        )
        package_root = str(Path(__file__).resolve().parents[1])
        inherited = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = package_root + (os.pathsep + inherited if inherited else "")
        return environment

    def _invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        environment = self._worker_environment()
        try:
            completed = subprocess.run(
                [str(self.worker_python), "-m", "video_script_reconstructor.paddle_ocr_worker"],
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise BlockedError(
                f"PP-OCRv5 worker timed out after {self.timeout_seconds:g}s"
            ) from exc
        payload: dict[str, Any] | None = None
        for line in reversed(completed.stdout.splitlines()):
            if not line.startswith(RESULT_PREFIX):
                continue
            try:
                candidate = json.loads(line[len(RESULT_PREFIX) :])
            except json.JSONDecodeError as exc:
                raise ValidationFailure("PP-OCRv5 worker returned malformed JSON") from exc
            if isinstance(candidate, dict):
                payload = candidate
            break
        if payload is None:
            diagnostic = (completed.stderr or completed.stdout).strip()[-1200:]
            raise ValidationFailure(f"PP-OCRv5 worker returned no result: {diagnostic}")
        if completed.returncode != 0 or not payload.get("ok"):
            detail = payload.get("error") or completed.stderr.strip()[-1200:]
            raise ValidationFailure(f"PP-OCRv5 worker failed: {detail}")
        return payload

    def _invoke_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        enabled = os.environ.get("VSR_PADDLE_OCR_PERSISTENT_WORKER", "1").strip().casefold()
        if enabled in {"0", "false", "no", "off"} or self._persistent_disabled:
            return self._invoke(request)
        if self._persistent_worker is None:
            self._persistent_worker = _PersistentOCRWorker(self)
        try:
            payload = self._persistent_worker.request(request, self.timeout_seconds)
            self.persistent_worker_used = True
            return payload
        except (BlockedError, OSError, ValidationFailure):
            # A persistent worker is an acceleration layer.  If an older
            # isolated worker lacks --persistent or exits unexpectedly, retry
            # this batch through the proven one-shot protocol and disable the
            # optimization for the remainder of this adapter instance.
            self._persistent_worker.close()
            self._persistent_disabled = True
            self.persistent_worker_fallback_count += 1
            return self._invoke(request)

    def close(self) -> None:
        if self._persistent_worker is not None:
            self._persistent_worker.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - interpreter shutdown is best effort
            return

    def _observation_from_payload(
        self,
        payload: dict[str, Any],
        image: Path,
        *,
        frame_id: str,
        observation_id: str,
        crop_id: str | None = None,
        language: str | None = None,
    ) -> OCRObservation:
        detector, recognizer = self._verified_models()
        lines = payload.get("lines")
        if not isinstance(lines, list):
            raise ValidationFailure("PP-OCRv5 worker returned no line array")
        tokens: list[OCRToken] = []
        uncertain: list[dict[str, object]] = []
        for index, line in enumerate(lines, 1):
            if not isinstance(line, dict):
                raise ValidationFailure("PP-OCRv5 worker returned a malformed line")
            box_value = line.get("bounding_box")
            if not isinstance(box_value, list) or len(box_value) != 4:
                raise ValidationFailure("PP-OCRv5 worker returned a malformed bounding box")
            box = (int(box_value[0]), int(box_value[1]), int(box_value[2]), int(box_value[3]))
            confidence = float(line["confidence"]) if line.get("confidence") is not None else None
            text = str(line.get("text", ""))
            tokens.append(OCRToken(text, confidence, box, 1, 1, 1, index, 1))
            if confidence is None or confidence < self.uncertainty_threshold:
                uncertain.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "bounding_box": list(box),
                        "reason": "engine confidence below threshold",
                    }
                )
        raw_text = "\n".join(token.text for token in tokens)
        confidences = [token.confidence for token in tokens if token.confidence is not None]
        region: tuple[int, int, int, int] | None = None
        if tokens:
            left = min(token.bounding_box[0] for token in tokens)
            top = min(token.bounding_box[1] for token in tokens)
            right = max(token.bounding_box[0] + token.bounding_box[2] for token in tokens)
            bottom = max(token.bounding_box[1] + token.bounding_box[3] for token in tokens)
            region = (left, top, right - left, bottom - top)
        versions = payload.get("package_versions", {})
        engine_version = (
            f"paddleocr {versions.get('paddleocr', 'unknown')}; "
            f"paddle {versions.get('paddlepaddle-gpu') or versions.get('paddlepaddle') or 'unknown'}; "
            f"det {detector.get('revision')}; rec {recognizer.get('revision')}"
        )
        return OCRObservation(
            observation_id=observation_id,
            frame_id=frame_id,
            crop_id=crop_id,
            bounding_region=region,
            raw_engine_text=raw_text,
            normalized_interpretation=normalize_ocr_text(raw_text),
            confidence=sum(confidences) / len(confidences) if confidences else None,
            alternatives=(),
            language=language,
            uncertain_characters=tuple(uncertain),
            engine="pp-ocrv5-server",
            engine_version=engine_version,
            human_decision=None,
            tokens=tuple(tokens),
        )

    def recognize(
        self,
        image_path: str | Path,
        *,
        frame_id: str,
        observation_id: str,
        crop_id: str | None = None,
        language: str | None = None,
    ) -> OCRObservation:
        image = Path(image_path).expanduser().resolve()
        if not image.is_file():
            raise InputError(f"OCR image does not exist: {image}")
        detector, recognizer = self._verified_models()
        payload = self._invoke_batch(
            {
                "mode": "recognize",
                "image_path": str(image),
                "detector_path": detector["directory"],
                "recognizer_path": recognizer["directory"],
                "device": self.device,
            }
        )
        return self._observation_from_payload(
            payload,
            image,
            frame_id=frame_id,
            observation_id=observation_id,
            crop_id=crop_id,
            language=language,
        )

    def recognize_many(
        self,
        images: list[Path],
        *,
        frame_ids: list[str],
        observation_ids: list[str],
        language: str | None = None,
    ) -> dict[str, OCRObservation]:
        if not images or len(images) != len(frame_ids) or len(images) != len(observation_ids):
            raise InputError("batch OCR images and identifiers must have equal non-zero lengths")
        detector, recognizer = self._verified_models()
        payload = self._invoke_batch(
            {
                "mode": "recognize_batch",
                "images": [str(image.expanduser().resolve()) for image in images],
                "detector_path": detector["directory"],
                "recognizer_path": recognizer["directory"],
                "device": self.device,
            }
        )
        items = payload.get("items")
        if not isinstance(items, list) or len(items) != len(images):
            raise ValidationFailure("PP-OCRv5 worker returned an invalid batch result")
        observations: dict[str, OCRObservation] = {}
        for image, frame_id, observation_id, item in zip(
            images, frame_ids, observation_ids, items, strict=True
        ):
            if not isinstance(item, dict) or item.get("image_path") != str(
                image.expanduser().resolve()
            ):
                raise ValidationFailure("PP-OCRv5 worker returned an out-of-order batch item")
            observations[frame_id] = self._observation_from_payload(
                {**payload, **item},
                image,
                frame_id=frame_id,
                observation_id=observation_id,
                language=language,
            )
        return observations
