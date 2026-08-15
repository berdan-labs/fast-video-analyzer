"""Production boundary for locally managed Qwen3 speech workers."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ids import deterministic_id
from .model_store import verify_model
from .subtitle_parse import ParsedTranscriptSegment
from .whisper_adapter import (
    ASRAdapter,
    ASRDependencyError,
    ASRError,
    ASRResult,
    offset_transcript_timestamps,
)

RESULT_PREFIX = "VSR_RESULT\t"


@dataclass(frozen=True, slots=True)
class ForcedAlignmentSpan:
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class ForcedAlignmentResult:
    spans: list[ForcedAlignmentSpan]
    language: str
    metadata: dict[str, Any]


class Qwen3ASRAdapter(ASRAdapter):
    """Run Qwen3-ASR in a dedicated, offline Python environment.

    The base project intentionally has no PyTorch dependency.  ``worker_python``
    must point to an environment containing the optional qwen-asr runtime; it
    can also be supplied through ``VSR_QWEN_SPEECH_PYTHON``.
    """

    backend_name = "qwen3-asr"
    is_production = True

    def __init__(
        self,
        *,
        worker_python: str | Path | None = None,
        model_root: str | Path | None = None,
        model_name: str = "qwen3-asr-1.7b",
        aligner_name: str | None = None,
        device: str = "cuda:0",
        dtype: str = "float16",
        max_new_tokens: int = 512,
        timeout_seconds: float = 1800.0,
    ) -> None:
        configured_worker = worker_python or os.environ.get("VSR_QWEN_SPEECH_PYTHON")
        if configured_worker is None:
            raise ASRDependencyError(
                "Qwen3-ASR requires an isolated worker Python; set "
                "VSR_QWEN_SPEECH_PYTHON or pass worker_python"
            )
        self.worker_python = Path(configured_worker).expanduser().resolve()
        if not self.worker_python.is_file():
            raise ASRDependencyError(f"Qwen speech worker Python is absent: {self.worker_python}")
        self.model_root = Path(model_root).expanduser().resolve() if model_root else None
        self.model_name = model_name
        self.aligner_name = aligner_name
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self.cache_identity = (
            f"{model_name}|aligner={aligner_name or 'none'}|device={device}|dtype={dtype}|"
            f"max_new_tokens={max_new_tokens}"
        )
        self._model_status: dict[str, Any] | None = None
        self._aligner_status: dict[str, Any] | None = None

    def _verified_status(self, name: str) -> dict[str, Any]:
        status = verify_model(name, self.model_root)
        if not status.get("verified"):
            reason = status.get("reason") or status.get("missing_files") or "integrity check failed"
            raise ASRDependencyError(f"Local model {name!r} is not hash-verified: {reason}")
        return status

    def _ensure_models(self, *, use_aligner: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if self._model_status is None:
            self._model_status = self._verified_status(self.model_name)
        if use_aligner and self.aligner_name is not None and self._aligner_status is None:
            self._aligner_status = self._verified_status(self.aligner_name)
        return self._model_status, self._aligner_status if use_aligner else None

    def _invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUTF8": "1",
            }
        )
        package_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            package_root + os.pathsep + existing_pythonpath if existing_pythonpath else package_root
        )
        try:
            completed = subprocess.run(
                [str(self.worker_python), "-m", "video_script_reconstructor.qwen_speech_worker"],
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
            raise ASRError(f"Qwen speech worker timed out after {self.timeout_seconds:g}s") from exc
        payload: dict[str, Any] | None = None
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith(RESULT_PREFIX):
                try:
                    candidate = json.loads(line[len(RESULT_PREFIX) :])
                except json.JSONDecodeError as exc:
                    raise ASRError("Qwen speech worker returned malformed JSON") from exc
                if isinstance(candidate, dict):
                    payload = candidate
                break
        if payload is None:
            diagnostic = (completed.stderr or completed.stdout).strip()[-1200:]
            raise ASRError(f"Qwen speech worker returned no result: {diagnostic}")
        if completed.returncode != 0 or not payload.get("ok"):
            detail = payload.get("error") or completed.stderr.strip()[-1200:]
            raise ASRError(f"Qwen speech worker failed: {detail}")
        return payload

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        interval_start_ms: int | None = None,
        interval_end_ms: int | None = None,
        context_padding_ms: int = 0,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> ASRResult:
        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise ASRError(f"Audio input does not exist: {path}")
        if context_padding_ms < 0:
            raise ASRError("context_padding_ms cannot be negative")
        if (interval_start_ms is None) != (interval_end_ms is None):
            raise ASRError("A bounded ASR request requires both interval boundaries")
        if (
            interval_start_ms is not None
            and interval_end_ms is not None
            and interval_end_ms <= interval_start_ms
        ):
            raise ASRError("ASR interval end must be after its start")
        if interval_start_ms is not None and interval_end_ms is not None:
            from .transcript_repair import extract_interval_audio

            with tempfile.TemporaryDirectory(prefix="vsr-qwen-asr-interval-") as temporary:
                extraction = extract_interval_audio(
                    path,
                    Path(temporary) / "interval.wav",
                    interval_start_ms,
                    interval_end_ms,
                    context_padding_ms=context_padding_ms,
                )
                local_result = self.transcribe(
                    extraction.output_path,
                    language=language,
                    word_timestamps=word_timestamps,
                )
                local_result.segments = offset_transcript_timestamps(
                    local_result.segments, extraction.actual_start_ms
                )
                local_result.metadata.update(
                    {
                        "interval_start_ms": interval_start_ms,
                        "interval_end_ms": interval_end_ms,
                        "context_padding_ms": context_padding_ms,
                        "extracted_start_ms": extraction.actual_start_ms,
                        "extracted_end_ms": extraction.actual_end_ms,
                    }
                )
                return local_result

        use_aligner = bool(word_timestamps and self.aligner_name)
        model_status, aligner_status = self._ensure_models(use_aligner=use_aligner)
        model_path = Path(str(model_status["directory"]))
        worker_language = (
            "Filipino"
            if language is not None and language.casefold().split("-")[0] in {"fil", "tl"}
            else language
        )
        request: dict[str, Any] = {
            "mode": "transcribe",
            "model_path": str(model_path),
            "audio_path": str(path),
            "language": worker_language,
            "device": self.device,
            "dtype": self.dtype,
            "max_new_tokens": self.max_new_tokens,
        }
        if aligner_status is not None:
            request["aligner_path"] = str(aligner_status["directory"])
        payload = self._invoke(request)
        text = str(payload.get("text", ""))
        duration_ms = max(0, int(payload.get("duration_ms", 0)))
        raw_timestamps = payload.get("time_stamps")
        words: list[dict[str, Any]] = []
        if isinstance(raw_timestamps, list):
            for index, item in enumerate(raw_timestamps):
                if not isinstance(item, dict):
                    continue
                word_text = str(item.get("text", item.get("word", "")))
                start_value = item.get("start_time", item.get("start"))
                end_value = item.get("end_time", item.get("end"))
                if start_value is None or end_value is None:
                    continue
                start_ms = int(round(float(start_value) * 1000))
                end_ms = int(round(float(end_value) * 1000))
                words.append(
                    {
                        "word_id": deterministic_id(
                            "word", self.backend_name, index, start_ms, end_ms, word_text
                        ),
                        "text": word_text,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "confidence": None,
                        "source": "qwen3-forced-aligner",
                        "language": payload.get("language", language),
                        "uncertainty_flags": [],
                    }
                )
        segment_start = min((int(word["start_ms"]) for word in words), default=0)
        segment_end = max((int(word["end_ms"]) for word in words), default=duration_ms)
        segment_id = deterministic_id(
            "transcript", self.backend_name, 0, segment_start, segment_end, text
        )
        aligned = bool(words)
        segment = ParsedTranscriptSegment(
            segment_id=segment_id,
            start_ms=segment_start,
            end_ms=segment_end,
            timing_provenance=(
                "qwen3_forced_alignment" if aligned else "qwen3_asr_utterance_bounds_estimated"
            ),
            raw_text=text,
            normalized_text=text.strip(),
            language=str(payload.get("language")) if payload.get("language") else language,
            words=words,
            confidence=None,
            verification_status="automatically_transcribed",
            uncertainty_items=[] if aligned else ["word_timestamps_unavailable"],
            substantive=bool(text.strip()),
        )
        metadata = {
            "backend": self.backend_name,
            "model": self.model_name,
            "model_path_or_revision": model_status.get("revision"),
            "model_directory": str(model_path),
            "aligner": self.aligner_name if aligned else None,
            "aligner_revision": aligner_status.get("revision") if aligner_status else None,
            "device": payload.get("device", self.device),
            "dtype": payload.get("dtype", self.dtype),
            "package_versions": payload.get("package_versions", {}),
            "timing_precision": "forced-aligned" if aligned else "utterance-bounds-estimated",
            "offline": True,
        }
        return ASRResult(
            [segment],
            language=segment.language,
            language_probability=None,
            metadata=metadata,
        )


class Qwen3ForcedAlignmentAdapter:
    """Align a supplied transcript to local audio without retranscribing it."""

    backend_name = "qwen3-forced-aligner"
    is_production = True

    def __init__(
        self,
        *,
        worker_python: str | Path | None = None,
        model_root: str | Path | None = None,
        model_name: str = "qwen3-forced-aligner-0.6b",
        device: str = "cuda:0",
        dtype: str = "float16",
        timeout_seconds: float = 900.0,
    ) -> None:
        self.model_name = model_name
        self.cache_identity = f"{model_name}|device={device}|dtype={dtype}"
        self._worker = Qwen3ASRAdapter(
            worker_python=worker_python,
            model_root=model_root,
            model_name=model_name,
            device=device,
            dtype=dtype,
            timeout_seconds=timeout_seconds,
        )

    def align(
        self,
        audio_path: str | Path,
        text: str,
        *,
        language: str,
    ) -> ForcedAlignmentResult:
        audio = Path(audio_path).expanduser().resolve()
        if not audio.is_file():
            raise ASRError(f"Audio input does not exist: {audio}")
        if not text.strip():
            raise ASRError("Forced alignment requires a non-empty transcript")
        if not language.strip():
            raise ASRError("Forced alignment requires an explicit language")
        model_status, _ = self._worker._ensure_models(use_aligner=False)
        payload = self._worker._invoke(
            {
                "mode": "align",
                "aligner_path": str(model_status["directory"]),
                "audio_path": str(audio),
                "text": text,
                "language": language,
                "device": self._worker.device,
                "dtype": self._worker.dtype,
            }
        )
        raw_timestamps = payload.get("time_stamps")
        if not isinstance(raw_timestamps, list) or not raw_timestamps:
            raise ASRError("Qwen forced aligner returned no aligned spans")
        spans: list[ForcedAlignmentSpan] = []
        for item in raw_timestamps:
            if not isinstance(item, dict):
                raise ASRError("Qwen forced aligner returned a malformed span")
            try:
                span = ForcedAlignmentSpan(
                    text=str(item["text"]),
                    start_ms=int(round(float(item["start_time"]) * 1000)),
                    end_ms=int(round(float(item["end_time"]) * 1000)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ASRError("Qwen forced aligner returned a malformed span") from exc
            if span.start_ms < 0 or span.end_ms <= span.start_ms:
                raise ASRError("Qwen forced aligner returned invalid span boundaries")
            if spans and span.start_ms < spans[-1].start_ms:
                raise ASRError("Qwen forced aligner returned non-monotonic spans")
            spans.append(span)
        return ForcedAlignmentResult(
            spans=spans,
            language=language,
            metadata={
                "backend": self.backend_name,
                "model": self.model_name,
                "model_path_or_revision": model_status.get("revision"),
                "model_directory": model_status.get("directory"),
                "device": payload.get("device", self._worker.device),
                "dtype": payload.get("dtype", self._worker.dtype),
                "package_versions": payload.get("package_versions", {}),
                "offline": True,
            },
        )
