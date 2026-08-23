"""Offline isolated worker for PP-OCRv5 detection and recognition."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

RESULT_PREFIX = "VSR_RESULT\t"


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _path(value: object, *, directory: bool, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty local path")
    path = Path(value).expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{field} does not exist: {path}")
    return path


def _load_engine(request: dict[str, Any]) -> tuple[Any, str]:
    detector_path = _path(request.get("detector_path"), directory=True, field="detector_path")
    recognizer_path = _path(request.get("recognizer_path"), directory=True, field="recognizer_path")
    device = str(request.get("device", "gpu:0"))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    import paddle
    from paddleocr import PaddleOCR

    if device.startswith("gpu") and not paddle.device.is_compiled_with_cuda():
        raise RuntimeError("GPU was requested but PaddlePaddle has no CUDA support")
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_dir=str(detector_path),
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir=str(recognizer_path),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
    )
    return ocr, device


def _lines_from_result(result: Any) -> list[dict[str, Any]]:
    """Convert one Paddle OCR result into the newline-protocol line shape."""

    payload = result.json
    if not isinstance(payload, dict) or not isinstance(payload.get("res"), dict):
        raise RuntimeError("PP-OCRv5 returned an unsupported result payload")
    data = payload["res"]
    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    boxes = data.get("rec_boxes", [])
    if not (isinstance(texts, list) and isinstance(scores, list) and isinstance(boxes, list)):
        raise RuntimeError("PP-OCRv5 result arrays are malformed")
    lines: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        if index >= len(scores) or index >= len(boxes):
            raise RuntimeError("PP-OCRv5 result arrays have inconsistent lengths")
        box = boxes[index]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise RuntimeError("PP-OCRv5 returned an invalid recognition box")
        left, top, right, bottom = (int(round(float(value))) for value in box)
        lines.append(
            {
                "text": str(text),
                "confidence": float(scores[index]),
                "bounding_box": [left, top, right - left, bottom - top],
                "line_index": index + 1,
            }
        )
    return lines


def _recognize_image(ocr: Any, image_path: Path, *, device: str) -> list[dict[str, Any]]:
    results = list(ocr.predict(str(image_path)))
    if len(results) != 1:
        raise RuntimeError(f"Expected one OCR page result, received {len(results)}")
    return _lines_from_result(results[0])


def _recognize_images(
    ocr: Any, image_paths: list[Path], *, device: str
) -> list[list[dict[str, Any]]]:
    """Run a bounded image list through Paddle in one predictor call.

    Paddle's predictor can accept a list of paths and amortize detector setup
    and GPU synchronization across the batch.  The returned sequence is kept
    in the exact input order; callers still validate its cardinality before
    attaching observations to frame IDs.
    """

    results = list(ocr.predict([str(path) for path in image_paths]))
    if len(results) != len(image_paths):
        raise RuntimeError(
            "Expected one OCR page result per image, "
            f"received {len(results)} for {len(image_paths)} images"
        )
    return [_lines_from_result(result) for result in results]


def _package_versions() -> dict[str, str | None]:
    return {
        "paddleocr": _version("paddleocr"),
        "paddlex": _version("paddlex"),
        "paddlepaddle-gpu": _version("paddlepaddle-gpu"),
        "paddlepaddle": _version("paddlepaddle"),
    }


def _recognize(
    request: dict[str, Any],
    *,
    ocr: Any | None = None,
    loaded_device: str | None = None,
    loaded_versions: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    image_path = _path(request.get("image_path"), directory=False, field="image_path")
    if ocr is None:
        ocr, device = _load_engine(request)
        versions = _package_versions()
    else:
        device = loaded_device or str(request.get("device", "gpu:0"))
        versions = loaded_versions or _package_versions()
    return {
        "ok": True,
        "mode": "recognize",
        "lines": _recognize_image(ocr, image_path, device=device),
        "device": device,
        "package_versions": versions,
    }


def _recognize_batch(
    request: dict[str, Any],
    *,
    ocr: Any | None = None,
    loaded_device: str | None = None,
    loaded_versions: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    images = request.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("images must be a non-empty list")
    if ocr is None:
        ocr, device = _load_engine(request)
        versions = _package_versions()
    else:
        device = loaded_device or str(request.get("device", "gpu:0"))
        versions = loaded_versions or _package_versions()
    image_paths = [_path(value, directory=False, field="image_path") for value in images]
    line_sets = _recognize_images(ocr, image_paths, device=device)
    items = [
        {"image_path": str(image_path), "lines": lines}
        for image_path, lines in zip(image_paths, line_sets, strict=True)
    ]
    return {
        "ok": True,
        "mode": "recognize_batch",
        "items": items,
        "device": device,
        "package_versions": versions,
    }


def _engine_key(request: dict[str, Any]) -> tuple[str, str, str]:
    """Return the immutable model/device identity for a persistent session."""

    detector_path = _path(request.get("detector_path"), directory=True, field="detector_path")
    recognizer_path = _path(request.get("recognizer_path"), directory=True, field="recognizer_path")
    return str(detector_path), str(recognizer_path), str(request.get("device", "gpu:0"))


class _PersistentSession:
    """Keep one PP-OCR engine alive while a bounded request stream is active.

    The adapter sends one request at a time, so the session can preserve exact
    input order without introducing a queue or concurrent Paddle calls.  Model
    and device identity is checked on every request; a caller cannot silently
    switch weights underneath an already-loaded engine.
    """

    def __init__(self) -> None:
        self._ocr: Any | None = None
        self._device: str | None = None
        self._engine_key: tuple[str, str, str] | None = None
        self._versions: dict[str, str | None] | None = None
        # Set only on the request that paid the one-time engine load; the
        # response builder reports it once and clears it so later requests
        # carry pure inference time.
        self.last_engine_load_seconds: float | None = None

    def _ensure_engine(self, request: dict[str, Any]) -> None:
        key = _engine_key(request)
        if self._ocr is None:
            load_started = time.perf_counter()
            self._ocr, self._device = _load_engine(request)
            self.last_engine_load_seconds = time.perf_counter() - load_started
            self._engine_key = key
            self._versions = _package_versions()
            return
        if key != self._engine_key:
            raise ValueError("persistent PP-OCR session cannot change model paths or device")

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        mode = request.get("mode")
        if mode not in {"recognize", "recognize_batch"}:
            raise ValueError(f"Unsupported persistent worker mode: {mode!r}")
        request_started = time.perf_counter()
        self._ensure_engine(request)
        if mode == "recognize":
            response = _recognize(
                request,
                ocr=self._ocr,
                loaded_device=self._device,
                loaded_versions=self._versions,
            )
        else:
            response = _recognize_batch(
                request,
                ocr=self._ocr,
                loaded_device=self._device,
                loaded_versions=self._versions,
            )
        # Instrumentation only: lets a benchmark separate engine warmup from
        # steady-state inference. Extra protocol keys never reach observations.
        response["elapsed_seconds"] = round(time.perf_counter() - request_started, 6)
        if self.last_engine_load_seconds is not None:
            response["engine_load_seconds"] = round(self.last_engine_load_seconds, 6)
            self.last_engine_load_seconds = None
        return response


def _probe() -> dict[str, Any]:
    import paddle

    return {
        "ok": True,
        "mode": "probe",
        "cuda_compiled": bool(paddle.device.is_compiled_with_cuda()),
        "gpu_count": int(paddle.device.cuda.device_count()),
        "package_versions": {
            "paddleocr": _version("paddleocr"),
            "paddlex": _version("paddlex"),
            "paddlepaddle-gpu": _version("paddlepaddle-gpu"),
            "paddlepaddle": _version("paddlepaddle"),
        },
    }


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    mode = request.get("mode")
    if mode == "probe":
        return _probe()
    if mode == "recognize":
        return _recognize(request)
    if mode == "recognize_batch":
        return _recognize_batch(request)
    raise ValueError(f"Unsupported worker mode: {mode!r}")


def _persistent_main() -> int:
    """Serve newline-delimited requests until the adapter closes stdin."""

    session = _PersistentSession()
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        request_id: object = None
        try:
            request = json.loads(raw_line)
            if not isinstance(request, dict):
                raise ValueError("Worker request must be a JSON object")
            request_id = request.get("request_id")
            response = session.handle(request)
            status = 0
        except Exception as exc:  # pragma: no cover - subprocess boundary
            response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
            status = 1
        if request_id is not None:
            response["request_id"] = request_id
        print(
            RESULT_PREFIX + json.dumps(response, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
        # A request failure is reported to the client, which tears down this
        # session and restarts from its durable OCR checkpoint.  Keeping the
        # process alive here makes the protocol inspectable and avoids a race
        # between error delivery and process teardown.
        if status:
            continue
    return 0


def main() -> int:
    if "--persistent" in sys.argv[1:]:
        return _persistent_main()
    one_shot_started = time.perf_counter()
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, dict):
            raise ValueError("Worker request must be a JSON object")
        response = handle_request(request)
        status = 0
    except Exception as exc:  # pragma: no cover - subprocess boundary
        response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        status = 1
    if status == 0:
        # The one-shot path includes interpreter and Paddle import time, so
        # this number is a cold-start total, not comparable to the persistent
        # session's per-request elapsed time.
        response["elapsed_seconds"] = round(time.perf_counter() - one_shot_started, 6)
    print(RESULT_PREFIX + json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
