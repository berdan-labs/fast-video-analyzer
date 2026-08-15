"""Isolated JSON-line worker for Qwen3 speech models.

This module deliberately imports the heavyweight ML stack only inside request
handlers.  The ordinary package can therefore stay usable without PyTorch or
qwen-asr installed.  The parent process launches this module in a dedicated
Python 3.12 environment and accepts only the final sentinel-prefixed result.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

RESULT_PREFIX = "VSR_RESULT\t"


def _local_file(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty local path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{field} does not exist: {path}")
    return path


def _local_directory(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty local path")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{field} does not exist: {path}")
    return path


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _timestamp_record(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    fields = ("text", "word", "start", "end", "start_time", "end_time")
    return {field: getattr(value, field) for field in fields if hasattr(value, field)}


def _transcribe(request: dict[str, Any]) -> dict[str, Any]:
    model_path = _local_directory(request.get("model_path"), field="model_path")
    audio_path = _local_file(request.get("audio_path"), field="audio_path")
    language = request.get("language")
    if language is not None and not isinstance(language, str):
        raise ValueError("language must be a string or null")
    device = str(request.get("device", "cuda:0"))
    dtype_name = str(request.get("dtype", "float16"))
    max_new_tokens = int(request.get("max_new_tokens", 512))
    if not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be between 1 and 4096")

    # These flags are set before importing Transformers or qwen-asr.  A worker
    # is an offline inference boundary, never a model acquisition boundary.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import soundfile as sf  # type: ignore[import-not-found]
    import torch
    from qwen_asr import Qwen3ASRModel  # type: ignore[import-not-found]

    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in the speech worker")

    aligner_path_raw = request.get("aligner_path")
    aligner_path = (
        _local_directory(aligner_path_raw, field="aligner_path")
        if aligner_path_raw is not None
        else None
    )
    model = Qwen3ASRModel.from_pretrained(
        str(model_path),
        forced_aligner=str(aligner_path) if aligner_path is not None else None,
        forced_aligner_kwargs={"device_map": device, "dtype": dtype}
        if aligner_path is not None
        else None,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=max_new_tokens,
    )
    results = model.transcribe(
        str(audio_path),
        language=language,
        return_time_stamps=bool(aligner_path),
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one transcription result, received {len(results)}")
    result = results[0]
    info = sf.info(str(audio_path))
    timestamps = getattr(result, "time_stamps", None)
    return {
        "ok": True,
        "mode": "transcribe",
        "text": str(getattr(result, "text", "")),
        "language": getattr(result, "language", language),
        "duration_ms": int(round(float(info.duration) * 1000)),
        "time_stamps": [_timestamp_record(item) for item in timestamps]
        if timestamps is not None
        else None,
        "model_path": str(model_path),
        "aligner_path": str(aligner_path) if aligner_path is not None else None,
        "device": device,
        "dtype": dtype_name,
        "package_versions": {
            "qwen-asr": _version("qwen-asr"),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
        },
    }


def _align(request: dict[str, Any]) -> dict[str, Any]:
    aligner_path = _local_directory(request.get("aligner_path"), field="aligner_path")
    audio_path = _local_file(request.get("audio_path"), field="audio_path")
    transcript = request.get("text")
    language = request.get("language")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("text must be a non-empty transcript")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a non-empty supported language name")
    device = str(request.get("device", "cuda:0"))
    dtype_name = str(request.get("dtype", "float16"))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from qwen_asr import Qwen3ForcedAligner

    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in the speech worker")
    aligner = Qwen3ForcedAligner.from_pretrained(str(aligner_path), device_map=device, dtype=dtype)
    results = aligner.align(str(audio_path), text=transcript, language=language)
    if len(results) != 1:
        raise RuntimeError(f"Expected one alignment result, received {len(results)}")
    return {
        "ok": True,
        "mode": "align",
        "text": transcript,
        "language": language,
        "time_stamps": [_timestamp_record(item) for item in results[0]],
        "aligner_path": str(aligner_path),
        "device": device,
        "dtype": dtype_name,
        "package_versions": {
            "qwen-asr": _version("qwen-asr"),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
        },
    }


def _probe() -> dict[str, Any]:
    import torch

    return {
        "ok": True,
        "mode": "probe",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "package_versions": {
            "qwen-asr": _version("qwen-asr"),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
        },
    }


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    mode = request.get("mode")
    if mode == "probe":
        return _probe()
    if mode == "transcribe":
        return _transcribe(request)
    if mode == "align":
        return _align(request)
    raise ValueError(f"Unsupported worker mode: {mode!r}")


def main() -> int:
    try:
        raw = sys.stdin.read()
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("Worker request must be a JSON object")
        response = handle_request(request)
        status = 0
    except Exception as exc:  # pragma: no cover - exercised by subprocess boundary tests
        response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        status = 1
    print(RESULT_PREFIX + json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
