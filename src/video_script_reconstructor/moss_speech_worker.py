"""Offline isolated worker for MOSS-Transcribe-Diarize."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

RESULT_PREFIX = "VSR_RESULT\t"


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_path(value: object, *, directory: bool, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty local path")
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        raise ValueError(f"{field} does not exist: {path}")
    return path


def _transcribe(request: dict[str, Any]) -> dict[str, Any]:
    model_path = _local_path(request.get("model_path"), directory=True, field="model_path")
    audio_path = _local_path(request.get("audio_path"), directory=False, field="audio_path")
    device_name = str(request.get("device", "cuda:0"))
    dtype_name = str(request.get("dtype", "bfloat16"))
    max_new_tokens = int(request.get("max_new_tokens", 4096))
    if not 1 <= max_new_tokens <= 65536:
        raise ValueError("max_new_tokens must be between 1 and 65536")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from moss_transcribe_diarize import parse_transcript  # type: ignore[import-not-found]
    from moss_transcribe_diarize.inference_utils import (  # type: ignore[import-not-found]
        DEFAULT_PROMPT,
        build_transcription_messages,
        generate_transcription,
    )
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore[import-not-found]

    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in the MOSS worker")
    device = torch.device(device_name)
    # trust_remote_code is intentionally limited to code inside the verified
    # local model directory; offline flags prevent resolution to a repository.
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            dtype="auto",
            local_files_only=True,
        )
        .to(dtype=dtype)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    generation = generate_transcription(
        model,
        processor,
        build_transcription_messages(audio_path, prompt=DEFAULT_PROMPT),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        device=device,
        dtype=dtype,
    )
    raw_text = str(generation.get("text", ""))
    segments = [
        {
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": str(segment.speaker),
            "text": str(segment.text),
        }
        for segment in parse_transcript(raw_text)
    ]
    return {
        "ok": True,
        "mode": "transcribe-diarize",
        "raw_output": raw_text,
        "segments": segments,
        "prompt": DEFAULT_PROMPT,
        "prompt_len": generation.get("prompt_len"),
        "generated_tokens": generation.get("generated_tokens"),
        "model_path": str(model_path),
        "device": device_name,
        "dtype": dtype_name,
        "package_versions": {
            "moss-transcribe-diarize": _version("moss-transcribe-diarize"),
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
            "moss-transcribe-diarize": _version("moss-transcribe-diarize"),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
        },
    }


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    mode = request.get("mode")
    if mode == "probe":
        return _probe()
    if mode == "transcribe-diarize":
        return _transcribe(request)
    raise ValueError(f"Unsupported worker mode: {mode!r}")


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, dict):
            raise ValueError("Worker request must be a JSON object")
        response = handle_request(request)
        status = 0
    except Exception as exc:  # pragma: no cover - subprocess boundary
        response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        status = 1
    print(RESULT_PREFIX + json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
