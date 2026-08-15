"""Read-only capability probes for optional local production backends."""

from __future__ import annotations

import ctypes.util
import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .model_store import MANIFEST_NAME, MODEL_SPECS, model_directory


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _whisper_runtime_probe() -> dict[str, Any]:
    """Report CUDA/cuBLAS visibility without loading model weights."""
    cuda = bool(shutil.which("nvidia-smi"))
    cublas = ctypes.util.find_library("cublas64_12") or ctypes.util.find_library("cublas")
    bundled_candidates: list[str] = []
    bundled_cublas: list[str] = []
    if os.name == "nt":
        for entry in sys.path:
            root = Path(entry)
            for relative in (
                Path("nvidia") / "cublas" / "bin" / "cublas64_12.dll",
                Path("nvidia") / "cudnn" / "bin" / "cudnn64_9.dll",
            ):
                candidate = (root / relative).resolve()
                if candidate.is_file():
                    bundled_candidates.append(str(candidate))
                    if candidate.name.casefold().startswith("cublas"):
                        bundled_cublas.append(str(candidate))
        if not cublas and bundled_cublas:
            cublas = bundled_cublas[0]
    return {
        "cuda_visible": cuda,
        "cublas_visible": bool(cublas),
        "cublas_library": cublas,
        "bundled_runtime_candidates": bundled_candidates,
        "cpu_fallback": True,
    }


def _command_version(command: list[str], *, timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return next(
        (line.strip() for line in (result.stdout or result.stderr).splitlines() if line.strip()),
        None,
    )


def _manifest_probe(name: str, root: Path | None) -> dict[str, Any]:
    directory = model_directory(name, root)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        return {
            "available": False,
            "directory": str(directory),
            "revision": None,
            "manifest_present": False,
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "available": False,
            "directory": str(directory),
            "revision": None,
            "manifest_present": True,
            "manifest_valid": False,
        }
    ledger = manifest.get("files")
    recorded = ledger if isinstance(ledger, dict) else {}
    files_present = bool(recorded) and all(
        (directory / str(relative)).is_file() for relative in recorded
    )
    return {
        "available": files_present,
        "directory": str(directory),
        "revision": manifest.get("revision"),
        "manifest_present": True,
        "manifest_valid": bool(recorded),
        "file_count": len(recorded),
        "hash_verification_command": f"models verify {name}",
    }


def _configured_faster_whisper_probe() -> dict[str, Any] | None:
    """Inspect an explicit external large-v3 directory without hashing weights.

    The normal model store has a manifest and a hash receipt.  An explicitly
    configured Hugging Face snapshot is intentionally allowed to remain in
    place, so the doctor probe can only establish file completeness here.  The
    resolver applies the same completeness check before constructing the
    adapter; no implicit search, copy, or download is performed.
    """

    configured = os.environ.get("VSR_FASTER_WHISPER_LARGE_V3_PATH", "").strip()
    if not configured:
        return None
    directory = Path(configured).expanduser().resolve()
    required = MODEL_SPECS["faster-whisper-large-v3"].required_files
    missing = [name for name in required if not (directory / name).is_file()]
    return {
        "available": directory.is_dir() and not missing,
        "directory": str(directory),
        "revision": None,
        "manifest_present": False,
        "manifest_valid": None,
        "source": "explicit_external",
        "configured": True,
        "required_files": list(required),
        "missing_files": missing if directory.is_dir() else list(required),
        "file_count": len(required) - len(missing) if directory.is_dir() else 0,
        "verification": "file-completeness-only",
        "hash_verification_command": None,
    }


def _tesseract_probe() -> dict[str, Any]:
    from .ocr import TesseractOCRAdapter

    adapter = TesseractOCRAdapter()
    executable = adapter.executable if adapter.available() else None
    languages: list[str] = []
    if executable:
        try:
            result = subprocess.run(
                [executable, "--list-langs"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is not None and result.returncode == 0:
            languages = [
                line.strip()
                for line in result.stdout.splitlines()[1:]
                if line.strip() and "available languages" not in line.casefold()
            ]
    available = bool(executable and languages)
    return {
        "status": "available" if available else "optional",
        "available": available,
        "executable": executable,
        "version": _command_version([executable, "--version"]) if executable else None,
        "languages": languages,
        "python_wrapper": _package_version("pytesseract"),
        "offline_ready": available,
        "fix": None
        if available
        else "Install Tesseract 5, at least one traineddata language, and the 'ocr' extra.",
    }


def _vision_probe(model_root: Path | None) -> dict[str, Any]:
    # Visual reasoning is now delegated to the host Codex/subagent workflow by
    # default.  Keep probing the local command/model only when the caller
    # explicitly opts into legacy compatibility; this avoids a slow model-store
    # walk and, more importantly, prevents a doctor/run preflight from implying
    # that Qwen is required for the supported offline path.
    legacy_enabled = os.environ.get("VSR_ALLOW_LEGACY_LOCAL_MODELS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    configured = os.environ.get("VSR_LOCAL_VISION_COMMAND")
    command = shlex.split(configured, posix=os.name != "nt") if configured else []
    executable = (
        shutil.which(command[0])
        or (str(Path(command[0]).resolve()) if Path(command[0]).is_file() else None)
        if command
        else None
    )
    llama_server = shutil.which("llama-server") if legacy_enabled else None
    model = _manifest_probe("qwen3-vl-4b-q4", model_root) if legacy_enabled else None
    managed_ready = bool(llama_server and model and model["available"])
    available = True  # CodexSubagentVisionProvider is file-only and keyless.
    return {
        "status": "available" if available else "optional",
        "available": available,
        "configured_command": command,
        "configured_executable": executable,
        "llama_server": llama_server,
        "llama_server_version": _command_version([llama_server, "--version"])
        if llama_server
        else None,
        "configured_model": os.environ.get("VSR_LOCAL_VISION_MODEL"),
        "local_model": model,
        "offline_ready": True,
        "provider": "codex-subagent",
        "route": "host_agent",
        "network_required": False,
        "legacy_local_available": bool(executable or managed_ready),
        "fix": None,
    }


def _worker_probe(
    environment_name: str, managed_name: str, development_name: str
) -> dict[str, Any]:
    configured = os.environ.get(environment_name)
    if configured:
        python = Path(configured).expanduser().resolve()
        source = "environment"
    else:
        from .worker_store import worker_python

        managed = worker_python(managed_name)
        development = (
            Path(__file__).resolve().parents[2]
            / ".artifacts"
            / "workers"
            / development_name
            / "Scripts"
            / "python.exe"
        )
        python = managed if managed.is_file() else development
        source = "managed" if managed.is_file() else "development"
    available = python.is_file()
    return {
        "status": "available" if available else "optional",
        "available": available,
        "python": str(python),
        "source": source,
    }


def probe_optional_backends(*, model_root: Path | None = None) -> dict[str, dict[str, Any]]:
    managed_large_v3 = _manifest_probe("faster-whisper-large-v3", model_root)
    external_large_v3 = _configured_faster_whisper_probe()
    # An explicit path is the same override used by the runtime resolver.  If
    # it is incomplete, retain the managed-store result as a fallback while
    # exposing the invalid override so ``doctor`` cannot hide a configuration
    # mistake behind an unrelated managed copy.
    if external_large_v3 is not None and external_large_v3["available"]:
        large_v3 = external_large_v3
    else:
        large_v3 = dict(managed_large_v3)
        if external_large_v3 is not None:
            large_v3["external_override"] = external_large_v3
    faster_whisper_package = _package_version("faster-whisper")
    large_v3_available = bool(faster_whisper_package and large_v3["available"])
    large_v3_fix = (
        None
        if large_v3_available
        else (
            "Configured VSR_FASTER_WHISPER_LARGE_V3_PATH is incomplete; add the missing "
            f"files ({', '.join(large_v3['external_override']['missing_files'])}) or unset it."
            if isinstance(large_v3.get("external_override"), dict)
            else "Install the 'asr' extra and run models fetch faster-whisper-large-v3."
        )
    )
    ecapa = _manifest_probe("speechbrain-ecapa-voxceleb", model_root)
    diarization_packages = {
        "onnxruntime": _package_version("onnxruntime"),
        "silero-vad": _package_version("silero-vad"),
        "speechbrain": _package_version("speechbrain"),
        "scikit-learn": _package_version("scikit-learn"),
    }
    diarization_available = bool(ecapa["available"] and all(diarization_packages.values()))
    legacy_enabled = os.environ.get("VSR_ALLOW_LEGACY_LOCAL_MODELS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    qwen_worker = (
        _worker_probe("VSR_QWEN_SPEECH_PYTHON", "qwen-speech", "qwen-asr")
        if legacy_enabled
        else {"available": False, "status": "legacy-disabled"}
    )
    moss_worker = (
        _worker_probe("VSR_MOSS_SPEECH_PYTHON", "moss-speech", "moss")
        if legacy_enabled
        else {"available": False, "status": "legacy-disabled"}
    )
    paddle_worker = _worker_probe("VSR_PADDLE_OCR_PYTHON", "paddle-ocr", "paddleocr")
    qwen_asr = _manifest_probe("qwen3-asr-1.7b", model_root) if legacy_enabled else {"available": False, "status": "legacy-disabled"}
    qwen_aligner = _manifest_probe("qwen3-forced-aligner-0.6b", model_root) if legacy_enabled else {"available": False, "status": "legacy-disabled"}
    moss = _manifest_probe("moss-transcribe-diarize-0.9b", model_root) if legacy_enabled else {"available": False, "status": "legacy-disabled"}
    paddle_detector = _manifest_probe("pp-ocrv5-server-det", model_root)
    paddle_recognizer = _manifest_probe("pp-ocrv5-server-rec", model_root)
    qwen_asr_ready = bool(qwen_worker["available"] and qwen_asr["available"])
    aligner_ready = bool(qwen_worker["available"] and qwen_aligner["available"])
    moss_ready = bool(moss_worker["available"] and moss["available"])
    paddle_ready = bool(
        paddle_worker["available"]
        and paddle_detector["available"]
        and paddle_recognizer["available"]
    )
    return {
        "primary_asr": {
            "status": "available" if large_v3_available else "optional",
            "available": large_v3_available,
            "worker": None,
            "model": large_v3,
            "offline_ready": large_v3_available,
            "backend": "faster-whisper",
            "legacy_qwen": {
                "worker": qwen_worker,
                "model": qwen_asr,
                "offline_ready": qwen_asr_ready,
            },
            "fix": None
            if large_v3_available
            else large_v3_fix,
        },
        "large_v3": {
            "status": "available" if large_v3_available else "unavailable",
            "available": large_v3_available,
            "package_version": faster_whisper_package,
            "model": large_v3,
            "runtime": _whisper_runtime_probe(),
            "offline_ready": large_v3_available,
            "fix": None
            if large_v3_available
            else (
                "Configured VSR_FASTER_WHISPER_LARGE_V3_PATH is incomplete; add the missing "
                f"files ({', '.join(large_v3['external_override']['missing_files'])}) or unset it."
                if isinstance(large_v3.get("external_override"), dict)
                else "Install the 'asr' extra and run models fetch faster-whisper-large-v3."
            ),
        },
        "tesseract": _tesseract_probe(),
        "primary_ocr": {
            "status": "available" if paddle_ready else "optional",
            "available": paddle_ready,
            "worker": paddle_worker,
            "detector": paddle_detector,
            "recognizer": paddle_recognizer,
            "offline_ready": paddle_ready,
            "fix": None
            if paddle_ready
            else (
                "Install workers install paddle-ocr and fetch pp-ocrv5-server-det plus "
                "pp-ocrv5-server-rec."
            ),
        },
        "neural_diarization": {
            "status": "available" if moss_ready else "optional",
            "available": moss_ready,
            "worker": moss_worker,
            "model": moss,
            "legacy_packages": diarization_packages,
            "legacy_speaker_embedding_model": ecapa,
            "legacy_available": diarization_available,
            "offline_ready": moss_ready,
            "fix": None
            if moss_ready
            else "Install workers install moss-speech and fetch moss-transcribe-diarize-0.9b.",
        },
        "forced_alignment": {
            "status": "available" if aligner_ready else "optional",
            "available": aligner_ready,
            "worker": qwen_worker,
            "model": qwen_aligner,
            "offline_ready": aligner_ready,
            "fix": None
            if aligner_ready
            else "Install workers install qwen-speech and fetch qwen3-forced-aligner-0.6b.",
        },
        "semantic_vision": _vision_probe(model_root),
    }


__all__ = ["probe_optional_backends"]
