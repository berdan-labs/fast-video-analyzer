"""Explicit installation and verification of isolated heavyweight workers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import BlockedError, InputError, ValidationFailure
from .security import atomic_write_json

WORKER_MANIFEST = "worker-manifest.json"
_WORKER_VERIFY_CACHE_TTL_SECONDS = 30.0
_WORKER_VERIFY_CACHE: dict[
    tuple[str, str],
    tuple[tuple[tuple[int, int, int] | None, ...], float, dict[str, Any]],
] = {}


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    name: str
    purpose: str
    module: str
    install_steps: tuple[tuple[str, ...], ...]


WORKER_SPECS: dict[str, WorkerSpec] = {
    "qwen-speech": WorkerSpec(
        name="qwen-speech",
        purpose="Qwen3-ASR 1.7B and Qwen3 forced alignment",
        module="video_script_reconstructor.qwen_speech_worker",
        install_steps=(
            (
                "pip-install",
                "torch==2.8.0+cu128",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ),
            ("pip-install", "qwen-asr==0.0.6"),
        ),
    ),
    "moss-speech": WorkerSpec(
        name="moss-speech",
        purpose="MOSS 0.9B joint transcription and anonymous diarization",
        module="video_script_reconstructor.moss_speech_worker",
        install_steps=(
            (
                "pip-install",
                "moss-transcribe-diarize[torch-runtime] @ "
                "git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git@"
                "0e3d1403fd8f1f1c674e883ece96b9f630794ebe",
                "--torch-backend",
                "cu128",
            ),
        ),
    ),
    "paddle-ocr": WorkerSpec(
        name="paddle-ocr",
        purpose="PP-OCRv5 server detection and multilingual recognition",
        module="video_script_reconstructor.paddle_ocr_worker",
        install_steps=(
            (
                "pip-install",
                "paddlepaddle-gpu==3.2.0",
                "--index-url",
                "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
            ),
            ("pip-install", "paddleocr==3.7.0"),
        ),
    ),
}


def default_worker_root() -> Path:
    configured = os.environ.get("VSR_WORKER_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return (base / "video-script-reconstructor" / "workers").resolve()


def worker_directory(name: str, root: Path | None = None) -> Path:
    if name not in WORKER_SPECS:
        raise InputError(f"Unknown optional worker: {name}")
    base = (root or default_worker_root()).expanduser().resolve()
    return (base / name).resolve()


def worker_python(name: str, root: Path | None = None) -> Path:
    directory = worker_directory(name, root)
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _environment(module_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(module_root) + (os.pathsep + inherited if inherited else "")
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "PYTHONUTF8": "1",
        }
    )
    return environment


def _run(command: list[str], *, timeout: float = 1800.0) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BlockedError(f"Worker installation command could not complete: {command[0]}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise BlockedError(f"Worker installation failed: {detail}")
    return completed


def install_worker(
    name: str,
    root: Path | None = None,
    *,
    uv_executable: str = "uv",
) -> dict[str, Any]:
    """Install one worker after an explicit CLI action; model weights remain separate."""

    spec = WORKER_SPECS.get(name)
    if spec is None:
        raise InputError(f"Unknown optional worker: {name}")
    uv_path = shutil.which(uv_executable)
    if uv_path is None:
        raise BlockedError("The uv executable is required to install isolated workers")
    directory = worker_directory(name, root)
    directory.parent.mkdir(parents=True, exist_ok=True)
    python = worker_python(name, root)
    commands: list[list[str]] = []
    if not python.is_file():
        command = [uv_path, "venv", "--python", "3.12", str(directory)]
        _run(command)
        commands.append(command)
    for step in spec.install_steps:
        if step[0] != "pip-install":
            raise ValidationFailure(f"Invalid worker installation step for {name}")
        command = [uv_path, "pip", "install", "--python", str(python), *step[1:]]
        _run(command)
        commands.append(command)
    freeze_command = [uv_path, "pip", "freeze", "--python", str(python)]
    frozen = _run(freeze_command, timeout=120.0).stdout.splitlines()
    commands.append(freeze_command)
    manifest = {
        "schema_version": "1.0",
        "name": name,
        "purpose": spec.purpose,
        "python": str(python),
        "python_version": "3.12",
        "module": spec.module,
        "commands": commands,
        "packages": sorted(line for line in frozen if line.strip()),
        "installed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    atomic_write_json(directory / WORKER_MANIFEST, manifest)
    return verify_worker(name, root)


def verify_worker(name: str, root: Path | None = None) -> dict[str, Any]:
    spec = WORKER_SPECS.get(name)
    if spec is None:
        raise InputError(f"Unknown optional worker: {name}")
    directory = worker_directory(name, root)
    python = worker_python(name, root)
    manifest_path = directory / WORKER_MANIFEST
    if not python.is_file() or not manifest_path.is_file():
        return {
            "name": name,
            "available": False,
            "verified": False,
            "directory": str(directory),
            "python": str(python),
            "purpose": spec.purpose,
            "reason": "worker Python or manifest is absent",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"Invalid worker manifest for {name}: {exc}") from exc
    if manifest.get("name") != name or manifest.get("module") != spec.module:
        raise ValidationFailure(f"Worker manifest identity mismatch for {name}")
    package_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [str(python), "-m", spec.module],
        input='{"mode":"probe"}',
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env=_environment(package_root),
    )
    prefix = "VSR_RESULT\t"
    payload: dict[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(prefix):
            value = json.loads(line[len(prefix) :])
            payload = value if isinstance(value, dict) else None
            break
    verified = completed.returncode == 0 and bool(payload and payload.get("ok"))
    return {
        "name": name,
        "available": True,
        "verified": verified,
        "directory": str(directory),
        "python": str(python),
        "purpose": spec.purpose,
        "probe": payload,
        "reason": None if verified else (completed.stderr or "worker probe failed")[-1200:],
    }


def _stat_signature(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (int(stat.st_mtime_ns), int(stat.st_size), int(stat.st_ino))


def _worker_probe_signature(
    name: str, root: Path | None = None
) -> tuple[tuple[int, int, int] | None, ...]:
    spec = WORKER_SPECS[name]
    directory = worker_directory(name, root)
    module_name = spec.module.rsplit(".", 1)[-1]
    module_path = Path(__file__).with_name(f"{module_name}.py")
    return (
        _stat_signature(worker_python(name, root)),
        _stat_signature(directory / WORKER_MANIFEST),
        _stat_signature(module_path),
    )


def _cached_verify_worker(
    name: str, root: Path | None = None, *, force: bool = False
) -> dict[str, Any]:
    directory = worker_directory(name, root)
    key = (name, str(directory))
    signature = _worker_probe_signature(name, root)
    now = time.monotonic()
    cached = _WORKER_VERIFY_CACHE.get(key)
    if (
        not force
        and cached is not None
        and cached[0] == signature
        and now - cached[1] < _WORKER_VERIFY_CACHE_TTL_SECONDS
    ):
        return deepcopy(cached[2])
    status = verify_worker(name, root)
    _WORKER_VERIFY_CACHE[key] = (signature, now, deepcopy(status))
    return status


def list_workers(
    root: Path | None = None, *, force: bool = False
) -> list[dict[str, Any]]:
    """List worker readiness, reusing a short stat-bound probe cache.

    ``force=True`` is reserved for an explicit ``workers verify`` action.  A
    normal capability/model report avoids repeatedly starting isolated Python
    processes while still invalidating on worker, manifest, or probe-module
    changes.
    """

    return [_cached_verify_worker(name, root, force=force) for name in WORKER_SPECS]
