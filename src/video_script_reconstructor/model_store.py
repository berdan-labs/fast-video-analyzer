"""Explicit, hash-verified storage for optional local model weights.

Normal reconstruction commands never call this module's networked ``fetch``
operation.  Model acquisition is an explicit CLI action; inference consumes
only a verified local directory.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import BlockedError, InputError, ValidationFailure
from .security import atomic_write_json, ensure_contained, sha256_file

MANIFEST_NAME = "model-manifest.json"
_VERIFICATION_CACHE_DIR = ".cache"
_VERIFICATION_CACHE_NAME = "model-verification.json"
_MODEL_VERIFY_CACHE: dict[
    tuple[str, str],
    tuple[
        dict[str, int],
        str,
        dict[str, dict[str, int]],
        dict[str, Any],
    ],
] = {}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    source: str
    required_files: tuple[str, ...]
    optional_extra: str
    purpose: str
    allow_patterns: tuple[str, ...] | None = None
    lifecycle: str = "current"
    replacement: str | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "faster-whisper-large-v3": ModelSpec(
        name="faster-whisper-large-v3",
        backend="faster-whisper",
        source="Systran/faster-whisper-large-v3",
        required_files=(
            "config.json",
            "model.bin",
            "preprocessor_config.json",
            "tokenizer.json",
            "vocabulary.json",
        ),
        optional_extra="asr",
        purpose="Production local ASR with word timestamps",
    ),
    "qwen3-asr-1.7b": ModelSpec(
        name="qwen3-asr-1.7b",
        backend="qwen3-asr-transformers",
        source="Qwen/Qwen3-ASR-1.7B",
        required_files=(
            "config.json",
            "generation_config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ),
        optional_extra="speech-qwen",
        purpose="Primary multilingual local ASR candidate",
        allow_patterns=(
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ),
    ),
    "qwen3-forced-aligner-0.6b": ModelSpec(
        name="qwen3-forced-aligner-0.6b",
        backend="qwen3-forced-aligner-transformers",
        source="Qwen/Qwen3-ForcedAligner-0.6B",
        required_files=(
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ),
        optional_extra="speech-qwen",
        purpose="Word/character forced alignment for supported languages",
        allow_patterns=(
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ),
    ),
    "moss-transcribe-diarize-0.9b": ModelSpec(
        name="moss-transcribe-diarize-0.9b",
        backend="moss-transcribe-diarize-transformers",
        source="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        required_files=(
            "config.json",
            "model-00000-of-00001.safetensors",
            "model.safetensors.index.json",
            "modeling_moss_transcribe_diarize.py",
            "processing_moss_transcribe_diarize.py",
            "configuration_moss_transcribe_diarize.py",
            "preprocessor_config.json",
            "tokenizer.json",
        ),
        optional_extra="speech-moss",
        purpose="Independent joint transcription, timestamps, and speaker-label candidate",
        allow_patterns=(
            "added_tokens.json",
            "chat_template.jinja",
            "config.json",
            "configuration_moss_transcribe_diarize.py",
            "generation_config.json",
            "merges.txt",
            "model-00000-of-00001.safetensors",
            "model.safetensors.index.json",
            "modeling_moss_transcribe_diarize.py",
            "preprocessor_config.json",
            "processing_moss_transcribe_diarize.py",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
    ),
    "pp-ocrv5-server-det": ModelSpec(
        name="pp-ocrv5-server-det",
        backend="paddleocr",
        source="PaddlePaddle/PP-OCRv5_server_det",
        required_files=("config.json", "inference.json", "inference.pdiparams", "inference.yml"),
        optional_extra="ocr-paddle-worker",
        purpose="Accuracy-first multilingual scene-text detection",
        allow_patterns=("config.json", "inference.json", "inference.pdiparams", "inference.yml"),
    ),
    "pp-ocrv5-server-rec": ModelSpec(
        name="pp-ocrv5-server-rec",
        backend="paddleocr",
        source="PaddlePaddle/PP-OCRv5_server_rec",
        required_files=("config.json", "inference.json", "inference.pdiparams", "inference.yml"),
        optional_extra="ocr-paddle-worker",
        purpose="Accuracy-first multilingual scene-text recognition",
        allow_patterns=("config.json", "inference.json", "inference.pdiparams", "inference.yml"),
    ),
    "speechbrain-ecapa-voxceleb": ModelSpec(
        name="speechbrain-ecapa-voxceleb",
        backend="speechbrain-ecapa",
        source="speechbrain/spkrec-ecapa-voxceleb",
        required_files=(
            "embedding_model.ckpt",
            "hyperparams.yaml",
            "mean_var_norm_emb.ckpt",
        ),
        optional_extra="diarization",
        purpose="Local neural speaker embeddings for diarization",
    ),
    "qwen2.5-vl-3b-q4": ModelSpec(
        name="qwen2.5-vl-3b-q4",
        backend="llama.cpp-multimodal",
        source="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
        required_files=(
            "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
            "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
        ),
        optional_extra="vision-local",
        purpose="Schema-constrained local semantic analysis of evidence frames",
        lifecycle="legacy",
        replacement="qwen3-vl-4b-q4",
        allow_patterns=(
            "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
            "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
        ),
    ),
    "qwen3-vl-4b-q4": ModelSpec(
        name="qwen3-vl-4b-q4",
        backend="llama.cpp-multimodal",
        source="Qwen/Qwen3-VL-4B-Instruct-GGUF",
        required_files=(
            "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
        ),
        optional_extra="vision-local",
        purpose="Primary schema-constrained local semantic analysis of evidence frames",
        allow_patterns=(
            "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
        ),
    ),
}

_MODEL_WORKERS: dict[str, str] = {
    "qwen3-asr-1.7b": "qwen-speech",
    "qwen3-forced-aligner-0.6b": "qwen-speech",
    "moss-transcribe-diarize-0.9b": "moss-speech",
    "pp-ocrv5-server-det": "paddle-ocr",
    "pp-ocrv5-server-rec": "paddle-ocr",
    "speechbrain-ecapa-voxceleb": "legacy-speechbrain",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def default_model_root() -> Path:
    configured = os.environ.get("VSR_MODEL_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return (base / "video-script-reconstructor" / "models").resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "video-script-reconstructor" / "models").resolve()


def _root(root: Path | None, *, create: bool) -> Path:
    resolved = (root or default_model_root()).expanduser().resolve()
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def model_directory(name: str, root: Path | None = None) -> Path:
    if name not in MODEL_SPECS:
        raise InputError(f"Unknown optional model: {name}")
    store = _root(root, create=True)
    return ensure_contained(store, store / name)


def _iter_model_files(
    directory: Path,
    *,
    skip_cache: bool = False,
    reject_symlinks: bool = False,
) -> list[Path]:
    """Return model files using one non-following stat per directory entry.

    The manifest and usage paths both walk model trees, which can contain many
    weight shards.  ``Path.rglob`` adds a path lookup for each ``is_file`` /
    ``is_symlink`` check.  This iterator keeps the same bounded tree walk,
    rejects symlinks for usage accounting, and retains historical manifest
    behavior of accepting symlinked regular files when hashing a manifest.
    """

    files: list[Path] = []
    pending = [directory]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in sorted(entries, key=lambda item: item.name):
                    path = Path(entry.path)
                    if skip_cache and entry.name == _VERIFICATION_CACHE_DIR:
                        continue
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    mode = entry_stat.st_mode
                    if stat.S_ISLNK(mode):
                        if reject_symlinks:
                            raise ValidationFailure(
                                f"Refusing to inspect symlink in model store: {path}"
                            )
                        # Preserve manifest behavior: a symlink to a regular
                        # file is hashable, but symlinked directories are not
                        # traversed.
                        if path.is_file():
                            files.append(path)
                        continue
                    if stat.S_ISDIR(mode):
                        pending.append(path)
                        continue
                    if stat.S_ISREG(mode):
                        files.append(path)
        except ValidationFailure:
            raise
        except OSError:
            continue
    return sorted(
        files,
        key=lambda path: path.relative_to(directory).as_posix(),
    )


def _manifest_files(directory: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in _iter_model_files(directory, skip_cache=True):
        if path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(directory).as_posix()
        files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return files


def _model_directory_usage(directory: Path) -> tuple[int, int]:
    """Count the complete on-disk footprint, including manifests and receipts."""

    file_count = 0
    total_bytes = 0
    if not directory.is_dir():
        return file_count, total_bytes
    for path in _iter_model_files(directory, reject_symlinks=True):
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ValidationFailure(f"Unable to stat model file: {path}") from exc
        file_count += 1
        total_bytes += size
    return file_count, total_bytes


def _file_stat_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(getattr(stat, "st_ctime_ns", 0)),
        "inode": int(getattr(stat, "st_ino", 0)),
    }


def _safe_file_stat_signature(path: Path) -> dict[str, int] | None:
    try:
        return _file_stat_signature(path)
    except OSError:
        return None


def _remember_verified_model(
    name: str,
    directory: Path,
    manifest_path: Path,
    manifest_sha256: str,
    recorded_files: dict[str, Any],
    status: dict[str, Any],
) -> None:
    manifest_signature = _safe_file_stat_signature(manifest_path)
    if manifest_signature is None:
        return
    file_signatures: dict[str, dict[str, int]] = {}
    for relative in recorded_files:
        try:
            file_path = ensure_contained(directory, directory / str(relative))
        except (InputError, ValidationFailure):
            return
        signature = _safe_file_stat_signature(file_path)
        if signature is None:
            return
        file_signatures[str(relative)] = signature
    _MODEL_VERIFY_CACHE[(name, str(directory))] = (
        manifest_signature,
        manifest_sha256,
        file_signatures,
        deepcopy(status),
    )


def _cached_verified_model(
    name: str,
    directory: Path,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, Any] | None:
    cached = _MODEL_VERIFY_CACHE.get((name, str(directory)))
    if (
        cached is None
        or _safe_file_stat_signature(manifest_path) != cached[0]
        or manifest_sha256 != cached[1]
    ):
        return None
    for relative, expected_signature in cached[2].items():
        try:
            file_path = ensure_contained(directory, directory / relative)
        except (InputError, ValidationFailure):
            return None
        if _safe_file_stat_signature(file_path) != expected_signature:
            return None
    status = deepcopy(cached[3])
    status["verification_source"] = "in-process-stat-cache"
    return status


def _verification_cache_path(directory: Path) -> Path:
    return directory / _VERIFICATION_CACHE_DIR / _VERIFICATION_CACHE_NAME


def _read_verification_cache(
    directory: Path,
    *,
    manifest_sha256: str,
    recorded_files: dict[str, Any],
) -> bool:
    """Accept a prior full hash only when every recorded file stat is unchanged."""

    path = _verification_cache_path(directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
            return False
        if payload.get("manifest_sha256") != manifest_sha256:
            return False
        cached_files = payload.get("files")
        if not isinstance(cached_files, dict) or set(cached_files) != set(recorded_files):
            return False
        for relative, record in recorded_files.items():
            cached = cached_files.get(relative)
            if not isinstance(record, dict) or not isinstance(cached, dict):
                return False
            if cached.get("sha256") != record.get("sha256"):
                return False
            file_path = ensure_contained(directory, directory / str(relative))
            if not file_path.is_file() or cached.get("stat") != _file_stat_signature(file_path):
                return False
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return True


def _write_verification_cache(
    directory: Path,
    *,
    manifest_sha256: str,
    recorded_files: dict[str, Any],
) -> None:
    files: dict[str, dict[str, Any]] = {}
    for relative, record in recorded_files.items():
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            return
        file_path = ensure_contained(directory, directory / str(relative))
        if not file_path.is_file():
            return
        files[str(relative)] = {
            "sha256": record["sha256"],
            "stat": _file_stat_signature(file_path),
        }
    atomic_write_json(
        _verification_cache_path(directory),
        {"schema_version": "1.0", "manifest_sha256": manifest_sha256, "files": files},
    )


def verify_model(
    name: str,
    root: Path | None = None,
    *,
    force_full: bool = False,
) -> dict[str, Any]:
    spec = MODEL_SPECS.get(name)
    if spec is None:
        raise InputError(f"Unknown optional model: {name}")
    directory = model_directory(name, root)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        return {
            "name": name,
            "available": False,
            "verified": False,
            "offline_ready": False,
            "directory": str(directory),
            "reason": "model manifest is absent",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"Invalid model manifest for {name}: {exc}") from exc
    if manifest.get("name") != name or manifest.get("source") != spec.source:
        raise ValidationFailure(f"Model manifest identity mismatch for {name}")
    recorded_files = manifest.get("files", {})
    if not isinstance(recorded_files, dict):
        raise ValidationFailure(f"Model manifest file ledger is invalid for {name}")
    manifest_sha256 = sha256_file(manifest_path)
    if not force_full:
        cached_status = _cached_verified_model(
            name, directory, manifest_path, manifest_sha256
        )
        if cached_status is not None:
            return cached_status
    missing = [relative for relative in spec.required_files if not (directory / relative).is_file()]
    if (
        not force_full
        and not missing
        and _read_verification_cache(
            directory,
            manifest_sha256=manifest_sha256,
            recorded_files=recorded_files,
        )
    ):
        status = {
            "name": name,
            "backend": spec.backend,
            "available": directory.is_dir(),
            "verified": True,
            "offline_ready": True,
            "directory": str(directory),
            "source": spec.source,
            "revision": manifest.get("revision"),
            "file_count": len(recorded_files),
            "total_bytes": sum(
                int(record.get("size_bytes", 0))
                for record in recorded_files.values()
                if isinstance(record, dict)
            ),
            "missing_files": [],
            "mismatched_files": [],
            "verification_source": "stat-bound-receipt",
        }
        _remember_verified_model(
            name, directory, manifest_path, manifest_sha256, recorded_files, status
        )
        return status
    mismatched: list[str] = []
    for relative, record in recorded_files.items():
        path = ensure_contained(directory, directory / str(relative))
        if not path.is_file() or not isinstance(record, dict):
            mismatched.append(str(relative))
            continue
        if record.get("size_bytes") != path.stat().st_size or record.get("sha256") != sha256_file(
            path
        ):
            mismatched.append(str(relative))
    verified = not missing and not mismatched and bool(recorded_files)
    if verified:
        _write_verification_cache(
            directory,
            manifest_sha256=manifest_sha256,
            recorded_files=recorded_files,
        )
    status = {
        "name": name,
        "backend": spec.backend,
        "available": directory.is_dir(),
        "verified": verified,
        "offline_ready": verified,
        "directory": str(directory),
        "source": spec.source,
        "revision": manifest.get("revision"),
        "file_count": len(recorded_files),
        "total_bytes": sum(
            int(record.get("size_bytes", 0))
            for record in recorded_files.values()
            if isinstance(record, dict)
        ),
        "missing_files": missing,
        "mismatched_files": mismatched,
        "verification_source": "full-hash",
    }
    if verified:
        _remember_verified_model(
            name, directory, manifest_path, manifest_sha256, recorded_files, status
        )
    return status


def list_models(root: Path | None = None, *, force_full: bool = False) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, spec in MODEL_SPECS.items():
        status = verify_model(name, root, force_full=force_full)
        status.update(
            {
                "purpose": spec.purpose,
                "optional_extra": spec.optional_extra,
            }
        )
        results.append(status)
    return results


def model_report(
    root: Path | None = None,
    *,
    force_full: bool = False,
    include_workers: bool = False,
) -> dict[str, Any]:
    """Summarize optional-model storage without downloading or removing anything."""

    store = _root(root, create=True)
    statuses = list_models(store, force_full=force_full)
    worker_statuses: dict[str, dict[str, Any]] = {}
    if include_workers:
        from .worker_store import list_workers

        worker_statuses = {
            str(item["name"]): item for item in list_workers() if isinstance(item, dict)
        }
    rows: list[dict[str, Any]] = []
    for status in statuses:
        name = str(status["name"])
        spec = MODEL_SPECS[name]
        directory = Path(str(status["directory"]))
        file_count, disk_bytes = _model_directory_usage(directory)
        worker_name = _MODEL_WORKERS.get(name)
        worker_status = worker_statuses.get(worker_name) if worker_name else None
        if worker_name is None:
            runtime_status = "self-contained"
        elif not include_workers:
            runtime_status = "not-probed"
        else:
            runtime_status = "verified" if worker_status and worker_status.get("verified") else "unavailable"
        runtime_unavailable = (
            include_workers
            and runtime_status == "unavailable"
            and bool(status.get("verified"))
            and disk_bytes > 0
        )
        rows.append(
            {
                "name": name,
                "purpose": spec.purpose,
                "optional_extra": spec.optional_extra,
                "lifecycle": spec.lifecycle,
                "replacement": spec.replacement,
                "verified": bool(status.get("verified")),
                "offline_ready": bool(status.get("offline_ready")),
                "disk_file_count": file_count,
                "disk_bytes": disk_bytes,
                "manifest_file_bytes": int(status.get("total_bytes") or 0),
                "runtime_worker": worker_name,
                "runtime_status": runtime_status,
                "runtime_unavailable": runtime_unavailable,
                "runtime_reason": (
                    str(worker_status.get("reason"))
                    if worker_status and worker_status.get("reason")
                    else None
                ),
                "cleanup_recommendation": (
                    f"Review before removal; prefer {spec.replacement} when compatible"
                    if spec.lifecycle == "legacy" and spec.replacement
                    else (
                        "Worker unavailable; remove if this capability is not required"
                        if runtime_unavailable
                        else "Keep while this capability is needed"
                    )
                ),
                "remove_command": (
                    f'video-script-reconstructor models remove {name} --root "{store}"'
                    if status.get("verified")
                    else None
                ),
                "removal_requires_explicit_confirmation": True,
                "removal_blocked_until_verified": not bool(status.get("verified")),
            }
        )
    rows.sort(key=lambda item: (-int(item["disk_bytes"]), str(item["name"])))
    return {
        "schema_version": "1.0",
        "root": str(store),
        "model_count": len(rows),
        "present_model_count": sum(int(item["disk_file_count"]) > 0 for item in rows),
        "verified_model_count": sum(bool(item["offline_ready"]) for item in rows),
        "legacy_model_count": sum(item["lifecycle"] == "legacy" for item in rows),
        "present_legacy_model_count": sum(
            item["lifecycle"] == "legacy" and int(item["disk_file_count"]) > 0
            for item in rows
        ),
        "legacy_bytes": sum(
            int(item["disk_bytes"]) for item in rows if item["lifecycle"] == "legacy"
        ),
        "runtime_unavailable_model_count": sum(
            bool(item["runtime_unavailable"]) for item in rows
        ),
        "runtime_unavailable_bytes": sum(
            int(item["disk_bytes"])
            for item in rows
            if bool(item["runtime_unavailable"])
        ),
        "workers_probed": include_workers,
        "total_files": sum(int(item["disk_file_count"]) for item in rows),
        "total_bytes": sum(int(item["disk_bytes"]) for item in rows),
        "verified_bytes": sum(
            int(item["disk_bytes"]) for item in rows if item["offline_ready"]
        ),
        "unverified_bytes": sum(
            int(item["disk_bytes"]) for item in rows if not item["offline_ready"]
        ),
        "models": rows,
    }


def fetch_model(
    name: str,
    root: Path | None = None,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Explicitly download one public Hugging Face model and pin its resolved commit."""

    spec = MODEL_SPECS.get(name)
    if spec is None:
        raise InputError(f"Unknown optional model: {name}")
    try:
        from huggingface_hub import model_info, snapshot_download
    except ImportError as exc:
        raise BlockedError(
            f"Fetching {name} requires huggingface-hub; install the '{spec.optional_extra}' extra"
        ) from exc
    directory = model_directory(name, root)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        existing = verify_model(name, root)
        if existing.get("verified"):
            return existing
        raise ValidationFailure(
            f"Refusing to overwrite incomplete or unverified model directory: {directory}"
        )
    info = model_info(spec.source, revision=revision)
    resolved_revision = str(info.sha)
    snapshot_download(
        repo_id=spec.source,
        revision=resolved_revision,
        local_dir=directory,
        allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
    )
    missing = [relative for relative in spec.required_files if not (directory / relative).is_file()]
    if missing:
        raise ValidationFailure(f"Downloaded model {name} lacks required files: {missing}")
    manifest = {
        "schema_version": "1.0",
        "name": name,
        "backend": spec.backend,
        "source": spec.source,
        "revision": resolved_revision,
        "fetched_at_utc": _now(),
        "files": _manifest_files(directory),
    }
    atomic_write_json(directory / MANIFEST_NAME, manifest)
    return verify_model(name, root)


def remove_model(name: str, root: Path | None = None) -> dict[str, Any]:
    directory = model_directory(name, root)
    store = _root(root, create=True)
    ensure_contained(store, directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValidationFailure(
            f"Refusing to remove an unverified directory without {MANIFEST_NAME}: {directory}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("name") != name:
        raise ValidationFailure("Refusing removal because model manifest identity mismatches")
    shutil.rmtree(directory)
    return {"name": name, "removed": True, "directory": str(directory)}


def model_specs() -> list[dict[str, Any]]:
    return [asdict(spec) for spec in MODEL_SPECS.values()]


__all__ = [
    "MANIFEST_NAME",
    "MODEL_SPECS",
    "ModelSpec",
    "default_model_root",
    "fetch_model",
    "list_models",
    "model_directory",
    "model_report",
    "model_specs",
    "remove_model",
    "verify_model",
]
