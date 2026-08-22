from __future__ import annotations

import gc
import hashlib
import importlib.util
import inspect
import json
import logging
import math
import os
import platform
import queue
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from . import __version__
from .audit import audit_project
from .cache import cache_key
from .config import config_digest, load_config
from .errors import BlockedError, InputError, ValidationFailure
from .manifest import ManifestBuilder
from .render_markdown import render_to_path
from .security import (
    atomic_update_json_fields,
    atomic_write_json,
    canonical_compact_for_payload,
    safe_slug,
    sha256_file,
    validate_remote_url,
)
from .validate_output import (
    ValidationResult,
    _metadata_verify_workers,
    read_trusted_validation_receipt,
    refresh_validation_receipt_signature,
    validate_project,
    write_validation_receipt,
)

if TYPE_CHECKING:
    from .ocr import OCRAdapter
    from .providers.base import VisionProvider

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}
TRANSCRIPT_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa", ".json", ".txt"}
LOGGER = logging.getLogger(__name__)
_TOOL_VERSION_CACHE: dict[str, tuple[tuple[int, int, int, int], str | None]] = {}

# Shared JSON receipts (OCR and visual-survey state) are written at most once
# per stage run, but a shared cache can receive receipts from many projects.
# Keep a stat-bound process-local size ledger so pruning does not rescan every
# prior receipt after each write.  Full inventory remains authoritative at the
# first write, at the bounded reconciliation interval, and whenever the
# estimated budget is crossed.  A new process starts with no ledger and thus
# always performs the safe inventory path before accepting a warm estimate.
_SHARED_JSON_PRUNE_INTERVAL = 32
_SHARED_JSON_PRUNE_STATE: dict[Path, tuple[int, int, dict[Path, int]]] = {}

# A normal host-agent run should cover the complete semantic frontier for the
# long-form sources this tool targets.  The previous implicit value (32) was a
# development-safe probe limit that silently forced a continuation apply for
# otherwise ordinary 2--5 hour videos.  Keep the bound finite for prompt,
# storage, and review safety; callers can still set an explicit lower budget.
_DEFAULT_HOST_REVIEW_MAX_PACKETS = 4_096


def default_output_root() -> Path:
    """Return the legacy shared root for batch/compatibility projects.

    ``VSR_OUTPUT_ROOT`` is an explicit escape hatch for a dedicated output
    volume or CI workspace.  Without it, generated artifacts stay outside the
    repository and source tree under the per-user Documents directory.
    """

    configured = os.environ.get("VSR_OUTPUT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "Documents" / "Script Reconstructor Outputs").resolve()


ANALYZER_OUTPUT_SUFFIX = " (Analyzer Outputs)"


def colocated_output_dir(source: str | Path) -> Path:
    """Return the default output directory next to one source video.

    The legacy ``default_output_root`` remains available for batch/compatibility
    callers.  A single public analyzer run is intentionally source-adjacent so
    its Markdown report and evidence can be moved with the input media.
    """

    source_path = Path(source).expanduser().resolve(strict=True)
    return source_path.parent / f"{source_path.stem}{ANALYZER_OUTPUT_SUFFIX}"


def _single_output_dir(source: Path, output_root: Path | None) -> tuple[Path, bool]:
    """Resolve one source's project directory and whether it is colocated.

    ``VSR_OUTPUT_ROOT`` is intentionally authoritative when set: it is the
    documented escape hatch for CI and dedicated output volumes.  With no
    explicit root, the public single-video command keeps its report beside the
    source media.
    """

    if output_root is not None:
        root = output_root.expanduser().resolve()
        return root / safe_slug(source.stem), False
    configured = os.environ.get("VSR_OUTPUT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve() / safe_slug(source.stem), False
    return colocated_output_dir(source), True


def _markdown_filename(source: Path, *, colocated: bool) -> str:
    """Choose the user-facing report name without changing legacy explicit roots."""

    if colocated:
        return f"{source.stem}.md"
    return f"{safe_slug(source.stem)}.reconstruction.md"


def _next_review_id(reviews: Sequence[Mapping[str, Any]]) -> str:
    """Return a review ID that cannot collide with an existing project item.

    Review items can be created by independent stages.  Visual sampling may
    skip a frame index, so deriving an ID from that index is not equivalent to
    allocating the next review number.  Always allocate from the highest
    numeric ID already present instead of using ``len(reviews)``.
    """

    numbers = [
        int(review_id.removeprefix("R"))
        for review in reviews
        if (review_id := str(review.get("review_id", ""))).startswith("R")
        and review_id.removeprefix("R").isdigit()
    ]
    return f"R{max(numbers, default=0) + 1:06d}"


def _executor_context(
    worker_pool: ThreadPoolExecutor | None,
    *,
    max_workers: int,
    thread_name_prefix: str,
) -> AbstractContextManager[ThreadPoolExecutor]:
    """Reuse a visual-stage pool when available, otherwise create one locally."""

    if worker_pool is not None:
        return nullcontext(worker_pool)
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)


class ASRAdapter(Protocol):
    def transcribe(
        self, media_path: str | Path, *, language: str | None, word_timestamps: bool
    ) -> Iterable[Any]: ...


@dataclass
class RunResult:
    project_dir: Path
    markdown_path: Path
    status: str
    exit_code: int
    validation: ValidationResult | None


def _publish_resource_telemetry(
    project_dir: Path,
    project: dict[str, Any],
    project_manifest: dict[str, Any],
) -> None:
    """Publish output telemetry until its own JSON write has settled.

    The canonical manifest and the run manifest are part of the generated
    project, so writing a byte count into them changes the byte count being
    reported.  A short fixed-point loop avoids publishing a stale pre-write
    size while keeping the normal path bounded (in practice it converges in
    two passes, even when a decimal-width boundary is crossed).
    """

    from .resource_usage import resource_snapshot

    performance = project_manifest.setdefault("performance", {})
    canonical_path = project_dir / ".state" / "canonical-project.json"
    run_manifest_path = project_dir / ".state" / "run-manifest.json"
    for _ in range(6):
        snapshot = resource_snapshot(project_dir)
        performance["resource_usage"] = snapshot
        project["manifest"] = project_manifest
        atomic_update_json_fields(
            canonical_path,
            {"manifest": project_manifest},
            fallback_payload=project,
        )
        atomic_write_json(run_manifest_path, project_manifest)
        settled = resource_snapshot(project_dir)
        if settled.get("output") == snapshot.get("output"):
            return


def _tool_version(binary: str | None) -> str | None:
    if not binary:
        return None
    try:
        stat = Path(binary).stat()
        signature = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(getattr(stat, "st_ctime_ns", 0)),
            int(getattr(stat, "st_ino", 0)),
        )
    except OSError:
        return None
    cached = _TOOL_VERSION_CACHE.get(binary)
    if cached is not None and cached[0] == signature:
        return cached[1]
    version: str | None = None
    try:
        completed = subprocess.run(
            [binary, "-version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    else:
        if completed.returncode == 0:
            version = (completed.stdout or completed.stderr).splitlines()[0].strip() or None
    _TOOL_VERSION_CACHE[binary] = (signature, version)
    return version


def _tool_versions_for_cache_key() -> dict[str, str | None]:
    """Probe independent FFmpeg tools concurrently for cache-key setup."""

    paths = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vsr-tool-probe") as executor:
        futures = {
            name: executor.submit(_tool_version, binary) for name, binary in paths.items()
        }
        return {name: futures[name].result() for name in paths}


def _crop_covers_full_frame(crop_xywh: Sequence[int], width: int, height: int) -> bool:
    """Identify a non-localized crop that would duplicate its parent pixels."""

    return tuple(int(value) for value in crop_xywh) == (0, 0, int(width), int(height))


@dataclass(frozen=True)
class _CandidatePath:
    path: Path
    source_type: str
    origin: str | None = None
    language: str | None = None
    authorship: str | None = None
    source_track: str | None = None


@dataclass(frozen=True)
class _PrecomputedVisualSurvey:
    """A detector result produced while the transcript stage is running.

    The shared frames are exact-safe hard-cut/periodic PNGs.  Adaptive samples
    remain measurements only and are re-extracted by the normal visual stage,
    so overlapping the detector with ASR changes scheduling but never the
    evidence authority.
    """

    candidates: tuple[Any, ...]
    shared_frames: tuple[Any, ...]
    shared_frame_dir: Path
    prefetched_frame_count: int = 0
    prefetched_batch_count: int = 0
    prefetch_failed_batch_count: int = 0
    prefetch_elapsed_seconds: float = 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _visual_reuse_config_digest(config: Any) -> str:
    """Hash configuration that can change source-pixel visual evidence.

    ASR window geometry changes the transcript checkpoint and downstream block
    text, but it cannot change a frame decoded from the same source at the same
    schedule.  Keep every other setting in this stage hash so visual settings,
    OCR policy, privacy, and model-independent reconstruction changes remain
    invalidators.  The narrow exclusion is intentionally local to visual-state
    reuse; the full configuration digest still keys the run and canonical
    reproducibility record.
    """

    data = config.model_dump(mode="json")
    asr = data.get("asr")
    if isinstance(asr, dict):
        asr.pop("chunk_seconds", None)
        asr.pop("overlap_seconds", None)
    encoded = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _visual_reuse_transcript_compatible(
    prior_project: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
) -> bool:
    """Confirm that prior visual links still point at the current transcript.

    Reusing decoded pixels is exact-safe, but the canonical frame envelopes also
    carry block and segment IDs.  Require stable block topology, timing, and
    segment IDs before taking the fast path; otherwise the normal visual stage
    rebuilds links and packets rather than leaving stale transcript references.
    """

    previous_blocks = [
        item
        for item in prior_project.get("script_blocks", [])
        if isinstance(item, Mapping)
    ]
    current_blocks = [item for item in blocks if isinstance(item, Mapping)]
    if len(previous_blocks) != len(current_blocks):
        return False
    for previous, current in zip(previous_blocks, current_blocks, strict=True):
        if (
            str(previous.get("block_id")) != str(current.get("block_id"))
            or previous.get("start_ms") != current.get("start_ms")
            or previous.get("end_ms") != current.get("end_ms")
            or [str(value) for value in previous.get("transcript_segment_ids", [])]
            != [str(value) for value in current.get("transcript_segment_ids", [])]
        ):
            return False

    block_ids = {str(block.get("block_id")) for block in current_blocks}
    segment_ids = {
        str(segment_id)
        for block in current_blocks
        for segment_id in block.get("transcript_segment_ids", [])
    }
    payload_by_image_id = {
        str(payload.get("image", {}).get("image_id")): payload
        for payload in prior_project.get("evidence_image_metadata", [])
        if isinstance(payload, Mapping)
    }
    for frame in prior_project.get("frames", []):
        if not isinstance(frame, Mapping):
            continue
        if any(str(value) not in block_ids for value in frame.get("linked_block_ids", [])):
            return False
        image_id = str(frame.get("frame_id"))
        payload = payload_by_image_id.get(image_id)
        if payload is None:
            payload = frame.get("metadata")
        if not isinstance(payload, Mapping):
            return False
        links = payload.get("links", {})
        if not isinstance(links, Mapping):
            return False
        if any(str(value) not in segment_ids for value in links.get("segment_ids", [])):
            return False
    return True


def _record(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return cast(dict[str, Any], value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def classify_input(value: str | Path) -> str:
    text = str(value)
    if text.startswith(("http://", "https://")):
        validate_remote_url(text)
        return "remote_media"
    suffix = Path(text).suffix.casefold()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in TRANSCRIPT_EXTENSIONS:
        return "transcript"
    raise InputError(
        "Unsupported input type. Expected MP4/MKV/MOV/WebM, WAV/FLAC/MP3/M4A, "
        "SRT/VTT/ASS/SSA, timestamped JSON, or plain text."
    )


def resolve_dependencies() -> dict[str, dict[str, Any]]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    tesseract = shutil.which("tesseract")
    package_root = Path(__file__).resolve().parent
    source_resources = package_root.parents[1]
    packaged_resources = package_root / "resources"
    resources_ok = all(
        (source_resources / name).exists() or (packaged_resources / name).exists()
        for name in ("configs", "prompts", "references", "assets", "agents", "SKILL.md")
    )
    return {
        "python": {
            "status": "available" if sys.version_info >= (3, 10) else "blocking-for-strict",
            "value": sys.version.split()[0],
            "fix": "Install Python 3.10 or newer.",
        },
        "package_import": {
            "status": "available",
            "value": str(Path(__file__).resolve()),
            "fix": None,
        },
        "ffmpeg": {
            "status": "available" if ffmpeg else "blocking-for-strict",
            "value": ffmpeg,
            "version": _tool_version(ffmpeg),
            "fix": "Install FFmpeg and place ffmpeg on PATH.",
        },
        "ffprobe": {
            "status": "available" if ffprobe else "blocking-for-strict",
            "value": ffprobe,
            "version": _tool_version(ffprobe),
            "fix": "Install FFmpeg and place ffprobe on PATH.",
        },
        "faster_whisper": {
            "status": "available" if importlib.util.find_spec("faster_whisper") else "unavailable",
            "value": "large-v3 required by default; weights are never downloaded without permission",
            "fix": "Install video-script-reconstructor[asr] and provision large-v3 locally, or explicitly permit its download.",
        },
        "ocr": {
            "status": "available" if tesseract else "optional",
            "value": tesseract,
            "fix": "Install Tesseract and video-script-reconstructor[ocr] to enable local OCR.",
        },
        "diarization": {
            "status": "optional",
            "value": "not required by the core",
            "fix": "Configure a tested local diarization adapter if needed.",
        },
        "image_metadata": {
            "status": "available",
            "value": "PNG iTXt, Description, schema/digest/read-back/pixel-invariance",
            "fix": None,
        },
        "vision_provider": {
            "status": "optional",
            "value": "host-agent route available; external providers require explicit permission",
            "fix": "Use evidence packet/observation commands or configure a permitted provider.",
        },
        "package_data": {
            "status": "available" if resources_ok else "blocking-for-strict",
            "value": resources_ok,
            "fix": "Reinstall a complete wheel containing configs, prompts, references, and assets.",
        },
    }


def _cache_directory_usage(path: Path) -> dict[str, Any]:
    """Return safe, read-only usage telemetry for a generated cache root.

    Cache telemetry is collected by ``doctor`` and can run over thousands of
    generated files.  ``Path.rglob`` followed by ``is_symlink``/``is_file``
    and ``stat`` performs multiple path-based metadata lookups per file.  A
    non-following ``DirEntry.stat`` gives us one lookup per entry while also
    keeping symlinked directories outside the cache boundary.
    """

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {"path": str(resolved), "file_count": 0, "bytes": 0, "exists": False}
    file_count = 0
    total_bytes = 0
    try:
        pending = [resolved]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    entry_stat = entry.stat(follow_symlinks=False)
                    mode = entry_stat.st_mode
                    if stat.S_ISLNK(mode):
                        continue
                    if stat.S_ISDIR(mode):
                        pending.append(Path(entry.path))
                        continue
                    if stat.S_ISREG(mode):
                        file_count += 1
                        total_bytes += entry_stat.st_size
    except OSError as exc:
        return {
            "path": str(resolved),
            "file_count": file_count,
            "bytes": total_bytes,
            "exists": True,
            "error": str(exc),
        }
    return {"path": str(resolved), "file_count": file_count, "bytes": total_bytes, "exists": True}


def doctor_report(*, output_path: Path | None = None, offline: bool = True) -> dict[str, Any]:
    target = (output_path or Path.cwd()).resolve()
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    writable = os.access(target, os.W_OK)
    checks = resolve_dependencies()
    from .backend_probe import probe_optional_backends

    backends = probe_optional_backends()
    checks["optional_backends"] = {
        "status": "available",
        "value": backends,
        "fix": "Install only the optional capability packs required for this workflow.",
    }
    primary_asr = backends["primary_asr"]
    large_v3 = backends["large_v3"]
    # large-v3 is the documented default for multilingual/Filipino runs.  A
    # complete explicit offline snapshot therefore makes the primary speech
    # path ready without requiring any legacy fallback worker.
    primary_speech_ready = bool(primary_asr["offline_ready"])
    primary_ocr = backends["primary_ocr"]
    tesseract_backend = backends["tesseract"]
    diarization_backend = backends["neural_diarization"]
    vision_backend = backends["semantic_vision"]
    checks["speech_recognition"] = {
        "status": "available" if primary_speech_ready else "optional",
        "value": {
            "primary": "faster-whisper large-v3 for Filipino/multilingual",
            "primary_offline_ready": primary_speech_ready,
            "legacy_qwen_fallback_offline_ready": bool(
                primary_asr.get("legacy_qwen", {}).get("offline_ready", False)
            ),
            "whisper_large_v3_offline_ready": large_v3["offline_ready"],
            "whisper_model_source": large_v3["model"].get("source", "managed_store"),
            "cpu_fallback": "enabled when CUDA/cuBLAS cannot load",
        },
        "fix": None if primary_speech_ready else (primary_asr["fix"] or large_v3["fix"]),
    }
    checks["faster_whisper"] = {
        "status": "available" if large_v3["offline_ready"] else "optional",
        "value": "preferred for Filipino/multilingual speech; CPU/int8 fallback is enabled",
        "fix": (
            large_v3["fix"]
            or (
                "NVIDIA GPU detected but cuBLAS is unavailable. Install the optional `cuda` "
                "extra (`pip install '.[asr,cuda]'`) to enable GPU large-v3; CPU/int8 fallback "
                "remains enabled and auditable."
                if large_v3.get("runtime", {}).get("cuda_visible")
                and not large_v3.get("runtime", {}).get("cublas_visible")
                else None
            )
        ),
    }
    checks["ocr"] = {
        "status": (
            "available"
            if primary_ocr["offline_ready"] or tesseract_backend["offline_ready"]
            else "optional"
        ),
        "value": {
            "primary": "PP-OCRv5 server detection/recognition",
            "primary_offline_ready": primary_ocr["offline_ready"],
            "fallback": "Tesseract",
            "fallback_offline_ready": tesseract_backend["offline_ready"],
        },
        "fix": primary_ocr["fix"] or tesseract_backend["fix"],
    }
    checks["diarization"] = {
        "status": "optional",
        "value": {
            "primary": "neutral speaker labels from Whisper/timeline; neural diarization is optional",
            "offline_ready": diarization_backend["offline_ready"],
            "legacy_moss_available": bool(diarization_backend.get("offline_ready", False)),
        },
        "fix": "Enable the legacy diarization worker only for an explicit corroboration experiment.",
    }
    checks["vision_provider"] = {
        "status": "available" if vision_backend["offline_ready"] else "optional",
        "value": {
            "primary": "Codex/subagent file review bundle",
            "offline_ready": vision_backend["offline_ready"],
            "network_required": False,
            "legacy_local_qwen_available": bool(
                vision_backend.get("legacy_local_available", False)
            ),
        },
        "fix": vision_backend["fix"],
    }
    checks["disk"] = {
        "status": "available" if usage.free >= 512 * 1024 * 1024 else "blocking-for-strict",
        "value": {"free_bytes": usage.free, "path": str(target)},
        "fix": "Free at least 512 MiB and estimate source-specific evidence usage before running.",
    }
    checks["output_write"] = {
        "status": "available" if writable else "blocking-for-strict",
        "value": str(target),
        "fix": "Select a writable output directory.",
    }
    checks["compute"] = {
        "status": "available",
        "value": {
            "cpu_count": os.cpu_count(),
            "machine": platform.machine(),
            "gpu": "nvidia-smi available" if shutil.which("nvidia-smi") else "not detected",
        },
        "fix": None,
    }
    checks["scheduling"] = {
        "status": "available",
        "value": {
            "frame_extract_workers": _visual_frame_workers(),
            "frame_analysis_workers": _visual_analysis_workers(),
            "crop_prepare_workers": _visual_crop_workers(),
            "survey_ffmpeg_threads": _visual_survey_ffmpeg_threads(),
            "ocr_workers": _ocr_workers(),
            "ocr_checkpoint_batch": _ocr_checkpoint_flush_interval(),
            "ocr_batch_size": _ocr_batch_size(),
            "paddle_ocr_workers": _paddle_ocr_batch_workers(),
            "asr_cpu_threads": _asr_cpu_threads(),
            "asr_num_workers": _faster_whisper_num_workers(),
            "validator_metadata_workers": _metadata_verify_workers(),
            "parallel_visual_survey": _parallel_visual_survey_enabled(),
            "parallel_visual_warmup": _parallel_visual_warmup_enabled(),
            # Read-only summary; doctor must never trigger a decode probe.
            "survey_hwaccel": _survey_hwaccel_telemetry(),
        },
        "fix": (
            "Override VSR_FRAME_EXTRACT_WORKERS, VSR_FRAME_ANALYSIS_WORKERS, "
            "VSR_CROP_PREP_WORKERS, "
            "VSR_SURVEY_FFMPEG_THREADS, VSR_OCR_WORKERS, VSR_OCR_CHECKPOINT_BATCH, "
            "VSR_OCR_BATCH_SIZE, "
            "VSR_PADDLE_OCR_WORKERS, "
            "VSR_ASR_CPU_THREADS, VSR_FASTER_WHISPER_NUM_WORKERS, or "
            "VSR_VALIDATOR_METADATA_WORKERS, VSR_PARALLEL_VISUAL_WARMUP_BATCH_SIZE, or "
            "VSR_PARALLEL_VISUAL_WARMUP_MAX_FRAMES "
            "only after a representative offline benchmark."
        ),
    }
    asr_cache_root = _asr_shared_cache_dir()
    visual_cache_root = _visual_shared_cache_dir()
    from .semantic_pipeline import _semantic_cache_limit, _semantic_cache_root

    semantic_cache_disabled = os.environ.get(
        "VSR_DISABLE_SEMANTIC_SHARED_CACHE", ""
    ).strip().casefold() in {"1", "true", "yes", "on"}
    semantic_cache_disabled = semantic_cache_disabled or (
        os.environ.get("VSR_DISABLE_VISUAL_SHARED_CACHE", "").strip().casefold()
        in {"1", "true", "yes", "on"}
    )
    semantic_cache_root = None if semantic_cache_disabled else _semantic_cache_root()
    shared_cache_specs: dict[str, dict[str, Any]] = {}
    for name, path, limit in (
        ("asr", asr_cache_root, _asr_shared_cache_limit()),
        (
            "visual_frames",
            visual_cache_root / "frames" if visual_cache_root is not None else None,
            _visual_shared_cache_limit(),
        ),
        (
            "ocr",
            visual_cache_root / "ocr" if visual_cache_root is not None else None,
            _ocr_shared_cache_limit(),
        ),
        (
            "visual_surveys",
            visual_cache_root / "surveys" if visual_cache_root is not None else None,
            _visual_shared_cache_limit(),
        ),
        (
            "semantic",
            semantic_cache_root,
            _semantic_cache_limit(),
        ),
    ):
        if path is None:
            shared_cache_specs[name] = {
                "enabled": False,
                "path": None,
                "file_count": 0,
                "bytes": 0,
                "limit_bytes": limit,
            }
            continue
        cache_usage = _cache_directory_usage(path)
        cache_usage.update({"enabled": True, "limit_bytes": limit})
        shared_cache_specs[name] = cache_usage
    checks["shared_caches"] = {
        "status": "available",
        "value": {
            "entries": shared_cache_specs,
            "total_bytes": sum(int(item.get("bytes", 0)) for item in shared_cache_specs.values()),
            "total_files": sum(int(item.get("file_count", 0)) for item in shared_cache_specs.values()),
        },
        "fix": "Use the VSR_DISABLE_*_SHARED_CACHE or VSR_*_SHARED_CACHE_MAX_BYTES settings to bound local acceleration state.",
    }
    checks["network_policy"] = {
        "status": "available",
        "value": "offline" if offline else "explicitly enabled actions only",
        "fix": None,
    }
    return {"schema_version": "1.0", "checks": checks}


def plan_input(
    input_value: str | Path,
    *,
    output_root: Path | None = None,
    subtitles: Sequence[Path] = (),
    transcript: Path | None = None,
    preset: str = "strict",
    config_path: Path | None = None,
    vision_mode: str = "host-agent",
    offline: bool = True,
) -> dict[str, Any]:
    # ``auto`` historically meant “start the local Qwen VLM when available.”
    # Keep that spelling as a compatibility alias, but route it to the
    # offline Codex/subagent handoff instead of silently starting a model.
    if vision_mode == "auto":
        vision_mode = "host-agent"
    kind = classify_input(input_value)
    if kind == "remote_media":
        source_path = None
        source_name = Path(str(input_value).split("?", 1)[0]).stem or "remote-media"
    else:
        source_path = Path(input_value).expanduser().resolve(strict=True)
        source_name = source_path.stem
    config = load_config(config_path or preset)
    probe_summary: dict[str, Any] | None = None
    embedded_sources: list[str] = []
    if source_path and kind in {"video", "audio"} and shutil.which("ffprobe"):
        from .media_probe import probe_media
        from .subtitle_sources import discover_embedded_subtitle_tracks

        probe = probe_media(source_path)
        embedded_tracks = discover_embedded_subtitle_tracks(source_path, probe=probe)
        embedded_sources = [
            f"embedded:stream:{track.stream_index}:{track.codec_name}:{track.language or 'und'}"
            for track in embedded_tracks
            if track.supported
        ]
        probe_summary = {
            "duration_ms": probe.duration_ms,
            "streams": [
                {
                    "index": stream.index,
                    "type": stream.codec_type,
                    "codec": stream.codec_name,
                    "language": stream.language,
                    "disposition": dict(stream.disposition),
                }
                for stream in probe.streams
            ],
        }
    transcript_sources = [str(path) for path in subtitles]
    if transcript:
        transcript_sources.append(str(transcript))
    if kind == "transcript":
        transcript_sources.append(str(source_path))
    transcript_sources.extend(embedded_sources)
    needs_asr = kind in {"video", "audio"} and not transcript_sources
    needs_visual = kind == "video"
    if source_path is not None:
        output, _ = _single_output_dir(source_path, output_root)
    else:
        output = (output_root or default_output_root()) / safe_slug(source_name)
    duration_ms = (probe_summary or {}).get("duration_ms")
    interval = min(int(config.visual.survey_interval_seconds), 30)
    evidence_estimate = (
        0 if not needs_visual or not duration_ms else max(1, duration_ms // (interval * 1000) + 1)
    )
    asr_plan: dict[str, Any] = {
        "required": needs_asr,
        "chunk_seconds": config.asr.chunk_seconds,
        "overlap_seconds": config.asr.overlap_seconds,
        "estimated_chunks": (
            0
            if not needs_asr or not duration_ms
            else max(
                1,
                (int(duration_ms) + config.asr.chunk_seconds * 1000 - 1)
                // (config.asr.chunk_seconds * 1000),
            )
        ),
        "checkpointed": needs_asr,
    }
    if needs_asr:
        from .backend_probe import _whisper_runtime_probe

        runtime = _whisper_runtime_probe()
        asr_plan["runtime"] = runtime
        if runtime.get("cuda_visible") and not runtime.get("cublas_visible"):
            asr_plan["warning"] = (
                "NVIDIA GPU detected but cuBLAS is unavailable; large-v3 will use the "
                "auditable CPU/int8 fallback until the optional cuda extra is installed."
            )
    return {
        "schema_version": "1.0",
        "input_classification": kind,
        "probe": probe_summary,
        "likely_transcript_sources": transcript_sources
        or ["embedded subtitles", "local ASR fallback"],
        "planned_stages": [
            "identity",
            "probe",
            "transcript validation/selection",
            "ASR or selective repair if required",
            "timeline",
            "visual survey/packets",
            "metadata-first enrichment",
            "reconstruction",
            "audits",
            "atomic Markdown render",
        ],
        "strict_prerequisites": [
            "FFmpeg/FFprobe for media",
            "an explicitly installed, hash-verified faster-whisper large-v3 worker (or an injected ASR adapter) when ASR is required",
            "a bounded host-agent/Codex-subagent review for semantic visual claims",
        ],
        "asr_expected": needs_asr,
        "ocr_expected": needs_visual and bool(config.visual.ocr),
        "visual_review_expected": needs_visual and vision_mode in {"host-agent", "none"},
        "image_metadata_plan": "creation + deterministic for every generated image; read-before-reanalysis; semantic revisions when a Codex/subagent observer is available",
        "semantic_pending_possible": needs_visual and vision_mode in {"host-agent", "none"},
        "network_actions_requiring_permission": ["remote input download"]
        if kind == "remote_media"
        else (
            ["optional speech-worker installation and model fetch as separate setup actions"]
            if needs_asr
            else []
        ),
        "estimated_evidence_images": evidence_estimate,
        "estimated_disk_bytes": evidence_estimate * 1_500_000
        + (source_path.stat().st_size if source_path else 0),
        "asr_plan": asr_plan,
        "output_path": str(output),
        "output_contract": "exactly one Markdown file, inline images, hidden canonical state, zero HTML/UI",
        "no_full_processing_statement": "No model was downloaded, no external service was called, and no full media processing occurred.",
        "offline": offline,
    }


def _prepare_tree(project_dir: Path) -> None:
    for relative in (
        "evidence/full",
        "evidence/crops",
        ".state/transcript/original",
        ".state/timeline",
        ".state/vision/packets",
        ".state/candidates",
        ".state/cache",
        ".state/checkpoints",
    ):
        (project_dir / relative).mkdir(parents=True, exist_ok=True)


def _rotate_incomplete_visual_state(project_dir: Path) -> None:
    """Preserve stale visual artifacts before rebuilding a failed visual stage.

    A previous blocked run may leave packets and images whose IDs no longer
    belong to the current canonical project. Keeping them in a timestamped
    history prevents stale packet validation and orphan-image false positives
    while retaining the raw diagnostics for inspection.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history_root = project_dir / ".state" / "visual-history" / stamp
    moved = False
    for relative in ("evidence/full", "evidence/crops", ".state/vision/packets"):
        source = project_dir / relative
        files = [item for item in source.glob("*.png" if "evidence" in relative else "*.json")]
        if not files:
            continue
        destination = history_root / relative
        destination.mkdir(parents=True, exist_ok=True)
        for item in files:
            shutil.move(str(item), str(destination / item.name))
        moved = True
    if moved:
        (history_root / "README.txt").write_text(
            "Preserved artifacts from an incomplete visual rebuild; never treat as canonical evidence.\n",
            encoding="utf-8",
        )


def _preserve_sidecar(path: Path, project_dir: Path) -> str:
    digest = sha256_file(path)
    name = f"{digest[:16]}__{safe_slug(path.name)}"
    destination = project_dir / ".state" / "transcript" / "original" / name
    if not destination.exists():
        shutil.copy2(path, destination)
    return destination.relative_to(project_dir).as_posix()


def _parse_candidates(
    paths: Sequence[tuple[Path, str] | _CandidatePath],
    project_dir: Path,
    *,
    media_duration_ms: int | None = None,
    expected_language: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    from .subtitle_parse import parse_transcript
    from .subtitle_sources import rank_candidates

    candidates: list[dict[str, Any]] = []
    for index, path_item in enumerate(paths, 1):
        candidate_input = (
            path_item
            if isinstance(path_item, _CandidatePath)
            else _CandidatePath(path=path_item[0], source_type=path_item[1])
        )
        path = candidate_input.path
        source_type = candidate_input.source_type
        candidate_id = f"TC{index:06d}"
        segments = parse_transcript(path, source_candidate_id=candidate_id)
        segment_records: list[dict[str, Any]] = []
        for segment in segments:
            record = _record(segment)
            if candidate_input.source_track is not None:
                record["source_track"] = candidate_input.source_track
            segment_records.append(record)
        authorship = candidate_input.authorship or (
            "human" if source_type in {"user_human_transcript", "user_subtitle"} else "unknown"
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source_type": source_type,
                "origin": candidate_input.origin or path.name,
                "language": candidate_input.language,
                "authorship": authorship,
                "human_authored": True if authorship == "human" else None,
                "auto_generated": True if authorship == "auto_generated" else None,
                "raw_preservation_path": _preserve_sidecar(path, project_dir),
                "segments": segment_records,
            }
        )
    if not candidates:
        return [], [], "No transcript candidate was available."
    ranked = rank_candidates(
        candidates,
        media_duration_ms=media_duration_ms,
        expected_language=expected_language,
    )
    if not ranked:
        return candidates, [], "All transcript candidates failed validation."

    def add_diagnostics(candidate: dict[str, Any]) -> None:
        rank = next(
            (value for value in ranked if value.candidate_id == candidate["candidate_id"]),
            None,
        )
        candidate["quality_metrics"] = dict(rank.quality_metrics) if rank else {}
        candidate["issues"] = (
            [f"{issue.code}: {issue.message}" for issue in rank.issues]
            if rank
            else ["candidate not ranked"]
        )
        candidate["selection_score"] = float(rank.selection_score) if rank else None
        candidate["reliable_intervals"] = (
            [{"start_ms": start, "end_ms": end} for start, end in rank.reliable_intervals]
            if rank
            else []
        )
        candidate["unreliable_intervals"] = (
            [{"start_ms": start, "end_ms": end} for start, end in rank.unreliable_intervals]
            if rank
            else []
        )

    selected = next((item for item in ranked if item.quality_metrics.get("usable")), None)
    if selected is None:
        for candidate in candidates:
            add_diagnostics(candidate)
            candidate["decision_rationale"] = (
                "Preserved but rejected because validation did not classify it as usable."
            )
        return (
            candidates,
            [],
            "All transcript candidates were preserved but failed usability validation.",
        )
    selected_id = selected.candidate_id
    selected_candidate = next(item for item in candidates if item["candidate_id"] == selected_id)
    for candidate in candidates:
        add_diagnostics(candidate)
        candidate["decision_rationale"] = (
            "Selected as the highest-ranked usable evidence candidate."
            if candidate["candidate_id"] == selected_id
            else "Preserved but not selected; a higher-ranked candidate was available."
        )
    return (
        candidates,
        list(selected_candidate["segments"]),
        f"Selected {selected_id} ({selected_candidate['origin']}) after candidate validation and provenance ranking.",
    )


def _asr_segments(
    adapter: ASRAdapter,
    source: Path,
    language: str | None,
    *,
    duration_ms: int | None = None,
    checkpoint_dir: Path | None = None,
    chunk_ms: int = 900_000,
    overlap_ms: int = 15_000,
    media_sha256: str | None = None,
    shared_cache_dir: Path | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    from .ids import deterministic_id, sequential_id

    if duration_ms is not None and duration_ms > 0 and checkpoint_dir is not None:
        from .whisper_adapter import transcribe_checkpointed_chunks

        result: Iterable[Any] = transcribe_checkpointed_chunks(
            cast(Any, adapter),
            source,
            duration_ms=duration_ms,
            checkpoint_dir=checkpoint_dir,
            chunk_ms=chunk_ms,
            overlap_ms=overlap_ms,
            language=language,
            media_sha256=media_sha256,
            shared_cache_dir=shared_cache_dir,
            progress_callback=progress_callback,
        )
    else:
        result = adapter.transcribe(source, language=language, word_timestamps=True)
    segments: list[dict[str, Any]] = []
    for index, value in enumerate(result, 1):
        item = _record(value)
        # Normalize adapter-local IDs at the canonical boundary.  Production
        # Markdown/state references use the visible T/W namespaces even when a
        # CI adapter or faster-whisper emits opaque IDs of its own.
        item["segment_id"] = sequential_id("transcript", index)
        item.setdefault("raw_text", item.get("text", ""))
        item.setdefault("normalized_text", item["raw_text"])
        item.setdefault("timing_provenance", "model-independent-test-adapter")
        item.setdefault("verification_status", "unverified")
        item.setdefault("uncertainty_items", [])
        item.setdefault("substantive", bool(str(item["raw_text"]).strip()))
        words = item.get("words")
        if isinstance(words, list):
            normalized_words: list[dict[str, Any]] = []
            for word_index, raw_word in enumerate(words):
                if not isinstance(raw_word, dict):
                    normalized_words.append(raw_word)
                    continue
                word = dict(raw_word)
                word["word_id"] = deterministic_id(
                    "word",
                    item["segment_id"],
                    word_index,
                    word.get("start_ms"),
                    word.get("end_ms"),
                    word.get("text", word.get("word", "")),
                )
                normalized_words.append(word)
            item["words"] = normalized_words
        segments.append(item)
    return segments


def _infer_primary_language(segments: Sequence[Mapping[str, Any]]) -> str | None:
    """Infer a conservative primary language from adapter-emitted segment labels.

    Automatic ASR often reports the detected language on each segment rather
    than on the candidate envelope.  Use substantive text length as a stable
    weight, require a clear majority, and leave genuinely mixed/ambiguous
    material as ``None`` so the canonical project can publish ``und`` instead
    of inventing a global language.  Filipino's ``fil`` and Whisper's ``tl``
    labels are normalized to the same ISO-639-1-style ``tl`` value.
    """

    weights: dict[str, int] = {}

    def normalize(value: Any) -> str | None:
        raw = str(value or "").strip().casefold().replace("_", "-")
        if not raw or raw in {"und", "unknown", "auto", "none", "null"}:
            return None
        code = raw.split("-", 1)[0]
        if code in {"fil", "tl"}:
            return "tl"
        return code if code.isalpha() and 2 <= len(code) <= 3 else None

    for segment in segments:
        text = str(segment.get("normalized_text") or segment.get("raw_text") or "").strip()
        segment_weight = max(1, len(text)) if text else 0
        language = normalize(segment.get("language"))
        if language is not None and segment_weight:
            weights[language] = weights.get(language, 0) + segment_weight
            continue
        words = segment.get("words")
        if not isinstance(words, list):
            continue
        for word in words:
            if not isinstance(word, Mapping):
                continue
            word_language = normalize(word.get("language"))
            word_text = str(word.get("text") or word.get("word") or "").strip()
            if word_language is not None and word_text:
                weights[word_language] = weights.get(word_language, 0) + len(word_text)
    if not weights:
        return None
    ordered = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    winner, winner_weight = ordered[0]
    total_weight = sum(weights.values())
    runner_up_weight = ordered[1][1] if len(ordered) > 1 else 0
    if winner_weight * 100 < total_weight * 60 or runner_up_weight * 100 >= winner_weight * 70:
        return None
    return winner


def _record_asr_candidate(
    candidates: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    adapter: ASRAdapter,
    *,
    language: str | None,
    media_duration_ms: int | None,
) -> None:
    from .subtitle_sources import rank_candidates

    candidate_id = f"TC{len(candidates) + 1:06d}"
    backend = str(getattr(adapter, "backend_name", adapter.__class__.__name__))
    segment_offset = sum(len(candidate.get("segments", [])) for candidate in candidates)
    from .ids import deterministic_id, sequential_id

    for segment_index, segment in enumerate(segments, 1):
        segment["segment_id"] = sequential_id("transcript", segment_offset + segment_index)
        segment["source_candidate_id"] = candidate_id
        segment["source_track"] = f"local ASR backend={backend}"
        for word_index, word in enumerate(segment.get("words", []) or []):
            if isinstance(word, dict):
                word["word_id"] = deterministic_id(
                    "word",
                    segment["segment_id"],
                    word_index,
                    word.get("start_ms"),
                    word.get("end_ms"),
                    word.get("text", word.get("word", "")),
                )
    candidate: dict[str, Any] = {
        "candidate_id": candidate_id,
        "source_type": "local_asr",
        "origin": backend,
        # A requested language is an ASR hint, not evidence that an empty
        # candidate contains speech in that language. Keep empty results
        # explicitly unknown so canonical language reporting cannot turn a
        # no-speech pass into an unsupported claim.
        "language": language if segments else None,
        "authorship": "auto_generated",
        "human_authored": False,
        "auto_generated": True,
        "raw_preservation_path": None,
        "segments": segments,
    }
    ranked = rank_candidates(
        [candidate],
        media_duration_ms=media_duration_ms,
        expected_language=language,
    )[0]
    candidate.update(
        {
            "quality_metrics": dict(ranked.quality_metrics),
            "reliable_intervals": [
                {"start_ms": start, "end_ms": end} for start, end in ranked.reliable_intervals
            ],
            "unreliable_intervals": [
                {"start_ms": start, "end_ms": end} for start, end in ranked.unreliable_intervals
            ],
            "issues": [f"{issue.code}: {issue.message}" for issue in ranked.issues],
            "selection_score": ranked.selection_score,
            "selected_intervals": [
                {"start_ms": start, "end_ms": end} for start, end in ranked.reliable_intervals
            ],
            "decision_rationale": "Selected because no usable supplied or embedded transcript candidate was available.",
        }
    )
    candidates.append(candidate)


def _select_asr_candidate_set(
    candidates: list[dict[str, Any]],
    *,
    media_duration_ms: int | None,
    expected_language: str | None,
    preferred_backend: str | None = None,
    prefer_local_asr: bool = False,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    """Rank independent ASR results and disclose exact candidate disagreements."""

    from .audit import high_impact_discrepancies
    from .subtitle_sources import rank_candidates

    ranked = rank_candidates(
        candidates,
        media_duration_ms=media_duration_ms,
        expected_language=expected_language,
    )
    all_local_asr = all(
        str(candidate.get("source_type", "")).casefold() == "local_asr"
        for candidate in candidates
    )
    candidate_label = "local ASR candidate(s)" if all_local_asr else "transcript candidate(s)"
    selected = None
    if prefer_local_asr:
        selected = next(
            (
                item
                for item in ranked
                if item.quality_metrics.get("usable")
                and str(item.source_type).casefold() == "local_asr"
            ),
            None,
        )
    if preferred_backend:
        selected = selected or next(
            (
                item
                for item in ranked
                if item.quality_metrics.get("usable")
                and str(item.origin).casefold() == preferred_backend.casefold()
            ),
            None,
        )
    if selected is None:
        selected = next((item for item in ranked if item.quality_metrics.get("usable")), None)
    if selected is None:
        raise ValueError(f"Every {candidate_label} failed transcript usability validation")
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    for rank in ranked:
        candidate = by_id[rank.candidate_id]
        candidate.update(
            {
                "quality_metrics": dict(rank.quality_metrics),
                "reliable_intervals": [
                    {"start_ms": start, "end_ms": end} for start, end in rank.reliable_intervals
                ],
                "unreliable_intervals": [
                    {"start_ms": start, "end_ms": end} for start, end in rank.unreliable_intervals
                ],
                "issues": [f"{issue.code}: {issue.message}" for issue in rank.issues],
                "selection_score": rank.selection_score,
                "selected_intervals": [
                    {"start_ms": start, "end_ms": end}
                    for start, end in (rank.reliable_intervals if rank is selected else [])
                ],
                "decision_rationale": (
                    (
                        "Selected as the explicitly preferred usable local ASR backend."
                        if preferred_backend
                        else "Selected as the explicit local ASR authority while comparing supplied transcript candidates."
                        if prefer_local_asr
                        else (
                            "Selected as the highest-ranked usable independent local ASR candidate."
                            if all_local_asr
                            else "Selected as the highest-ranked usable independent transcript candidate."
                        )
                    )
                    if rank is selected
                    else "Preserved as an independent corroboration candidate; not selected."
                ),
            }
        )

    selected_candidate = by_id[selected.candidate_id]
    moss_candidate = next(
        (
            candidate
            for candidate in candidates
            if candidate is not selected_candidate
            and candidate.get("origin") == "moss-transcribe-diarize"
        ),
        None,
    )
    if moss_candidate is not None:
        for segment in selected_candidate["segments"]:
            start_ms = segment.get("start_ms")
            end_ms = segment.get("end_ms")
            if start_ms is None or end_ms is None:
                continue
            overlapping_labels = {
                str(turn.get("speaker_label"))
                for turn in moss_candidate.get("segments", [])
                if turn.get("speaker_label")
                and turn.get("start_ms") is not None
                and turn.get("end_ms") is not None
                and int(turn["start_ms"]) < int(end_ms)
                and int(turn["end_ms"]) > int(start_ms)
            }
            if len(overlapping_labels) == 1:
                segment["speaker_label"] = next(iter(overlapping_labels))
            elif len(overlapping_labels) > 1:
                uncertainties = list(segment.get("uncertainty_items", []))
                if "uncertain_speaker_boundary" not in uncertainties:
                    uncertainties.append("uncertain_speaker_boundary")
                segment["uncertainty_items"] = uncertainties
        selected_candidate.setdefault("issues", []).append(
            "speaker_crosscheck: anonymous labels were assigned only where all overlapping "
            "MOSS turns agreed; multi-speaker overlaps remain uncertain"
        )
    selected_text = " ".join(
        str(segment.get("normalized_text") or segment.get("raw_text") or "")
        for segment in selected_candidate["segments"]
    ).strip()
    disagreements: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate is selected_candidate:
            continue
        candidate_text = " ".join(
            str(segment.get("normalized_text") or segment.get("raw_text") or "")
            for segment in candidate["segments"]
        ).strip()
        ratio = SequenceMatcher(None, selected_text.casefold(), candidate_text.casefold()).ratio()
        discrepancy = high_impact_discrepancies(selected_text, candidate_text)
        if ratio >= 0.995 and not discrepancy["missing"] and not discrepancy["added"]:
            continue
        record = {
            "selected_candidate_id": selected.candidate_id,
            "other_candidate_id": candidate["candidate_id"],
            "ordered_text_similarity": round(ratio, 6),
            "selected_high_impact_missing_from_other": discrepancy["missing"],
            "other_high_impact_added": discrepancy["added"],
        }
        disagreements.append(record)
        summary = "candidate_disagreement: " + json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        selected_candidate.setdefault("issues", []).append(summary)
        candidate.setdefault("issues", []).append(summary)
        if discrepancy["missing"] or discrepancy["added"]:
            high_impact_summary = "high_impact_candidate_disagreement: " + json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            selected_candidate["issues"].append(high_impact_summary)
            candidate["issues"].append(high_impact_summary)
    decision = (
        f"Selected {selected.candidate_id} ({selected_candidate['origin']}) after independently "
        f"validating {len(candidates)} {candidate_label}."
    )
    if prefer_local_asr and str(selected.source_type).casefold() == "local_asr":
        decision += " Explicit local ASR authority applied while comparing supplied transcript candidates."
    elif preferred_backend and str(selected.origin).casefold() == preferred_backend.casefold():
        decision += f" Explicit preference applied for backend={preferred_backend}."
    if disagreements:
        decision += f" Preserved {len(disagreements)} material candidate disagreement(s)."
    return list(selected_candidate["segments"]), decision, disagreements


def _developer_worker_path(
    environment_name: str, relative: Sequence[str], *, managed_name: str
) -> Path | None:
    configured = os.environ.get(environment_name)
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.is_file() else None
    from .worker_store import worker_python

    managed = worker_python(managed_name)
    if managed.is_file():
        return managed
    repository_candidate = Path(__file__).resolve().parents[2].joinpath(*relative)
    return repository_candidate if repository_candidate.is_file() else None


_FASTER_WHISPER_REQUIRED_FILES = (
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
)


def _configured_faster_whisper_model() -> Path | None:
    """Resolve an explicit local large-v3 directory without copying weights.

    Hugging Face snapshots and other offline model stores can be complete and
    usable without carrying this project's manifest.  The path is opt-in,
    validated for the exact files faster-whisper needs, and never downloaded or
    searched for implicitly.
    """

    configured = os.environ.get("VSR_FASTER_WHISPER_LARGE_V3_PATH", "").strip()
    if not configured:
        return None
    directory = Path(configured).expanduser().resolve()
    if not directory.is_dir():
        LOGGER.warning("Configured large-v3 directory does not exist: %s", directory)
        return None
    missing = [name for name in _FASTER_WHISPER_REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        LOGGER.warning(
            "Configured large-v3 directory is incomplete (%s missing): %s",
            ", ".join(missing),
            directory,
        )
        return None
    return directory


def _faster_whisper_model_identity(
    directory: Path, *, revision: str | None = None
) -> tuple[str | None, str | None]:
    """Return a cheap model revision/stat token for ASR checkpoint identity.

    Managed model directories provide a verified manifest revision through
    ``verify_model``. Explicit external directories may not carry that
    manifest, so include a deterministic signature of the required-file
    ``size``/``mtime_ns`` values as a replacement detector without hashing a
    multi-gigabyte ``model.bin`` on every run. The token is cache metadata only;
    model verification remains the authority before automatic resolution.
    """

    resolved_revision = str(revision).strip() if revision else None
    if resolved_revision is None:
        manifest_path = directory / "model-manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, Mapping):
            raw_revision = manifest.get("revision")
            if raw_revision:
                resolved_revision = str(raw_revision).strip() or None

    stat_parts: list[str] = []
    for filename in _FASTER_WHISPER_REQUIRED_FILES:
        path = directory / filename
        try:
            stat = path.stat()
        except OSError:
            continue
        stat_parts.append(f"{filename}:{stat.st_size}:{stat.st_mtime_ns}")
    signature = None
    if stat_parts:
        signature = hashlib.sha256("|".join(stat_parts).encode("utf-8")).hexdigest()
    return resolved_revision, signature


def _faster_whisper_inference_options() -> dict[str, Any]:
    """Read the explicit experimental batched-decoder opt-in.

    Standard faster-whisper decoding remains the accuracy-first default.  The
    batched path is useful for throughput experiments on long media, but its
    different scheduling can alter segment boundaries, so it is only enabled by
    an explicit environment setting and is marked for transcript review by the
    adapter metadata.
    """

    raw_mode = os.environ.get("VSR_FASTER_WHISPER_INFERENCE_MODE", "").strip().casefold()
    batched_flag = os.environ.get("VSR_FASTER_WHISPER_BATCHED", "").strip().casefold()
    if not raw_mode and batched_flag in {"1", "true", "yes", "on"}:
        raw_mode = "batched"
    if raw_mode not in {"", "standard", "batched"}:
        LOGGER.warning(
            "Ignoring unsupported VSR_FASTER_WHISPER_INFERENCE_MODE=%r; using standard",
            raw_mode,
        )
        raw_mode = ""
    mode = raw_mode or "standard"
    raw_batch_size = os.environ.get("VSR_FASTER_WHISPER_BATCH_SIZE", "8").strip()
    try:
        batch_size = int(raw_batch_size)
    except ValueError:
        LOGGER.warning(
            "Ignoring invalid VSR_FASTER_WHISPER_BATCH_SIZE=%r; using 8", raw_batch_size
        )
        batch_size = 8
    if not 1 <= batch_size <= 64:
        LOGGER.warning(
            "Clamping VSR_FASTER_WHISPER_BATCH_SIZE=%s to the supported range 1..64",
            batch_size,
        )
        batch_size = min(64, max(1, batch_size))
    return {"inference_mode": mode, "batch_size": batch_size}


_FASTER_WHISPER_COMPUTE_TYPES = frozenset(
    {"default", "float16", "int8", "int8_float16", "int8_bfloat16"}
)


def _faster_whisper_compute_type() -> str:
    """Choose the CTranslate2 compute type for automatically resolved Whisper.

    CUDA hosts load float16 weights and CPU-only hosts stay on int8 so the
    automatic policy remains auditable without an explicit decision.
    ``VSR_FASTER_WHISPER_COMPUTE_TYPE`` mirrors the model-dependent test knob
    and opts into one of the supported faster-whisper values; anything else
    fails closed to the host policy instead of surprising a run with an
    untested precision mode.
    """

    fallback = "float16" if shutil.which("nvidia-smi") else "int8"
    override = os.environ.get("VSR_FASTER_WHISPER_COMPUTE_TYPE", "").strip().casefold()
    if not override:
        return fallback
    if override not in _FASTER_WHISPER_COMPUTE_TYPES:
        LOGGER.warning(
            "Ignoring unsupported VSR_FASTER_WHISPER_COMPUTE_TYPE=%r; using %s",
            override,
            fallback,
        )
        return fallback
    return override


def _auto_asr_adapters(
    *,
    language: str | None = None,
    compare_candidates: bool = True,
    duration_ms: int | None = None,
) -> list[Any]:
    """Resolve already-installed local speech backends in an explicit quality order.

    Strict/default mode keeps the independent candidate ensemble. When candidate
    comparison is explicitly disabled, probe and construct only the first usable
    backend in that order; resolving unused workers would add manifest/worker
    startup work without changing the selected result.
    """

    from .model_store import verify_model

    # New/default workflows use the user's preferred Whisper large-v3 path.
    # Qwen/MOSS/forced-alignment workers remain available only when a caller
    # explicitly opts into the legacy local-model ensemble; they are never
    # probed, loaded, or selected by an ordinary run.
    legacy_local_models = str(
        os.environ.get("VSR_ALLOW_LEGACY_LOCAL_MODELS", "")
    ).casefold() in {"1", "true", "yes", "on"}
    if not legacy_local_models:
        compare_candidates = False

    prefer_whisper = (
        str(os.environ.get("VSR_PREFER_WHISPER", "")).casefold() in {"1", "true", "yes"}
        or (language is not None and language.casefold().split("-")[0] in {"fil", "tl"})
    )

    if not compare_candidates:
        def resolve_whisper() -> Any | None:
            configured_model = _configured_faster_whisper_model()
            status = (
                {"offline_ready": True, "directory": str(configured_model)}
                if configured_model is not None
                else verify_model("faster-whisper-large-v3")
            )
            if (
                status.get("offline_ready")
                and importlib.util.find_spec("faster_whisper") is not None
            ):
                from .whisper_adapter import FasterWhisperAdapter

                model_path = Path(str(status["directory"]))
                model_revision, model_signature = _faster_whisper_model_identity(
                    model_path,
                    revision=(str(status["revision"]) if status.get("revision") else None),
                )
                return FasterWhisperAdapter(
                    model=model_path,
                    model_revision=model_revision,
                    model_signature=model_signature,
                    device="cuda" if shutil.which("nvidia-smi") else "cpu",
                    compute_type=_faster_whisper_compute_type(),
                    cpu_threads=_asr_cpu_threads(),
                    num_workers=_faster_whisper_num_workers(duration_ms=duration_ms),
                    **_faster_whisper_inference_options(),
                    allow_model_download=False,
                    allow_cpu_fallback=True,
                )
            return None

        def resolve_qwen() -> Any | None:
            worker = _developer_worker_path(
                "VSR_QWEN_SPEECH_PYTHON",
                (".artifacts", "workers", "qwen-asr", "Scripts", "python.exe"),
                managed_name="qwen-speech",
            )
            if worker is None:
                return None
            status = verify_model("qwen3-asr-1.7b")
            aligner = verify_model("qwen3-forced-aligner-0.6b")
            if status.get("offline_ready"):
                from .qwen_asr_adapter import Qwen3ASRAdapter

                return Qwen3ASRAdapter(
                    worker_python=worker,
                    aligner_name=(
                        "qwen3-forced-aligner-0.6b"
                        if aligner.get("offline_ready")
                        else None
                    ),
                )
            return None

        def resolve_moss() -> Any | None:
            worker = _developer_worker_path(
                "VSR_MOSS_SPEECH_PYTHON",
                (".artifacts", "workers", "moss", "Scripts", "python.exe"),
                managed_name="moss-speech",
            )
            if worker is None:
                return None
            status = verify_model("moss-transcribe-diarize-0.9b")
            if status.get("offline_ready"):
                from .moss_adapter import MossTranscribeDiarizeAdapter

                return MossTranscribeDiarizeAdapter(worker_python=worker)
            return None

        resolvers = (
            (resolve_whisper,)
            if not legacy_local_models
            else (
                (resolve_whisper, resolve_qwen, resolve_moss)
                if prefer_whisper
                else (resolve_qwen, resolve_whisper, resolve_moss)
            )
        )
        for resolve in resolvers:
            adapter = resolve()
            if adapter is not None:
                return [adapter]
        return []

    adapters: list[Any] = []
    whisper_adapter: Any | None = None
    qwen_adapter: Any | None = None
    moss_adapter: Any | None = None
    qwen_worker = _developer_worker_path(
        "VSR_QWEN_SPEECH_PYTHON",
        (".artifacts", "workers", "qwen-asr", "Scripts", "python.exe"),
        managed_name="qwen-speech",
    )
    if qwen_worker is not None:
        qwen_status = verify_model("qwen3-asr-1.7b")
        aligner_status = verify_model("qwen3-forced-aligner-0.6b")
    else:
        qwen_status = {"offline_ready": False}
        aligner_status = {"offline_ready": False}
    if qwen_status.get("offline_ready"):
        from .qwen_asr_adapter import Qwen3ASRAdapter

        qwen_adapter = Qwen3ASRAdapter(
                worker_python=qwen_worker,
                aligner_name=(
                    "qwen3-forced-aligner-0.6b" if aligner_status.get("offline_ready") else None
                ),
            )
    moss_worker = _developer_worker_path(
        "VSR_MOSS_SPEECH_PYTHON",
        (".artifacts", "workers", "moss", "Scripts", "python.exe"),
        managed_name="moss-speech",
    )
    moss_status = (
        verify_model("moss-transcribe-diarize-0.9b")
        if moss_worker is not None
        else {"offline_ready": False}
    )
    if moss_status.get("offline_ready"):
        from .moss_adapter import MossTranscribeDiarizeAdapter

        moss_adapter = MossTranscribeDiarizeAdapter(worker_python=moss_worker)
    configured_whisper_model = _configured_faster_whisper_model()
    whisper_status = (
        {"offline_ready": True, "directory": str(configured_whisper_model)}
        if configured_whisper_model is not None
        else verify_model("faster-whisper-large-v3")
    )
    if (
        whisper_status.get("offline_ready")
        and importlib.util.find_spec("faster_whisper") is not None
    ):
        from .whisper_adapter import FasterWhisperAdapter

        model_path = Path(str(whisper_status["directory"]))
        model_revision, model_signature = _faster_whisper_model_identity(
            model_path,
            revision=(
                str(whisper_status["revision"])
                if whisper_status.get("revision")
                else None
            ),
        )
        whisper_adapter = FasterWhisperAdapter(
                model=model_path,
                model_revision=model_revision,
                model_signature=model_signature,
                device="cuda" if shutil.which("nvidia-smi") else "cpu",
                compute_type=_faster_whisper_compute_type(),
                cpu_threads=_asr_cpu_threads(),
                num_workers=_faster_whisper_num_workers(duration_ms=duration_ms),
                **_faster_whisper_inference_options(),
                allow_model_download=False,
                allow_cpu_fallback=True,
            )
    if prefer_whisper:
        adapters.extend(item for item in (whisper_adapter, qwen_adapter, moss_adapter) if item is not None)
    else:
        adapters.extend(item for item in (qwen_adapter, whisper_adapter, moss_adapter) if item is not None)
    return adapters


def _release_asr_adapter(adapter: Any) -> None:
    """Release native ASR state before loading an independent candidate."""
    close = getattr(adapter, "close", None)
    if callable(close):
        close()
    # Native runtimes may retain allocator references after close(). Keep the
    # cleanup local and best-effort; failure to release is diagnostic, not a
    # reason to discard an already-produced candidate.
    gc.collect()


def _auto_ocr_adapter() -> Any | None:
    """Prefer verified PP-OCRv5, then use the installed Tesseract executable."""

    paddle_worker = _developer_worker_path(
        "VSR_PADDLE_OCR_PYTHON",
        (".artifacts", "workers", "paddleocr", "Scripts", "python.exe"),
        managed_name="paddle-ocr",
    )
    if paddle_worker is not None:
        from .model_store import verify_model

        detector = verify_model("pp-ocrv5-server-det")
        recognizer = verify_model("pp-ocrv5-server-rec")
        if detector.get("offline_ready") and recognizer.get("offline_ready"):
            from .paddle_ocr_adapter import PaddleOCRV5Adapter

            candidate = PaddleOCRV5Adapter(worker_python=paddle_worker)
            if candidate.available():
                return candidate
    from .ocr import TesseractOCRAdapter

    fallback = TesseractOCRAdapter()
    return fallback if fallback.available() else None


def _build_blocks(segments: Sequence[dict[str, Any]], fidelity_mode: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, 1):
        text = str(
            segment.get("human_verified_text")
            or segment.get("repaired_text")
            or segment.get("normalized_text")
            or segment.get("raw_text")
            or ""
        )
        blocks.append(
            {
                "block_id": f"B{index:06d}",
                "chapter_id": "C001",
                "start_ms": segment.get("start_ms"),
                "end_ms": segment.get("end_ms"),
                "speaker": segment.get("speaker_label") or "Speaker 1",
                "spoken_text": text,
                "visual_description": None,
                "on_screen_text": [],
                "relevant_non_speech_audio": [],
                "frame_ids": [],
                "transcript_segment_ids": [segment.get("segment_id")],
                "visual_event_ids": [],
                "image_claim_ids": [],
                "metadata_revision_ids": [],
                "metadata_sufficiency_decision_ids": [],
                "transformation_ids": [],
                "fidelity_mode": fidelity_mode,
                "confidence": segment.get("confidence")
                if segment.get("confidence") is not None
                else 1.0,
                "verification_status": segment.get("verification_status") or "unverified",
                "uncertainty": list(segment.get("uncertainty_items", [])),
                "residual_source_text": None,
            }
        )
    return blocks


def _clip_source_transcript_bounds_to_media(
    segments: Sequence[dict[str, Any]], duration_ms: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bound subtitle-derived cues to playable media while retaining provenance.

    Subtitle files occasionally retain a final cue after a video was trimmed.
    The media probe is authoritative for the playable range, so a cue that
    starts inside the media but ends beyond it is clipped in the canonical
    transcript. The original candidate/source bytes remain available through
    the transcript candidate and sidecar path; each adjustment becomes a
    non-blocking review item. ASR/model timings and cues wholly outside the
    media are left untouched and continue to block validation.
    """

    if duration_ms is None or duration_ms < 0:
        return [dict(segment) for segment in segments], []
    clipped: list[dict[str, Any]] = []
    adjustments: list[dict[str, Any]] = []
    for segment in segments:
        item = dict(segment)
        provenance = str(item.get("timing_provenance") or "")
        start = item.get("start_ms")
        end = item.get("end_ms")
        if (
            provenance.startswith("source_")
            and isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < duration_ms < end
        ):
            uncertainty_items = list(item.get("uncertainty_items") or [])
            uncertainty_items.append(
                f"Source cue end {end}ms exceeded media duration {duration_ms}ms; "
                "canonical end clipped to playable media."
            )
            item["end_ms"] = duration_ms
            item["timing_provenance"] = f"{provenance}_clipped_to_media"
            item["uncertainty_items"] = uncertainty_items
            adjustments.append(
                {
                    "segment_id": str(item.get("segment_id") or ""),
                    "start_ms": start,
                    "original_end_ms": end,
                    "clipped_end_ms": duration_ms,
                }
            )
        clipped.append(item)
    return clipped, adjustments


def _can_use_visual_only_fallback(
    *,
    kind: str,
    segments: Sequence[dict[str, Any]],
    asr_completed_without_segments: bool,
    asr_produced_segments: bool,
    asr_had_failure: bool,
) -> bool:
    """Allow a bounded visual reconstruction when ASR proved no speech.

    An empty result from every completed ASR pass is different from an
    unavailable/crashed backend or a rejected candidate that contained
    speech. The former can describe measured frames without inventing
    dialogue; the latter remains a hard prerequisite failure. This fallback is
    intentionally video-only because an audio-only output has no visual
    evidence to reconstruct.
    """

    return bool(
        kind == "video"
        and not segments
        and asr_completed_without_segments
        and not asr_produced_segments
        and not asr_had_failure
    )


def _derive_chapters(
    blocks: Sequence[dict[str, Any]], duration_ms: int | None
) -> list[dict[str, Any]]:
    if not blocks:
        return [
            {
                "chapter_id": "C001",
                "title": "Incomplete reconstruction (navigational)",
                "start_ms": 0,
                "end_ms": duration_ms or 0,
                "block_ids": [],
                "source_authored": False,
            }
        ]
    chapters: list[dict[str, Any]] = []
    for block in blocks:
        start = block.get("start_ms")
        bucket = 0 if start is None else int(start) // 600_000
        while len(chapters) <= bucket:
            index = len(chapters)
            chapters.append(
                {
                    "chapter_id": f"C{index + 1:03d}",
                    "title": f"Navigational chapter {index + 1}",
                    "start_ms": index * 600_000 if duration_ms is not None else None,
                    "end_ms": min((index + 1) * 600_000, duration_ms)
                    if duration_ms is not None
                    else None,
                    "block_ids": [],
                    "source_authored": False,
                }
            )
        chapter = chapters[bucket]
        block["chapter_id"] = chapter["chapter_id"]
        chapter["block_ids"].append(block["block_id"])
    return chapters


def _creation_payload(
    extracted: Any,
    *,
    media_id: str,
    relative_path: str,
    block_id: str,
    segment_ids: list[str],
    revision_id: str,
    revision_digest: str,
) -> dict[str, Any]:
    from .image_metadata import normalized_pixel_hash

    return {
        "schema_name": "video-script-reconstructor.evidence-image",
        "schema_version": "1.0",
        "image": {
            "image_id": extracted.frame_id,
            "media_id": media_id,
            "parent_full_frame_id": None,
            "origin": "extracted_full_frame",
            "derivation": {"method": "ffmpeg-decoded-frame", "transformation_ids": []},
            "requested_ms": extracted.requested_ms,
            "actual_ms": extracted.actual_ms,
            "pts": {
                "value": extracted.raw_pts,
                "time_base": extracted.time_base,
                "source": extracted.timestamp_source,
            },
            "role": "context",
            "width": extracted.width,
            "height": extracted.height,
            "orientation": 0,
            "crop_xywh": None,
            "pixel_hash": {
                "algorithm": "sha256-rgba8-srgb-v1",
                "value": normalized_pixel_hash(extracted.path),
            },
        },
        "links": {
            "chapter_ids": [],
            "block_ids": [block_id],
            "segment_ids": segment_ids,
            "visual_event_ids": [],
            "ocr_observation_ids": [],
            "review_item_ids": [],
            "neighbor_image_ids": [],
            "candidate_ids": [],
        },
        "knowledge": {
            "selection_reason": "Time-aligned full frame retained to preserve visual context for the reconstruction block.",
            "why_it_matters": "Provides observable visual evidence aligned to the spoken interval.",
            "current_factual_description": None,
            "claims": [],
            "supported_claim_ids": [],
            "disputed_claim_ids": [],
            "rejected_claim_ids": [],
            "unresolved_claim_ids": [],
            "explicit_unknowns": ["Semantic visible content has not yet been analyzed."],
            "statements_not_inferred": [
                "No identity, intent, action, or hidden state is inferred from this unobserved frame."
            ],
            "before_action_after": None,
        },
        "analysis": {
            "enrichment_level": "creation",
            "semantic_status": "unobserved",
            "sufficiency": {
                "status": "semantic_observer_unavailable",
                "evaluated_question_ids": [],
                "answered_question_ids": [],
                "unanswered_questions": [
                    "What meaningful visible state or text supports this block?"
                ],
                "recommended_next_action": "Inspect the full frame with its time-aligned transcript and adjacent frames.",
            },
            "latest_revision_id": revision_id,
            "revision_number": 1,
            "observation_history": [],
        },
        "integrity": {
            "previous_revision_id": None,
            "previous_payload_digest": None,
            "payload_digest_algorithm": "sha256-canonical-json-with-digest-omitted-v1",
            "payload_digest": "0" * 64,
            "canonical_revision_locator": f".state/vision/image-observations.json#{revision_id}",
            "canonical_revision_digest": revision_digest,
        },
    }


def _bounded_visual_block_points(
    blocks: Sequence[dict[str, Any]], *, duration_ms: int, spacing_ms: int
) -> list[tuple[int, str]]:
    """Return first/last and spaced transcript midpoints for visual context."""
    if spacing_ms <= 0:
        raise InputError("visual block spacing must be positive")
    candidates: list[tuple[int, str]] = []
    for block in blocks:
        start = block.get("start_ms")
        end = block.get("end_ms")
        if start is None:
            continue
        point = int(start) if end is None else int(start + max(0, int(end) - int(start)) // 2)
        # Keep tail context inside the last measurable frame window. A final
        # subtitle cue can extend past the probed media duration; requesting
        # exactly ``duration_ms - 1`` then gives FFmpeg no decodable frame on
        # some VFR/H.264 files. This guard matches the bounded subtitle/timeline
        # tolerance and does not estimate frame timing.
        candidates.append(
            (
                min(point, max(duration_ms - _VISUAL_TAIL_GUARD_MS, 0)),
                str(block["block_id"]),
            )
        )
    if not candidates:
        return []
    ordered = sorted(candidates)
    kept: list[tuple[int, str]] = [ordered[0]]
    for item in ordered[1:-1]:
        if item[0] - kept[-1][0] >= spacing_ms:
            kept.append(item)
    if ordered[-1] != kept[-1]:
        kept.append(ordered[-1])
    return kept


_MAX_SHARED_SURVEY_EMISSION_TIMES = 256
_VISUAL_TAIL_GUARD_MS = 250
# Windows FFmpeg builds can reject an otherwise valid combined survey when the
# periodic ``select`` expression grows too large for the filter graph allocator.
# Keep the first attempt generous, then use deterministic smaller schedules
# before falling all the way back to the guarded per-request extractor.  This
# is an acceleration hint only: every omitted timestamp remains in the
# candidate set and is recovered by exact extraction below.
_SHARED_SURVEY_RETRY_CAPS = (128, 96, 64, 32, 16, 8, 1)


def _bounded_shared_survey_emission_times(
    requested_times_ms: Sequence[int],
    *,
    max_count: int = _MAX_SHARED_SURVEY_EMISSION_TIMES,
) -> tuple[int, ...]:
    """Bound timestamps embedded in one FFmpeg filter argument.

    The shared survey output is only an acceleration cache.  Structural and
    contextual candidates are still retained independently, and any omitted
    timestamp is extracted through the exact guarded frame path below.  Keep
    the first/last timestamps and choose the remaining points at deterministic
    evenly spaced indices so a long subtitle scaffold cannot exceed Windows'
    command-line limit while preserving temporal coverage.
    """

    if max_count <= 0:
        raise InputError("shared survey emission bound must be positive")
    ordered = tuple(sorted(set(int(value) for value in requested_times_ms)))
    if len(ordered) <= max_count:
        return ordered
    if max_count == 1:
        return (ordered[0],)
    last_index = len(ordered) - 1
    selected = tuple(
        ordered[round(index * last_index / (max_count - 1))]
        for index in range(max_count)
    )
    return tuple(dict.fromkeys(selected))


def _bounded_packet_frames(
    packet_frames: Sequence[dict[str, Any]],
    *,
    focus_ms: int,
    max_span_ms: int = 15_000,
) -> list[dict[str, Any]]:
    """Keep a deterministic packet window inside its complete time bound.

    The bound applies to the complete before/focus/after set, not to each
    neighbor independently. The nearest frame is always retained so a stale
    or malformed neighbor cannot make the entire visual stage fail.
    """
    if max_span_ms <= 0:
        raise InputError("vision packet span must be positive")
    if not packet_frames:
        return []
    ordered = sorted(
        packet_frames,
        key=lambda item: (
            abs(int(item["actual_ms"]) - focus_ms),
            int(item["actual_ms"]),
            str(item["frame_id"]),
        ),
    )
    kept: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in ordered:
        frame_id = str(item["frame_id"])
        if frame_id in seen_ids:
            continue
        candidate = kept + [item]
        times = [int(value["actual_ms"]) for value in candidate]
        if max(times) - min(times) <= max_span_ms:
            kept.append(item)
            seen_ids.add(frame_id)
    if kept:
        kept_times = [int(value["actual_ms"]) for value in kept]
        if max(kept_times) - min(kept_times) > max_span_ms:
            kept = [min(kept, key=lambda item: abs(int(item["actual_ms"]) - focus_ms))]
    return sorted(kept, key=lambda item: (int(item["actual_ms"]), str(item["frame_id"])))


def _visual_survey_cache_identity(
    source: Path,
    project_dir: Path,
    *,
    duration_ms: int,
    interval_seconds: float,
    strict: bool,
    scene_detection: bool,
    adaptive_detection: bool,
    speech_reference_times_ms: Sequence[int],
    source_sha256: str | None,
    cache_filename: str = "visual-survey.json",
    decode_mode: str | None = None,
) -> tuple[Path, str, str, str]:
    checkpoints = project_dir / ".state" / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    source_digest = source_sha256 or sha256_file(source)
    ffmpeg_version = _tool_version(shutil.which("ffmpeg")) or "unavailable"
    key = cache_key(
        # The full-decode detector fix restores hard/adaptive candidates after
        # the old synthetic keepalive ended early. Version the receipt so a
        # resume cannot restore that incomplete structural schedule. The v3
        # tail-guard behavior remains part of this policy.
        "visual-survey-v4-full-decode",
        source_digest,
        duration_ms,
        round(interval_seconds, 6),
        strict,
        scene_detection,
        adaptive_detection,
        tuple(int(value) for value in speech_reference_times_ms),
        "scene-threshold=0.3",
        "adaptive-threshold=0.003",
        "adaptive-fps=2",
        ffmpeg_version,
        __version__,
        # A hardware-decoded survey must never silently reuse a software
        # receipt (or vice versa).  The component is appended only for a
        # non-default decode mode so existing software cache keys stay
        # byte-identical.
        *( [f"decode-mode={decode_mode}"] if decode_mode else [] ),
    )
    return checkpoints / cache_filename, key, source_digest, ffmpeg_version


def _visual_survey_structural_cache_identity(
    source: Path,
    project_dir: Path,
    *,
    duration_ms: int,
    interval_seconds: float,
    strict: bool,
    scene_detection: bool,
    adaptive_detection: bool,
    source_sha256: str | None,
    decode_mode: str | None = None,
) -> tuple[Path, str, str, str]:
    """Identify context-free scene/adaptive survey results for reuse.

    Structural candidates do not depend on transcript reference times. Keeping
    this receipt separate lets a sidecar/transcript edit reuse the expensive
    full-duration detector pass while contextual candidates are rebuilt and
    merged deterministically for the new transcript.
    """

    return _visual_survey_cache_identity(
        source,
        project_dir,
        duration_ms=duration_ms,
        interval_seconds=interval_seconds,
        strict=strict,
        scene_detection=scene_detection,
        adaptive_detection=adaptive_detection,
        speech_reference_times_ms=(),
        source_sha256=source_sha256,
        cache_filename="visual-survey-structural.json",
        decode_mode=decode_mode,
    )


def _restore_visual_survey_cache(
    cache_path: Path,
    key: str,
    *,
    source_digest: str | None = None,
    ffmpeg_version: str | None = None,
) -> tuple[Any, ...] | None:
    """Restore candidates only after validating the complete cache envelope."""

    from .scene_detection import SurveyCandidate

    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if not (
            isinstance(cached, dict)
            and cached.get("schema_version") == "1.0"
            and cached.get("cache_key") == key
        ):
            return None
        if source_digest is not None and cached.get("source_sha256") != source_digest:
            return None
        if ffmpeg_version is not None and cached.get("ffmpeg_version") != ffmpeg_version:
            return None
        raw_candidates = cached.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError("visual survey cache candidates are not a list")
        restored: list[SurveyCandidate] = []
        for raw in raw_candidates:
            if not isinstance(raw, dict):
                raise ValueError("visual survey cache candidate is not an object")
            raw_reasons = raw.get("reasons")
            if not isinstance(raw_reasons, list) or not all(
                isinstance(item, str) for item in raw_reasons
            ):
                raise ValueError("visual survey cache candidate reasons are invalid")
            score = float(raw["score"])
            if not math.isfinite(score):
                raise ValueError("visual survey cache candidate score is not finite")
            restored.append(
                SurveyCandidate(
                    candidate_id=str(raw["candidate_id"]),
                    requested_ms=int(raw["requested_ms"]),
                    actual_ms=int(raw["actual_ms"]) if raw.get("actual_ms") is not None else None,
                    raw_pts=int(raw["raw_pts"]) if raw.get("raw_pts") is not None else None,
                    time_base=str(raw["time_base"]) if raw.get("time_base") is not None else None,
                    reasons=tuple(raw_reasons),
                    score=score,
                    timestamp_source=str(raw["timestamp_source"]),
                )
            )
        LOGGER.info("Reused source-keyed visual survey cache: %s", cache_path)
        return tuple(restored)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        LOGGER.info("Visual survey cache miss or corruption: %s", cache_path)
        return None


def _write_visual_survey_cache(
    cache_path: Path,
    *,
    key: str,
    source_digest: str,
    ffmpeg_version: str,
    candidates: Sequence[Any],
) -> None:
    atomic_write_json(
        cache_path,
        {
            "schema_version": "1.0",
            "cache_key": key,
            "source_sha256": source_digest,
            "ffmpeg_version": ffmpeg_version,
            "candidates": [asdict(candidate) for candidate in candidates],
        },
    )


def _load_or_run_visual_survey(
    source: Path,
    project_dir: Path,
    *,
    duration_ms: int,
    interval_seconds: float,
    strict: bool,
    scene_detection: bool,
    adaptive_detection: bool,
    speech_reference_times_ms: Sequence[int] = (),
    source_sha256: str | None = None,
    decode_mode: str | None = None,
    hwaccel: str | None = None,
) -> tuple[Any, ...]:
    """Reuse a validated source-keyed survey across downstream rebuilds.

    ``decode_mode`` partitions the cache identity so a hardware-decoded
    receipt can never satisfy a software request, while ``hwaccel`` reaches
    only the combined detector decode pass.
    """

    from .scene_detection import (
        contextual_candidates,
        merge_survey_candidates,
        survey_video_candidates,
    )

    cache_path, key, source_digest, ffmpeg_version = _visual_survey_cache_identity(
        source,
        project_dir,
        duration_ms=duration_ms,
        interval_seconds=interval_seconds,
        strict=strict,
        scene_detection=scene_detection,
        adaptive_detection=adaptive_detection,
        speech_reference_times_ms=speech_reference_times_ms,
        source_sha256=source_sha256,
        decode_mode=decode_mode,
    )
    restored = _restore_visual_survey_cache(
        cache_path, key, source_digest=source_digest, ffmpeg_version=ffmpeg_version
    )
    if restored is not None:
        return restored
    structural_path, structural_key, structural_digest, structural_ffmpeg_version = (
        _visual_survey_structural_cache_identity(
            source,
            project_dir,
            duration_ms=duration_ms,
            interval_seconds=interval_seconds,
            strict=strict,
            scene_detection=scene_detection,
            adaptive_detection=adaptive_detection,
            source_sha256=source_sha256,
            decode_mode=decode_mode,
        )
    )
    structural = _restore_visual_survey_cache(
        structural_path,
        structural_key,
        source_digest=structural_digest,
        ffmpeg_version=structural_ffmpeg_version,
    )
    shared_structural_path = _visual_shared_survey_path(structural_key)
    if structural is None and shared_structural_path is not None:
        structural = _restore_visual_survey_cache(
            shared_structural_path,
            structural_key,
            source_digest=structural_digest,
            ffmpeg_version=structural_ffmpeg_version,
        )
        if structural is not None:
            _write_visual_survey_cache(
                structural_path,
                key=structural_key,
                source_digest=structural_digest,
                ffmpeg_version=structural_ffmpeg_version,
                candidates=structural,
            )
    if structural is None:
        structural = survey_video_candidates(
            source,
            duration_ms=duration_ms,
            interval_seconds=interval_seconds,
            strict=strict,
            scene_detection=scene_detection,
            adaptive_detection=adaptive_detection,
            ffmpeg_threads=_visual_survey_ffmpeg_threads(),
            timeout_seconds=_visual_survey_timeout_seconds(duration_ms),
            speech_reference_times_ms=(),
            hwaccel=hwaccel,
        )
        _write_visual_survey_cache(
            structural_path,
            key=structural_key,
            source_digest=structural_digest,
            ffmpeg_version=structural_ffmpeg_version,
            candidates=structural,
        )
        if shared_structural_path is not None:
            _write_visual_survey_cache(
                shared_structural_path,
                key=structural_key,
                source_digest=structural_digest,
                ffmpeg_version=structural_ffmpeg_version,
                candidates=structural,
            )
            _prune_shared_json_cache(
                shared_structural_path.parent,
                current_path=shared_structural_path,
                cache_limit=_visual_shared_cache_limit(),
            )
    candidates = merge_survey_candidates(
        (
            structural,
            contextual_candidates(speech_reference_times_ms=speech_reference_times_ms),
        )
    )
    _write_visual_survey_cache(
        cache_path,
        key=key,
        source_digest=source_digest,
        ffmpeg_version=ffmpeg_version,
        candidates=candidates,
    )
    return candidates


def _load_or_run_visual_survey_with_frames(
    source: Path,
    project_dir: Path,
    *,
    duration_ms: int,
    interval_seconds: float,
    strict: bool,
    scene_detection: bool,
    adaptive_detection: bool,
    speech_reference_times_ms: Sequence[int],
    periodic_times_ms: Sequence[int],
    frame_output_dir: Path,
    source_sha256: str | None = None,
    decode_mode: str | None = None,
    hwaccel: str | None = None,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Run one survey decode and optionally emit exact-safe periodic frames.

    The existing candidate cache remains the source of truth. On a cold cache,
    the hard-cut, adaptive, and periodic branches are measured in the same
    decode; hard-cut and periodic/contextual PNGs were proven pixel-identical
    to guarded exact extraction. Adaptive samples are never reused as evidence.
    ``decode_mode`` partitions the survey cache identity and ``hwaccel`` is the
    verified accelerator handed to the combined decode pass only.
    """

    from .scene_detection import (
        contextual_candidates,
        detect_combined_survey_frames,
        merge_survey_candidates,
        periodic_candidates,
        survey_video_candidates,
    )
    # Resolve before ASR/visual worker overlap can mutate runtime search paths
    # while faster-whisper registers CUDA DLL folders.
    ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"
    survey_timeout_seconds = _visual_survey_timeout_seconds(duration_ms)

    cache_path, key, source_digest, ffmpeg_version = _visual_survey_cache_identity(
        source,
        project_dir,
        duration_ms=duration_ms,
        interval_seconds=interval_seconds,
        strict=strict,
        scene_detection=scene_detection,
        adaptive_detection=adaptive_detection,
        speech_reference_times_ms=speech_reference_times_ms,
        source_sha256=source_sha256,
        decode_mode=decode_mode,
    )
    restored = _restore_visual_survey_cache(
        cache_path, key, source_digest=source_digest, ffmpeg_version=ffmpeg_version
    )
    if restored is not None:
        return restored, ()

    structural_path, structural_key, structural_digest, structural_ffmpeg_version = (
        _visual_survey_structural_cache_identity(
            source,
            project_dir,
            duration_ms=duration_ms,
            interval_seconds=interval_seconds,
            strict=strict,
            scene_detection=scene_detection,
            adaptive_detection=adaptive_detection,
            source_sha256=source_sha256,
            decode_mode=decode_mode,
        )
    )
    structural_restored = _restore_visual_survey_cache(
        structural_path,
        structural_key,
        source_digest=structural_digest,
        ffmpeg_version=structural_ffmpeg_version,
    )
    shared_structural_path = _visual_shared_survey_path(structural_key)
    if structural_restored is None and shared_structural_path is not None:
        structural_restored = _restore_visual_survey_cache(
            shared_structural_path,
            structural_key,
            source_digest=structural_digest,
            ffmpeg_version=structural_ffmpeg_version,
        )
        if structural_restored is not None:
            _write_visual_survey_cache(
                structural_path,
                key=structural_key,
                source_digest=structural_digest,
                ffmpeg_version=structural_ffmpeg_version,
                candidates=structural_restored,
            )
    if structural_restored is not None:
        candidates = merge_survey_candidates(
            (
                structural_restored,
                contextual_candidates(speech_reference_times_ms=speech_reference_times_ms),
            )
        )
        _write_visual_survey_cache(
            cache_path,
            key=key,
            source_digest=source_digest,
            ffmpeg_version=ffmpeg_version,
            candidates=candidates,
        )
        return tuple(candidates), ()

    shared_frames: tuple[Any, ...] = ()
    if scene_detection and adaptive_detection:
        try:
            # ``periodic_times_ms`` can include a dense subtitle scaffold.
            # Keep the candidate set complete, but bound only the timestamps
            # encoded into this single FFmpeg filter graph. Some Windows
            # builds fail filter allocation before decoding when the generated
            # expression is large; retry with a smaller deterministic schedule
            # before giving up on shared frame acceleration. Missing shared
            # frames are always recovered by exact extraction below.
            ordered_periodic_times = tuple(sorted(set(int(value) for value in periodic_times_ms)))
            initial_cap = min(_MAX_SHARED_SURVEY_EMISSION_TIMES, len(ordered_periodic_times))
            attempt_caps = [initial_cap]
            attempt_caps.extend(
                cap
                for cap in _SHARED_SURVEY_RETRY_CAPS
                if cap < initial_cap and cap not in attempt_caps
            )
            last_emission_error: ValidationFailure | None = None
            for attempt_index, cap in enumerate(attempt_caps):
                emitted_periodic_times = _bounded_shared_survey_emission_times(
                    ordered_periodic_times,
                    max_count=max(1, cap),
                )
                try:
                    # A failed FFmpeg graph may leave a sentinel or partial PNGs
                    # behind. The directory is disposable survey state, so
                    # clear it before retrying and never let stale files enter
                    # the measured-frame validator.
                    if attempt_index:
                        frame_output_dir.mkdir(parents=True, exist_ok=True)
                        for stale_path in frame_output_dir.iterdir():
                            if stale_path.is_dir() and not stale_path.is_symlink():
                                shutil.rmtree(stale_path, ignore_errors=True)
                            elif not stale_path.is_symlink():
                                stale_path.unlink(missing_ok=True)
                    measured_hard, measured_adaptive, emitted = detect_combined_survey_frames(
                        source,
                        frame_output_dir,
                        emitted_periodic_times,
                        ffmpeg_threads=_visual_survey_ffmpeg_threads(),
                        ffmpeg_bin=ffmpeg_path,
                        timeout_seconds=survey_timeout_seconds,
                        hwaccel=hwaccel,
                    )
                    if attempt_index:
                        LOGGER.info(
                            "Shared survey frame emission recovered with cap=%d after %d failed graph attempt(s)",
                            cap,
                            attempt_index,
                        )
                    last_emission_error = None
                    break
                except ValidationFailure as exc:
                    last_emission_error = exc
                    detail = str(exc).casefold()
                    retryable = any(
                        marker in detail
                        for marker in (
                            "cannot allocate memory",
                            "error initializing filters",
                            "error initializing filter",
                        )
                    )
                    if not retryable or attempt_index == len(attempt_caps) - 1:
                        raise
            if last_emission_error is not None:
                raise last_emission_error
            # The frame-emission graph applies the same hard/adaptive detector
            # filters as the candidate-only graph. Rebuild the canonical
            # structural policy from those measured streams: add the periodic
            # safety schedule, then run the same merge/clustering pass used by
            # ``survey_video_candidates``. This avoids decoding the complete
            # source a second time. Synthetic late-cut and real 80-minute
            # parity fixtures require every timestamp, raw PTS, time base,
            # reason, score, and source label to match across the two graphs.
            structural_candidates = merge_survey_candidates(
                (
                    periodic_candidates(
                        duration_ms,
                        interval_seconds=interval_seconds,
                        strict=strict,
                    ),
                    measured_hard,
                    measured_adaptive,
                )
            )
            candidates = merge_survey_candidates(
                (
                    structural_candidates,
                    contextual_candidates(speech_reference_times_ms=speech_reference_times_ms),
                )
            )
            shared_frames = tuple(emitted)
            _write_visual_survey_cache(
                structural_path,
                key=structural_key,
                source_digest=structural_digest,
                ffmpeg_version=structural_ffmpeg_version,
                candidates=structural_candidates,
            )
            if shared_structural_path is not None:
                _write_visual_survey_cache(
                    shared_structural_path,
                    key=structural_key,
                    source_digest=structural_digest,
                    ffmpeg_version=structural_ffmpeg_version,
                    candidates=structural_candidates,
                )
                _prune_shared_json_cache(
                    shared_structural_path.parent,
                    current_path=shared_structural_path,
                    cache_limit=_visual_shared_cache_limit(),
                )
        except ValidationFailure:
            LOGGER.info("Shared survey frame emission unavailable; using guarded survey fallback")
            candidates = survey_video_candidates(
                source,
                duration_ms=duration_ms,
                interval_seconds=interval_seconds,
                strict=strict,
                scene_detection=scene_detection,
                adaptive_detection=adaptive_detection,
                ffmpeg_threads=_visual_survey_ffmpeg_threads(),
                ffmpeg_bin=ffmpeg_path,
                timeout_seconds=survey_timeout_seconds,
                speech_reference_times_ms=speech_reference_times_ms,
                # The combined frame graph already failed; recover with the
                # independently guarded software survey path.
                hwaccel=None,
            )
    else:
        candidates = survey_video_candidates(
            source,
            duration_ms=duration_ms,
            interval_seconds=interval_seconds,
            strict=strict,
            scene_detection=scene_detection,
            adaptive_detection=adaptive_detection,
            ffmpeg_threads=_visual_survey_ffmpeg_threads(),
            ffmpeg_bin=ffmpeg_path,
            timeout_seconds=survey_timeout_seconds,
            speech_reference_times_ms=speech_reference_times_ms,
            # No combined frame graph is active in this branch.
            hwaccel=None,
        )
    _write_visual_survey_cache(
        cache_path,
        key=key,
        source_digest=source_digest,
        ffmpeg_version=ffmpeg_version,
        candidates=candidates,
    )
    return tuple(candidates), shared_frames


def _copy_file_atomic(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    durable: bool = True,
) -> tuple[str, int]:
    """Copy and hash a generated frame without exposing a partial destination.

    Restore copies are disposable acceleration work, so callers may skip the
    per-file fsync after hashing and atomic replacement. Checkpoint commits keep
    the durable default.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as source_stream, os.fdopen(handle, "wb") as target_stream:
            # ``os.fdopen`` owns the descriptor once it succeeds. Mark it as
            # transferred so the cleanup path cannot double-close a descriptor
            # number that Windows may already have reused in another worker.
            handle = -1
            while chunk := source_stream.read(1024 * 1024):
                digest.update(chunk)
                target_stream.write(chunk)
                size += len(chunk)
            target_stream.flush()
            if durable:
                os.fsync(target_stream.fileno())
        actual_sha256 = digest.hexdigest()
        if expected_size is not None and size != expected_size:
            raise ValueError("cached frame size changed during restore")
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError("cached frame digest does not match its manifest")
        os.replace(temporary, destination)
        return actual_sha256, size
    finally:
        # ``os.fdopen`` closes the descriptor on the normal path. If opening
        # the source failed first, close the original descriptor here.
        try:
            os.close(handle)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def _link_or_copy_file_atomic(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    durable: bool = True,
) -> tuple[str, int]:
    """Materialize an immutable checkpoint with a hardlink when safe.

    Visual checkpoint files are immutable raw PNG bytes. On the common
    same-volume path, a hardlink avoids another full read/write pass while
    preserving independent paths: later metadata writes replace the evidence
    pathname atomically and leave the raw checkpoint inode untouched. Different
    volumes, filesystems without hardlink support, and explicit opt-outs fall
    back to the guarded atomic-copy implementation.
    """

    disabled = os.environ.get("VSR_DISABLE_CHECKPOINT_HARDLINKS", "").strip().casefold()
    if disabled in {"1", "true", "yes", "on"}:
        return _copy_file_atomic(
            source,
            destination,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            durable=durable,
        )

    temporary: Path | None = None
    handle = -1
    try:
        source_stat = source.stat()
        if source.is_symlink() or not source.is_file():
            raise OSError(f"checkpoint source is not a regular file: {source}")
        actual_sha256 = sha256_file(source)
        size = int(source_stat.st_size)
        if expected_size is not None and size != expected_size:
            raise ValueError("cached frame size changed during restore")
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError("cached frame digest does not match its manifest")
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        os.close(handle)
        handle = -1
        temporary.unlink(missing_ok=True)
        os.link(source, temporary)
        if durable:
            try:
                descriptor = os.open(temporary, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                # The source bytes are already durable; some Windows/filesystem
                # combinations reject fsync on a newly linked read-only handle.
                # Keep the link and preserve the atomic path invariant.
                pass
        os.replace(temporary, destination)
        return actual_sha256, size
    except ValueError:
        raise
    except OSError:
        return _copy_file_atomic(
            source,
            destination,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            durable=durable,
        )
    finally:
        try:
            os.close(handle)
        except OSError:
            pass
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _visual_frame_cache_limit() -> int:
    """Return the raw-frame checkpoint budget in bytes (zero disables writes)."""

    default_limit = 512 * 1024 * 1024
    raw_limit = os.environ.get("VSR_VISUAL_FRAME_CACHE_MAX_BYTES", "").strip()
    if not raw_limit:
        return default_limit
    try:
        return max(0, int(raw_limit))
    except ValueError:
        LOGGER.warning("Ignoring invalid VSR_VISUAL_FRAME_CACHE_MAX_BYTES=%r", raw_limit)
        return default_limit


def _prune_visual_frame_checkpoints(
    checkpoint_root: Path,
    *,
    current_cache_dir: Path,
    cache_limit: int,
) -> None:
    """Keep all schedule-bound visual receipts within one bounded budget."""

    if cache_limit <= 0 or not checkpoint_root.is_dir():
        return
    current_resolved = current_cache_dir.resolve()
    receipts: list[tuple[int, int, Path]] = []
    total_bytes = 0
    try:
        for manifest_path in checkpoint_root.glob("*/manifest.json"):
            cache_dir = manifest_path.parent
            if cache_dir.is_symlink() or cache_dir.resolve() == current_resolved:
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            raw_bytes = payload.get("cache_bytes")
            if not isinstance(raw_bytes, int) or raw_bytes < 0:
                continue
            usage = raw_bytes
            total_bytes += usage
            receipts.append((manifest_path.stat().st_mtime_ns, usage, cache_dir))
        current_manifest = current_cache_dir / "manifest.json"
        if current_manifest.is_file():
            payload = json.loads(current_manifest.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("cache_bytes"), int):
                total_bytes += max(0, int(payload["cache_bytes"]))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        # A malformed acceleration receipt must never block canonical output.
        LOGGER.info("Skipping visual checkpoint pruning after receipt inspection failure")
        return

    if total_bytes <= cache_limit:
        return
    for _mtime_ns, usage, cache_dir in sorted(receipts, key=lambda item: (item[0], item[2].as_posix())):
        if total_bytes <= cache_limit:
            break
        try:
            if cache_dir.is_symlink() or cache_dir.resolve().parent != checkpoint_root.resolve():
                continue
            shutil.rmtree(cache_dir)
            total_bytes -= usage
        except OSError:
            LOGGER.warning("Unable to prune old visual checkpoint: %s", cache_dir)


def _bounded_shared_frame_records(
    records: Sequence[Mapping[str, Any]], max_bytes: int
) -> tuple[list[dict[str, Any]], int]:
    """Choose a deterministic, temporally spread subset for a bounded cache.

    A complete visual schedule can be larger than the shared-cache budget even
    though each raw PNG is independently reusable.  Keeping no receipt in that
    case wastes all cross-project acceleration.  This helper keeps an evenly
    spaced subset when the complete schedule does not fit; the prior-schedule
    restore path can then reuse those exact records while the remaining frames
    take the normal guarded extraction path.  The subset is acceleration-only:
    callers still materialize every requested frame in the project output.

    The byte budget covers frame payloads, matching the existing checkpoint
    accounting.  Invalid/oversized records are simply skipped and never weaken
    the SHA/size gates used by the materializer.
    """

    if max_bytes <= 0 or not records:
        return [], 0
    normalized = [dict(record) for record in records]
    sizes = [
        int(record.get("size_bytes", 0))
        if isinstance(record.get("size_bytes"), int)
        else 0
        for record in normalized
    ]
    if any(size <= 0 for size in sizes):
        # A malformed record must not enter the shared acceleration receipt.
        valid = [
            (record, size)
            for record, size in zip(normalized, sizes, strict=True)
            if size > 0
        ]
        normalized = [record for record, _size in valid]
        sizes = [size for _record, size in valid]
    if not normalized:
        return [], 0
    total = sum(sizes)
    if total <= max_bytes:
        return normalized, total

    # Estimate how many records fit, then place those slots uniformly over the
    # timeline.  A second pass fills any slack caused by variable PNG sizes.
    average = max(1, total // len(normalized))
    target_count = max(1, min(len(normalized), max_bytes // average))
    if target_count == 1:
        candidate_indexes = [0]
    else:
        denominator = target_count - 1
        last_index = len(normalized) - 1
        candidate_indexes = sorted(
            {round(index * last_index / denominator) for index in range(target_count)}
        )

    selected: set[int] = set()
    selected_bytes = 0
    for index in candidate_indexes:
        size = sizes[index]
        if selected_bytes + size > max_bytes:
            continue
        selected.add(index)
        selected_bytes += size
    for index, size in enumerate(sizes):
        if index in selected or selected_bytes + size > max_bytes:
            continue
        selected.add(index)
        selected_bytes += size
    chosen = [normalized[index] for index in sorted(selected)]
    return chosen, selected_bytes


def _ocr_cache_limit() -> int:
    """Return the resumable OCR checkpoint budget in bytes.

    OCR results are small relative to decoded frames, so a separate budget keeps
    the cache useful on long videos without allowing an unbounded JSON ledger.
    Set ``VSR_OCR_CACHE_MAX_BYTES=0`` to disable persistence while retaining the
    normal in-process OCR path.
    """

    default_limit = 64 * 1024 * 1024
    raw_limit = os.environ.get("VSR_OCR_CACHE_MAX_BYTES", "").strip()
    if not raw_limit:
        return default_limit
    try:
        return max(0, int(raw_limit))
    except ValueError:
        LOGGER.warning("Ignoring invalid VSR_OCR_CACHE_MAX_BYTES=%r", raw_limit)
        return default_limit


def _ocr_checkpoint_flush_interval() -> int:
    """Bound how many newly recognized pixels can be lost on interruption.

    OCR observations are independent and the checkpoint envelope is atomic, so
    flushing a bounded batch preserves the same evidence while making a retry
    resume from durable completed work.  The default keeps JSON/fsync overhead
    small; a lower value is useful on an interruption-prone host.
    """

    default_interval = 16
    raw_interval = os.environ.get("VSR_OCR_CHECKPOINT_BATCH", "").strip()
    if not raw_interval:
        return default_interval
    try:
        return max(1, min(64, int(raw_interval)))
    except ValueError:
        LOGGER.warning("Ignoring invalid VSR_OCR_CHECKPOINT_BATCH=%r", raw_interval)
        return default_interval


def _ocr_batch_size() -> int:
    """Bound images sent to one isolated batch OCR worker invocation.

    Paddle's detector/recognizer worker is initialized per subprocess and the
    adapter has a finite wall-clock timeout.  Keeping a bounded default avoids
    timing out on long recordings while still amortizing model startup; the
    environment override is useful for hosts with different GPU throughput.
    """

    default_size = 256
    raw_size = os.environ.get("VSR_OCR_BATCH_SIZE", "").strip()
    if not raw_size:
        return default_size
    try:
        return max(1, min(1024, int(raw_size)))
    except ValueError:
        LOGGER.warning("Ignoring invalid VSR_OCR_BATCH_SIZE=%r", raw_size)
        return default_size


def _paddle_ocr_batch_workers(adapter: Any | None = None) -> int:
    """Return the bounded opt-in fan-out for persistent Paddle OCR batches.

    A Paddle adapter owns a stateful isolated worker, so concurrent requests
    require independent ``spawn_worker`` instances.  Keep the default at one
    process for predictable memory use and make the measured two-worker path
    explicit.  Adapters without the factory retain the sequential path.
    """

    raw = os.environ.get("VSR_PADDLE_OCR_WORKERS", "").strip()
    if not raw:
        return 1
    try:
        requested = max(1, min(2, int(raw)))
    except ValueError:
        LOGGER.warning("Ignoring invalid VSR_PADDLE_OCR_WORKERS=%r", raw)
        return 1
    if requested <= 1 or adapter is None:
        return requested
    if not callable(getattr(adapter, "recognize_many", None)) or not callable(
        getattr(adapter, "spawn_worker", None)
    ):
        LOGGER.warning(
            "VSR_PADDLE_OCR_WORKERS=%d requested but the OCR adapter has no "
            "independent worker factory; using one batch worker",
            requested,
        )
        return 1
    return requested


def _ocr_shared_cache_limit() -> int:
    """Return the total shared OCR-cache budget in bytes."""

    default_limit = 256 * 1024 * 1024
    raw_limit = os.environ.get("VSR_OCR_SHARED_CACHE_MAX_BYTES", "").strip()
    if not raw_limit:
        return default_limit
    try:
        return max(0, int(raw_limit))
    except ValueError:
        LOGGER.warning("Ignoring invalid VSR_OCR_SHARED_CACHE_MAX_BYTES=%r", raw_limit)
        return default_limit


def _prune_shared_json_cache(
    cache_root: Path, *, current_path: Path, cache_limit: int
) -> None:
    """Keep flat shared JSON acceleration state within a bounded budget.

    Shared OCR/survey receipts are immutable content-addressed files, so the
    common path is one new file per invocation.  A full ``glob``/``stat`` pass
    after every write makes a warm shared cache quadratic in receipt count.
    Account for the just-written file with one stat and reconcile the complete
    directory only periodically (or as soon as the estimate crosses the
    configured byte budget).  The reconciliation path remains authoritative,
    and failures discard the hint so a later write retries a safe inventory.
    """

    if cache_limit <= 0 or not cache_root.is_dir():
        # A removed/recreated cache directory or a temporary budget opt-out
        # must not inherit a stale size estimate from an earlier run.
        try:
            _SHARED_JSON_PRUNE_STATE.pop(cache_root.resolve(), None)
        except OSError:
            pass
        return

    root = cache_root.resolve()
    current_resolved = current_path.resolve()
    state = _SHARED_JSON_PRUNE_STATE.get(root)
    if state is not None:
        writes, total_bytes, sizes = state
        current_size = 0
        try:
            if current_path.is_file() and not current_path.is_symlink():
                current_size = int(current_path.stat().st_size)
        except OSError:
            # Atomic replacement may have been interrupted between the write
            # and this bookkeeping call.  Discard the estimate and let the
            # next write establish a fresh authoritative inventory.
            current_size = 0
        previous_size = sizes.get(current_resolved, 0)
        total_bytes = max(0, total_bytes - previous_size) + current_size
        if current_size:
            sizes[current_resolved] = current_size
        else:
            sizes.pop(current_resolved, None)
        writes += 1
        _SHARED_JSON_PRUNE_STATE[root] = (writes, total_bytes, sizes)
        if total_bytes <= cache_limit and writes < _SHARED_JSON_PRUNE_INTERVAL:
            return

    try:
        entries = [
            item
            for item in root.glob("*.json")
            if item.is_file() and not item.is_symlink()
        ]
        entry_sizes = {item.resolve(): int(item.stat().st_size) for item in entries}
        total_bytes = sum(entry_sizes.values())
        if total_bytes <= cache_limit:
            _SHARED_JSON_PRUNE_STATE[root] = (0, total_bytes, entry_sizes)
            return

        current_resolved = current_path.resolve()
        ordered = sorted(entries, key=lambda item: (item.stat().st_mtime_ns, item.as_posix()))
        for item in ordered:
            if total_bytes <= cache_limit:
                break
            if item.resolve() == current_resolved:
                continue
            try:
                size = item.stat().st_size
                item.unlink()
                total_bytes -= size
                entry_sizes.pop(item.resolve(), None)
            except OSError:
                LOGGER.warning("Unable to prune shared cache entry: %s", item)
        _SHARED_JSON_PRUNE_STATE[root] = (0, total_bytes, entry_sizes)
    except OSError:
        _SHARED_JSON_PRUNE_STATE.pop(root, None)
        LOGGER.info("Skipping shared JSON-cache pruning after receipt inspection failure")


def _ocr_cache_adapter_identity(adapter: Any, adapter_key: str) -> str:
    """Add stable engine settings to the OCR checkpoint identity."""

    values: list[str] = [adapter_key]
    for name in (
        "executable",
        "default_language",
        "page_segmentation_mode",
        "timeout_seconds",
        "worker_python",
        "device",
        "backend_name",
        "model_name_or_path",
        "adapter_version",
    ):
        value = getattr(adapter, name, None)
        if value is not None and isinstance(value, (str, int, float, bool)):
            values.append(f"{name}={value}")
    descriptor = getattr(adapter, "descriptor", None)
    if descriptor is not None:
        for name in ("provider_id", "route", "model", "model_version", "adapter_version"):
            value = getattr(descriptor, name, None)
            if value is not None:
                values.append(f"descriptor.{name}={value}")
    version = getattr(adapter, "version", None)
    if callable(version):
        try:
            value = version()
        except Exception:  # pragma: no cover - optional backend boundary
            value = None
        if value is not None:
            values.append(f"engine_version={value}")
    return "|".join(values)


def _restore_ocr_cache_payload(
    cache_path: Path,
    *,
    cache_key_value: str,
    source_digest: str,
    adapter_identity: str,
    cache_limit: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Restore OCR results and retain the validated payload for materialization.

    Shared-cache warm hits need to copy the exact validated envelope into the
    project-local checkpoint.  Returning the parsed payload avoids reading and
    decoding the same (potentially large) JSON file a second time while keeping
    all existing size, identity, entry, and observation validation gates in one
    place.
    """

    from .ocr import deserialize_observation

    try:
        if not cache_path.is_file() or cache_limit <= 0:
            return None
        if cache_path.stat().st_size > cache_limit:
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if (
            payload.get("schema_version") != "1.0"
            or payload.get("cache_key") != cache_key_value
            or payload.get("source_sha256") != source_digest
            or payload.get("adapter_identity") != adapter_identity
        ):
            return None
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            return None
        restored: dict[str, Any] = {}
        for raw in raw_entries:
            if not isinstance(raw, dict):
                return None
            pixel_hash = raw.get("pixel_hash")
            if not isinstance(pixel_hash, str) or not pixel_hash or pixel_hash in restored:
                return None
            observation = raw.get("observation")
            if observation is None:
                restored[pixel_hash] = None
            elif isinstance(observation, Mapping):
                restored[pixel_hash] = deserialize_observation(observation)
            else:
                return None
        LOGGER.info("Reused OCR checkpoint: %s (%d entries)", cache_path, len(restored))
        return restored, payload
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError, ValidationFailure):
        LOGGER.info("OCR checkpoint miss or corruption: %s", cache_path)
        return None


def _restore_ocr_cache(
    cache_path: Path,
    *,
    cache_key_value: str,
    source_digest: str,
    adapter_identity: str,
    cache_limit: int,
) -> dict[str, Any] | None:
    """Restore OCR results only after validating the complete checkpoint envelope."""

    restored = _restore_ocr_cache_payload(
        cache_path,
        cache_key_value=cache_key_value,
        source_digest=source_digest,
        adapter_identity=adapter_identity,
        cache_limit=cache_limit,
    )
    return restored[0] if restored is not None else None


def _write_ocr_cache(
    cache_path: Path,
    *,
    cache_key_value: str,
    source_digest: str,
    adapter_identity: str,
    entries: Mapping[str, Any],
    cache_limit: int,
) -> bool:
    """Persist completed OCR results atomically when within the configured bound."""

    from .ocr import serialize_observation

    if cache_limit <= 0:
        return False
    serialized_entries: list[dict[str, Any]] = []
    for pixel_hash in sorted(entries):
        observation = entries[pixel_hash]
        serialized_entries.append(
            {
                "pixel_hash": pixel_hash,
                "observation": (
                    serialize_observation(observation) if observation is not None else None
                ),
            }
        )
    payload = {
        "schema_version": "1.0",
        "cache_key": cache_key_value,
        "source_sha256": source_digest,
        "adapter_identity": adapter_identity,
        "entries": serialized_entries,
    }
    # ``atomic_write_json`` appends one trailing newline, so include it in the
    # budget calculation to keep the admission check byte-exact.
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    if len(encoded.encode("utf-8")) > cache_limit:
        LOGGER.info(
            "Skipping OCR checkpoint: %d bytes exceeds budget %d",
            len(encoded.encode("utf-8")),
            cache_limit,
        )
        return False
    # The size guard above measures the compact representation.  Persist the
    # same representation so a checkpoint accepted by the budget check is not
    # inflated by pretty-print whitespace and then rejected on restore.  This
    # also reduces repeated incremental checkpoint write amplification without
    # changing the JSON data or cache identity.
    atomic_write_json(cache_path, payload, compact=True)
    return True


class _StreamingOCRPrefetch:
    """Consume finalized frame callbacks through bounded OCR workers.

    Frame extraction and OCR are independent after a PNG has been atomically
    materialized.  This queue keeps that overlap bounded and preserves a safe
    fallback: if an optional worker fails, the canonical OCR pass reruns any
    missing frame through the normal adapter path.  Fan-out is capped at one
    in-flight batch per independent adapter, so it cannot create an unbounded
    process or memory pool.  The class is intentionally private; it is an
    acceleration layer, not a new evidence contract.
    """

    def __init__(self, adapter: Any, *, batch_size: int, worker_count: int = 1) -> None:
        recognize_many = getattr(adapter, "recognize_many", None)
        if not callable(recognize_many):
            raise InputError("streaming OCR prefetch requires a batch-capable adapter")
        self.adapter = adapter
        self.batch_size = max(1, int(batch_size))
        self.worker_count = max(1, min(2, int(worker_count)))
        self._workers: list[Any] = [adapter]
        self._owned_workers: list[Any] = []
        if self.worker_count > 1:
            spawn_worker = getattr(adapter, "spawn_worker", None)
            try:
                if not callable(spawn_worker):
                    raise InputError("OCR adapter cannot create an independent worker")
                for _ in range(self.worker_count - 1):
                    worker = spawn_worker()
                    if worker is adapter or not callable(getattr(worker, "recognize_many", None)):
                        raise InputError("OCR adapter returned an invalid independent worker")
                    available = getattr(worker, "available", None)
                    if callable(available) and not available():
                        raise InputError("OCR independent worker is unavailable")
                    self._workers.append(worker)
                    self._owned_workers.append(worker)
            except BaseException as exc:  # noqa: BLE001 - optional acceleration boundary
                for worker in self._owned_workers:
                    close = getattr(worker, "close", None)
                    if callable(close):
                        close()
                self._workers = [adapter]
                self._owned_workers = []
                self.worker_count = 1
                LOGGER.warning(
                    "OCR worker fan-out unavailable; falling back to one worker: %s", exc
                )
        self._queue: queue.Queue[Any | None] = queue.Queue(maxsize=self.batch_size * 2)
        self._observations: dict[str, Any] = {}
        self._error: BaseException | None = None
        self._failed = threading.Event()
        self._closed = False
        self._submitted_count = 0
        self._recognized_count = 0
        self._batch_count = 0
        self._thread = threading.Thread(
            target=self._run,
            name="vsr-streaming-ocr",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame: Any) -> None:
        """Queue one stable frame without allowing unbounded pending work."""

        if self._closed or self._failed.is_set():
            return
        # A recognizer failure can terminate the worker while the producer is
        # blocked on a full queue.  Timed puts let the producer observe that
        # terminal state instead of deadlocking the visual stage.
        while True:
            if self._closed or self._failed.is_set():
                return
            try:
                self._queue.put(frame, timeout=0.25)
            except queue.Full:
                continue
            self._submitted_count += 1
            return

    @staticmethod
    def _valid_frames(batch: Sequence[Any]) -> list[Any]:
        return [
            frame
            for frame in batch
            if isinstance(getattr(frame, "frame_id", None), str)
            and Path(getattr(frame, "path", "")).is_file()
        ]

    @staticmethod
    def _observation_id(frame: Any) -> str:
        frame_id = str(getattr(frame, "frame_id", ""))
        digits = "".join(character for character in frame_id if character.isdigit())
        return f"P{int(digits):06d}" if digits else "P000000"

    def _recognize_batch(
        self, worker_index: int, batch: list[Any]
    ) -> tuple[list[Any], Mapping[str, Any]]:
        valid = self._valid_frames(batch)
        if not valid:
            return [], {}
        worker = self._workers[worker_index]
        results = worker.recognize_many(
            [Path(frame.path) for frame in valid],
            frame_ids=[str(frame.frame_id) for frame in valid],
            observation_ids=[self._observation_id(frame) for frame in valid],
        )
        if not isinstance(results, Mapping):
            raise ValidationFailure("streaming OCR adapter returned a non-mapping result")
        return valid, results

    def _record_batch(self, valid: list[Any], results: Mapping[str, Any]) -> None:
        if not valid:
            return
        recognized = 0
        for frame in valid:
            frame_id = str(frame.frame_id)
            observation = results.get(frame_id)
            if observation is not None:
                self._observations[frame_id] = observation
                recognized += 1
        self._recognized_count += recognized
        self._batch_count += 1

    def _fail(self, exc: BaseException) -> None:
        if self._error is None:
            self._error = exc
        self._failed.set()

    def _flush(self, batch: list[Any]) -> None:
        if not batch or self._failed.is_set():
            return
        try:
            valid, results = self._recognize_batch(0, batch)
            self._record_batch(valid, results)
        except BaseException as exc:  # noqa: BLE001 - optional acceleration boundary
            self._fail(exc)

    def _run_fanout(self) -> None:
        """Dispatch at most one batch per independent adapter at a time."""

        executor = ThreadPoolExecutor(
            max_workers=self.worker_count,
            thread_name_prefix="vsr-streaming-ocr-batch",
        )
        pending: dict[Future[tuple[list[Any], Mapping[str, Any]]], int] = {}
        available = list(range(self.worker_count))

        def collect(done: Iterable[Future[tuple[list[Any], Mapping[str, Any]]]]) -> None:
            for future in done:
                worker_index = pending.pop(future)
                try:
                    valid, results = future.result()
                    self._record_batch(valid, results)
                    available.append(worker_index)
                except BaseException as exc:  # noqa: BLE001 - optional acceleration boundary
                    self._fail(exc)
                    for remaining in pending:
                        remaining.cancel()
                    raise

        def dispatch(batch: list[Any]) -> None:
            if not batch or self._failed.is_set():
                return
            while not available and pending and not self._failed.is_set():
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                collect(done)
            if self._failed.is_set() or not available:
                return
            worker_index = available.pop(0)
            future = executor.submit(self._recognize_batch, worker_index, batch)
            pending[future] = worker_index

        batch: list[Any] = []
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    dispatch(batch)
                    batch = []
                    while pending and not self._failed.is_set():
                        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                        collect(done)
                    return
                batch.append(item)
                if len(batch) >= self.batch_size:
                    dispatch(batch)
                    batch = []
                    if self._failed.is_set():
                        return
        except BaseException as exc:  # noqa: BLE001 - optional acceleration boundary
            self._fail(exc)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def _run(self) -> None:
        if self.worker_count > 1:
            self._run_fanout()
            return
        batch: list[Any] = []
        while True:
            item = self._queue.get()
            if item is None:
                self._flush(batch)
                return
            batch.append(item)
            if len(batch) >= self.batch_size:
                self._flush(batch)
                batch = []
                if self._failed.is_set():
                    return

    def finish(self) -> None:
        """Stop the worker and wait for all successfully queued OCR batches."""

        if self._closed:
            return
        self._closed = True
        if not self._failed.is_set():
            # A worker can fail between the state check and this enqueue. Use
            # timed puts so a full queue cannot strand shutdown indefinitely;
            # the worker's failure event then lets us abandon the sentinel and
            # join the already-terminating thread safely.
            while not self._failed.is_set():
                try:
                    self._queue.put(None, timeout=0.25)
                    break
                except queue.Full:
                    continue
        self._thread.join()
        for worker in self._owned_workers:
            close = getattr(worker, "close", None)
            if callable(close):
                close()

    @property
    def observations(self) -> dict[str, Any]:
        return dict(self._observations)

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "submitted_count": self._submitted_count,
            "recognized_count": self._recognized_count,
            "batch_count": self._batch_count,
            "worker_count": self.worker_count,
            "error": str(self._error) if self._error is not None else None,
        }


def _load_or_run_ocr(
    source: Path,
    project_dir: Path,
    ordered_frames: Sequence[dict[str, Any]],
    *,
    adapter: Any | None,
    adapter_key: str,
    source_sha256: str | None = None,
    shared_cache_dir: Path | None = None,
    prefetched_by_frame_id: Mapping[str, Any] | None = None,
    prefetch_metrics: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resume completed OCR observations and process only uncached pixels.

    Cache entries are keyed by normalized source pixels rather than filenames or
    frame IDs.  This makes a retry safe after stale evidence is rotated and lets a
    changed request schedule reuse unaffected OCR results.  Observation/frame IDs
    are always remapped to the current deterministic order before they enter the
    canonical project.
    """

    from .ocr import OCRObservation, run_optional_ocr

    if adapter is None:
        return {}, {"status": "unavailable", "cache_hit_count": 0, "cache_miss_count": 0}
    if not adapter.available():
        return {}, {"status": "unavailable", "cache_hit_count": 0, "cache_miss_count": 0}
    if not ordered_frames:
        return {}, {"status": "completed", "cache_hit_count": 0, "cache_miss_count": 0}

    source_digest = source_sha256 or sha256_file(source)
    adapter_identity = _ocr_cache_adapter_identity(adapter, adapter_key)
    checkpoint_key = cache_key(
        "ocr-observations-v1",
        source_digest,
        adapter_identity,
        __version__,
    )
    cache_path = project_dir / ".state" / "checkpoints" / "ocr" / f"{checkpoint_key}.json"
    cache_limit = _ocr_cache_limit()
    shared_cache_limit = _ocr_shared_cache_limit()
    shared_cache_path: Path | None = None
    if shared_cache_dir is not None and shared_cache_limit > 0:
        candidate_shared_dir = Path(shared_cache_dir).expanduser()
        try:
            candidate_shared_dir.mkdir(parents=True, exist_ok=True)
            if candidate_shared_dir.is_dir():
                shared_cache_path = candidate_shared_dir / "ocr" / f"{checkpoint_key}.json"
                if shared_cache_path.resolve() == cache_path.resolve():
                    shared_cache_path = None
        except OSError as exc:
            LOGGER.warning("OCR shared cache unavailable at %s: %s", candidate_shared_dir, exc)
    cached = _restore_ocr_cache(
        cache_path,
        cache_key_value=checkpoint_key,
        source_digest=source_digest,
        adapter_identity=adapter_identity,
        cache_limit=cache_limit,
    ) or {}
    shared_cache_hit = False
    if not cached and shared_cache_path is not None:
        restored_shared = _restore_ocr_cache_payload(
            shared_cache_path,
            cache_key_value=checkpoint_key,
            source_digest=source_digest,
            adapter_identity=adapter_identity,
            cache_limit=shared_cache_limit,
        )
        shared_payload: dict[str, Any] | None = None
        if restored_shared is not None:
            cached, shared_payload = restored_shared
        else:
            cached = {}
        shared_cache_hit = bool(cached)
        if shared_cache_hit:
            try:
                # Match the shared checkpoint's compact encoding locally;
                # otherwise pretty-print whitespace can push a valid shared
                # payload over the project-local restore budget.  The payload
                # was already parsed and fully validated above.
                if shared_payload is not None:
                    atomic_write_json(cache_path, shared_payload, compact=True)
            except (OSError, TypeError, ValueError):
                LOGGER.warning("Unable to materialize OCR shared checkpoint %s", shared_cache_path)

    def pixel_key(frame: Mapping[str, Any]) -> str:
        raw_hash = frame.get("pixel_hash")
        if isinstance(raw_hash, Mapping) and isinstance(raw_hash.get("value"), str):
            return str(raw_hash["value"])
        return sha256_file(project_dir / str(frame["full_frame_path"]))

    frame_keys = [pixel_key(frame) for frame in ordered_frames]
    ocr_by_frame: dict[str, Any] = {}
    pending: dict[str, list[int]] = {}
    cache_hits = 0
    for index, (frame, frame_key) in enumerate(zip(ordered_frames, frame_keys, strict=True)):
        frame_id = str(frame["frame_id"])
        if frame_key in cached:
            cached_observation = cached[frame_key]
            if cached_observation is not None:
                ocr_by_frame[frame_id] = replace(
                    cached_observation,
                    frame_id=frame_id,
                    observation_id=f"O{index + 1:06d}",
                )
            cache_hits += 1
        else:
            pending.setdefault(frame_key, []).append(index)

    representative_indices = [indices[0] for indices in pending.values()]
    representative_count = len(representative_indices)
    completed_by_key: dict[str, Any] = {}
    batch_method = getattr(adapter, "recognize_many", None)
    ocr_worker_count = 1 if callable(batch_method) else _ocr_workers()
    cache_entries = dict(cached)
    prefetched_hits = 0
    prefetched = prefetched_by_frame_id or {}
    if prefetched and representative_indices:
        from .ocr import OCRObservation

        remaining_indices: list[int] = []
        for index in representative_indices:
            frame_id = str(ordered_frames[index]["frame_id"])
            observation = prefetched.get(frame_id)
            if isinstance(observation, OCRObservation):
                completed_by_key[frame_keys[index]] = observation
                cache_entries[frame_keys[index]] = observation
                prefetched_hits += 1
            else:
                remaining_indices.append(index)
        representative_indices = remaining_indices
    checkpoint_flush_count = 0
    checkpoint_write_failures = 0

    def flush_incremental_checkpoint() -> None:
        """Persist local completed work without making acceleration mandatory."""

        nonlocal checkpoint_flush_count, checkpoint_write_failures
        if not cache_entries or cache_limit <= 0:
            return
        try:
            if _write_ocr_cache(
                cache_path,
                cache_key_value=checkpoint_key,
                source_digest=source_digest,
                adapter_identity=adapter_identity,
                entries=cache_entries,
                cache_limit=cache_limit,
            ):
                checkpoint_flush_count += 1
        except (OSError, TypeError, ValueError) as exc:
            checkpoint_write_failures += 1
            LOGGER.warning("Unable to persist incremental OCR checkpoint: %s", exc)

    if callable(batch_method) and representative_indices:
        # Production Paddle OCR loads a detector/recognizer worker once per
        # ``recognize_many`` call.  A single call over several thousand frames
        # can therefore exceed the worker timeout even though each frame is
        # independently tractable.  Keep batches bounded and checkpoint after
        # every completed batch so a timeout/restart resumes from the last
        # durable chunk instead of leaving orphaned visual artifacts.
        batch_size = _ocr_batch_size()
        for start in range(0, len(representative_indices), batch_size):
            batch_indices = representative_indices[start : start + batch_size]
            images = [
                project_dir / str(ordered_frames[index]["full_frame_path"])
                for index in batch_indices
            ]
            frame_ids = [str(ordered_frames[index]["frame_id"]) for index in batch_indices]
            observation_ids = [f"O{index + 1:06d}" for index in batch_indices]
            batch_results = batch_method(
                images,
                frame_ids=frame_ids,
                observation_ids=observation_ids,
            )
            if not isinstance(batch_results, Mapping):
                raise ValidationFailure("OCR batch adapter returned a non-mapping result")
            for index, frame_id in zip(batch_indices, frame_ids, strict=True):
                result = batch_results.get(frame_id)
                if result is not None and not isinstance(result, OCRObservation):
                    raise ValidationFailure("OCR batch adapter returned an invalid observation")
                completed_by_key[frame_keys[index]] = result
                cache_entries[frame_keys[index]] = result
            # Persist every chunk.  This is intentionally independent of the
            # observation checkpoint interval: one batch is the worker's
            # failure/retry boundary and is small enough to keep the cache
            # writes bounded on long recordings.
            flush_incremental_checkpoint()
    elif representative_indices:
        def recognize_one(index: int) -> tuple[str, Any]:
            frame = ordered_frames[index]
            frame_id = str(frame["frame_id"])
            result = run_optional_ocr(
                adapter,
                project_dir / str(frame["full_frame_path"]),
                frame_id=frame_id,
                observation_id=f"O{index + 1:06d}",
            )
            observation = result.observations[0] if result.observations else None
            if observation is not None and not isinstance(observation, OCRObservation):
                raise ValidationFailure("OCR adapter returned an invalid observation")
            return frame_keys[index], observation

        with ThreadPoolExecutor(
            max_workers=ocr_worker_count, thread_name_prefix="vsr-ocr"
        ) as pool:
            pending_futures = {
                pool.submit(recognize_one, index): index for index in representative_indices
            }
            try:
                for completed_count, future in enumerate(as_completed(pending_futures), 1):
                    frame_key, observation = future.result()
                    completed_by_key[frame_key] = observation
                    cache_entries[frame_key] = observation
                    if completed_count % _ocr_checkpoint_flush_interval() == 0:
                        flush_incremental_checkpoint()
            except BaseException:
                # Preserve any observations that completed before a worker
                # failed or the user interrupted the run; a retry can then
                # resume only the missing pixels.
                flush_incremental_checkpoint()
                raise

    cache_entries.update(completed_by_key)
    # A complete cache hit has no new OCR state to persist.  Avoid rewriting
    # the same compact JSON (and fsyncing it) on every warm visual resume.  A
    # partial hit still takes the normal write path so newly recognized pixels
    # remain durable before the stage returns; interruption behavior is
    # unchanged because incremental flushes happen inside that path.
    cache_reused_without_rewrite = not representative_indices and bool(cached)
    cache_written = False
    if not cache_reused_without_rewrite:
        try:
            cache_written = _write_ocr_cache(
                cache_path,
                cache_key_value=checkpoint_key,
                source_digest=source_digest,
                adapter_identity=adapter_identity,
                entries=cache_entries,
                cache_limit=cache_limit,
            )
        except (OSError, TypeError, ValueError) as exc:
            # A checkpoint is an acceleration artifact only.  Disk/serialization
            # trouble must never turn an otherwise valid OCR result into a blocked
            # visual stage.
            LOGGER.warning("Unable to persist OCR checkpoint: %s", exc)
    shared_cache_written = False
    # A local warm hit normally has a matching shared receipt from the same
    # run.  If that optional receipt was evicted between runs, backfill only
    # the missing shared file; never rewrite an existing warm receipt.
    shared_cache_needs_backfill = (
        cache_reused_without_rewrite
        and shared_cache_path is not None
        and not shared_cache_path.is_file()
    )
    if shared_cache_path is not None and (cache_written or shared_cache_needs_backfill):
        try:
            shared_cache_written = _write_ocr_cache(
                shared_cache_path,
                cache_key_value=checkpoint_key,
                source_digest=source_digest,
                adapter_identity=adapter_identity,
                entries=cache_entries,
                cache_limit=shared_cache_limit,
            )
            if shared_cache_written:
                _prune_shared_json_cache(
                    shared_cache_path.parent,
                    current_path=shared_cache_path,
                    cache_limit=shared_cache_limit,
                )
        except (OSError, TypeError, ValueError) as exc:
            LOGGER.warning("Unable to persist OCR shared checkpoint: %s", exc)
    for frame_key, indices in pending.items():
        observation = completed_by_key.get(frame_key)
        for index in indices:
            if observation is None:
                continue
            frame_id = str(ordered_frames[index]["frame_id"])
            ocr_by_frame[frame_id] = replace(
                observation,
                frame_id=frame_id,
                observation_id=f"O{index + 1:06d}",
            )
    return ocr_by_frame, {
        "status": "completed",
        "cache_path": str(cache_path),
        "shared_cache_path": str(shared_cache_path) if shared_cache_path is not None else None,
        "cache_hit_count": cache_hits,
        "shared_cache_hit": shared_cache_hit,
        "cache_miss_count": representative_count,
        "prefetch_hit_count": prefetched_hits,
        "prefetch_submitted_count": int((prefetch_metrics or {}).get("submitted_count", 0)),
        "prefetch_recognized_count": int((prefetch_metrics or {}).get("recognized_count", 0)),
        "prefetch_batch_count": int((prefetch_metrics or {}).get("batch_count", 0)),
        "prefetch_worker_count": int((prefetch_metrics or {}).get("worker_count", 1)),
        "prefetch_error": (prefetch_metrics or {}).get("error"),
        "cache_deduplicated_count": max(0, len(pending) - len(representative_indices)),
        "cache_written": cache_written,
        "shared_cache_written": shared_cache_written,
        "cache_reused_without_rewrite": cache_reused_without_rewrite,
        "checkpoint_flush_count": checkpoint_flush_count,
        "checkpoint_write_failures": checkpoint_write_failures,
        "worker_count": ocr_worker_count,
        "batch_size": _ocr_batch_size() if callable(batch_method) else None,
        "batch_count": (
            (len(representative_indices) + _ocr_batch_size() - 1) // _ocr_batch_size()
            if callable(batch_method) and representative_indices
            else 0
        ),
        "persistent_worker_used": bool(getattr(adapter, "persistent_worker_used", False)),
        "persistent_worker_fallback_count": int(
            getattr(adapter, "persistent_worker_fallback_count", 0)
        ),
        "observation_count": len(ocr_by_frame),
    }


def _load_or_extract_visual_frames(
    source: Path,
    project_dir: Path,
    output_dir: Path,
    requested_times_ms: Sequence[int],
    *,
    duration_ms: int,
    max_workers: int,
    source_sha256: str | None = None,
    shared_frames: Sequence[Any] = (),
    shared_cache_dir: Path | None = None,
    worker_pool: ThreadPoolExecutor | None = None,
    on_frame: Callable[[Any], None] | None = None,
) -> tuple[Any, ...]:
    """Reuse raw, measured PNG frames across an interrupted visual rebuild.

    The checkpoint stores the pre-metadata PNG bytes plus the measured timing
    record. Every cache entry is verified by size and SHA-256 before restore;
    a malformed, partial, or conflicting entry falls back to FFmpeg extraction.
    The optional shared cache applies the same validation across output
    directories. Metadata is never cached here, so a restored frame receives
    the same fresh canonical envelope as a cold run.
    """

    from .frame_extract import ExtractedFrame, extract_frames, format_frame_timestamp
    from .frame_quality import normalized_pixel_hash

    ordered_times = tuple(sorted(int(value) for value in requested_times_ms))
    if not ordered_times:
        return ()
    source_digest = source_sha256 or sha256_file(source)
    ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"
    ffmpeg_version = _tool_version(ffmpeg_path) or "unavailable"
    # Worker count only changes scheduling (including the bounded decoder
    # thread budget). It cannot change the guarded evidence semantics,
    # measured PTS, or validated PNG pixels, so it must not split an otherwise
    # identical receipt into duplicate cache trees when a host is tuned or a
    # resume runs on a machine with a different CPU count.
    key = cache_key(
        # Requested schedules now exclude an unmeasurable container tail; do
        # not restore a frame checkpoint created before that contract change.
        "visual-frames-v3-tail-guard",
        source_digest,
        duration_ms,
        ordered_times,
        "batch=true",
        "video-stream=0",
        ffmpeg_version,
        __version__,
    )
    checkpoint_root = project_dir / ".state" / "checkpoints" / "visual-frames"
    cache_dir = checkpoint_root / key
    manifest_path = cache_dir / "manifest.json"
    shared_checkpoint_root: Path | None = None
    shared_cache_path: Path | None = None
    shared_manifest_path: Path | None = None
    if shared_cache_dir is not None:
        candidate_root = Path(shared_cache_dir).expanduser()
        try:
            candidate_root.mkdir(parents=True, exist_ok=True)
            if candidate_root.is_dir():
                shared_checkpoint_root = candidate_root / "frames"
                shared_cache_path = shared_checkpoint_root / key
                shared_manifest_path = shared_cache_path / "manifest.json"
                if shared_cache_path.resolve() == cache_dir.resolve():
                    shared_checkpoint_root = None
                    shared_cache_path = None
                    shared_manifest_path = None
        except OSError as exc:
            LOGGER.warning("Visual shared cache unavailable at %s: %s", candidate_root, exc)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_indexes = {requested_ms: index for index, requested_ms in enumerate(ordered_times)}
    published_indexes: set[int] = set()

    def publish_frame(frame: Any) -> None:
        """Expose one measured frame at a stable evidence path for prefetch."""

        if on_frame is None:
            return
        requested_ms = getattr(frame, "requested_ms", None)
        if not isinstance(requested_ms, int):
            return
        index = requested_indexes.get(requested_ms)
        if index is None or index in published_indexes:
            return
        frame_id = f"F{index + 1:06d}"
        destination = output_dir / (
            f"{frame_id}__{format_frame_timestamp(int(frame.actual_ms))}__full.png"
        )
        source_path = Path(frame.path)
        if source_path.resolve() != destination.resolve():
            if destination.exists():
                if normalized_pixel_hash(destination) != normalized_pixel_hash(source_path):
                    raise ValidationFailure(
                        f"Streaming frame has conflicting pixels: {destination.name}"
                    )
            else:
                try:
                    os.link(source_path, destination)
                except OSError:
                    _copy_file_atomic(source_path, destination, durable=False)
        published_indexes.add(index)
        on_frame(replace(frame, frame_id=frame_id, path=destination.resolve()))

    cache_limit = _visual_frame_cache_limit()
    shared_cache_limit = _visual_shared_cache_limit()
    if shared_cache_limit <= 0:
        shared_checkpoint_root = None
        shared_cache_path = None
        shared_manifest_path = None

    def restore_cached(
        cache_location: Path, manifest_location: Path, *, max_bytes: int
    ) -> tuple[Any, ...] | None:
        try:
            cached = json.loads(manifest_location.read_text(encoding="utf-8"))
            if not isinstance(cached, dict):
                return None
            if cached.get("schema_version") != "1.0" or cached.get("cache_key") != key:
                return None
            if cached.get("source_sha256") != source_digest or cached.get("ffmpeg_version") != ffmpeg_version:
                return None
            if int(cached.get("cache_bytes", 0)) > max_bytes:
                return None
            raw_frames = cached.get("frames")
            if not isinstance(raw_frames, list) or len(raw_frames) != len(ordered_times):
                return None
            if tuple(int(value) for value in cached.get("requested_times_ms", [])) != ordered_times:
                return None
            restore_jobs: list[tuple[Path, Path, int, str, dict[str, Any]]] = []
            seen_filenames: set[str] = set()
            for index, raw in enumerate(raw_frames):
                if not isinstance(raw, dict):
                    return None
                filename = raw.get("filename")
                requested_raw = raw.get("requested_ms")
                if (
                    not isinstance(filename, str)
                    or Path(filename).name != filename
                    or filename in seen_filenames
                    or not isinstance(requested_raw, int)
                    or requested_raw != ordered_times[index]
                ):
                    return None
                seen_filenames.add(filename)
                cached_path = cache_location / filename
                expected_size = raw.get("size_bytes")
                if (
                    cached_path.is_symlink()
                    or not cached_path.is_file()
                    or not isinstance(expected_size, int)
                    or expected_size != cached_path.stat().st_size
                ):
                    return None
                expected_hash = raw.get("sha256")
                if not isinstance(expected_hash, str):
                    return None
                destination = output_dir / filename
                restore_jobs.append(
                    (cached_path, destination, expected_size, expected_hash, raw)
                )
            # Checkpoint restores are independent byte copies. Keep the
            # reconstructed frame order deterministic while avoiding a serial
            # fsync/hash pass when a long run resumes from a valid cache.
            with _executor_context(
                worker_pool,
                max_workers=max_workers,
                thread_name_prefix="vsr-frame-restore",
            ) as pool:
                restore_futures = [
                    pool.submit(
                        # A restored frame is still raw immutable checkpoint
                        # data at this point.  Materialize a hardlink when the
                        # cache and evidence tree share a volume; the later
                        # metadata writer replaces the evidence pathname
                        # atomically, so the checkpoint inode remains
                        # untouched.  The helper falls back to the guarded
                        # copy path across volumes/filesystems.
                        _link_or_copy_file_atomic,
                        cached_path,
                        destination,
                        expected_sha256=expected_hash,
                        expected_size=expected_size,
                        durable=False,
                    )
                    for cached_path, destination, expected_size, expected_hash, _raw in restore_jobs
                ]
                for future in restore_futures:
                    future.result()
            restored: list[Any] = []
            for _cached_path, destination, _expected_size, _expected_hash, raw in restore_jobs:
                restored.append(
                    ExtractedFrame(
                        frame_id=str(raw["frame_id"]),
                        path=destination.resolve(),
                        requested_ms=int(raw["requested_ms"]),
                        actual_ms=int(raw["actual_ms"]),
                        raw_pts=int(raw["raw_pts"]),
                        time_base=(
                            str(raw["time_base"]) if raw.get("time_base") is not None else None
                        ),
                        frame_index=(
                            int(raw["frame_index"]) if raw.get("frame_index") is not None else None
                        ),
                        offset_ms=int(raw["offset_ms"]),
                        timestamp_source=str(raw["timestamp_source"]),
                        width=int(raw["width"]),
                        height=int(raw["height"]),
                    )
                )
            LOGGER.info("Reused raw visual frame checkpoint: %s", cache_location)
            return tuple(restored)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            LOGGER.info("Visual frame checkpoint miss or corruption: %s", cache_location)
            return None

    restored = restore_cached(cache_dir, manifest_path, max_bytes=cache_limit)
    if restored is None and shared_cache_path is not None and shared_manifest_path is not None:
        restored = restore_cached(
            shared_cache_path, shared_manifest_path, max_bytes=shared_cache_limit
        )
        if restored is not None:
            LOGGER.info("Reused cross-project visual frame checkpoint: %s", shared_cache_path)
    if restored is not None:
        for frame in restored:
            publish_frame(frame)
        return restored

    def restore_prior_schedule_frames() -> dict[int, ExtractedFrame]:
        """Reuse verified frames from an earlier request schedule.

        The schedule-bound manifest remains the fast path above.  When a
        transcript/context change adds or removes one request, however, that
        key intentionally changes and would otherwise force every adaptive
        timestamp through FFmpeg again.  Prior manifests are immutable,
        source-digest-bound receipts; copying only exact requested timestamps
        preserves their measured PTS, dimensions, and lossless PNG bytes while
        leaving new timestamps on the guarded extraction path.
        """

        roots: list[tuple[Path, int]] = []
        seen_roots: set[Path] = set()
        for root, limit in (
            (checkpoint_root, cache_limit),
            (shared_checkpoint_root, shared_cache_limit),
        ):
            if root is None or limit <= 0 or not root.is_dir():
                continue
            try:
                resolved_root = root.resolve()
            except OSError:
                continue
            if resolved_root in seen_roots:
                continue
            seen_roots.add(resolved_root)
            roots.append((root, limit))
        if not roots:
            return {}
        try:
            # Keep the current shared receipt eligible for the prior-schedule
            # scan.  A bounded shared schedule can intentionally contain only
            # a subset of frames (the exact restore above rejects it by length),
            # and those records are still safe to reuse one-by-one.  The
            # project-local exact manifest remains excluded because it is the
            # current run's own checkpoint and has already been validated.
            current_manifests = {manifest_path.resolve()}
            manifests = [
                (path, limit)
                for root, limit in roots
                for path in root.glob("*/manifest.json")
                if path.resolve() not in current_manifests
            ]
            manifests.sort(
                key=lambda item: (
                    -item[0].stat().st_mtime_ns,
                    item[0].as_posix(),
                )
            )
        except OSError:
            return {}

        # A project can accumulate many interrupted schedules.  Keep this
        # lookup bounded; newer manifests are the most likely to overlap the
        # current context and old entries remain available after a later miss.
        manifests = manifests[:64]
        candidates: dict[int, tuple[Path, int, str, dict[str, Any]]] = {}
        conflicts: set[int] = set()
        for prior_manifest, manifest_limit in manifests:
            try:
                payload = json.loads(prior_manifest.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                if (
                    payload.get("schema_version") != "1.0"
                    or payload.get("source_sha256") != source_digest
                    or payload.get("ffmpeg_version") != ffmpeg_version
                ):
                    continue
                if int(payload.get("cache_bytes", 0)) > manifest_limit:
                    continue
                raw_frames = payload.get("frames")
                if not isinstance(raw_frames, list):
                    continue
                for raw in raw_frames:
                    if not isinstance(raw, dict):
                        continue
                    requested_raw = raw.get("requested_ms")
                    filename = raw.get("filename")
                    expected_size = raw.get("size_bytes")
                    expected_hash = raw.get("sha256")
                    actual_raw = raw.get("actual_ms")
                    raw_pts = raw.get("raw_pts")
                    offset_raw = raw.get("offset_ms")
                    width_raw = raw.get("width")
                    height_raw = raw.get("height")
                    timestamp_source = raw.get("timestamp_source")
                    if (
                        not isinstance(requested_raw, int)
                        or requested_raw not in ordered_times
                        or not isinstance(filename, str)
                        or Path(filename).name != filename
                        or not isinstance(expected_size, int)
                        or expected_size <= 0
                        or not isinstance(expected_hash, str)
                        or len(expected_hash) != 64
                        or not isinstance(actual_raw, int)
                        or actual_raw < 0
                        or not isinstance(raw_pts, int)
                        or not isinstance(offset_raw, int)
                        or not isinstance(width_raw, int)
                        or width_raw <= 0
                        or not isinstance(height_raw, int)
                        or height_raw <= 0
                        or not isinstance(timestamp_source, str)
                        or not timestamp_source
                    ):
                        continue
                    cached_path = prior_manifest.parent / filename
                    if (
                        cached_path.is_symlink()
                        or not cached_path.is_file()
                        or cached_path.stat().st_size != expected_size
                    ):
                        continue
                    signature = (
                        expected_size,
                        expected_hash,
                        actual_raw,
                        raw_pts,
                        raw.get("time_base"),
                        timestamp_source,
                        width_raw,
                        height_raw,
                    )
                    existing = candidates.get(requested_raw)
                    if existing is not None:
                        existing_signature = (
                            existing[1],
                            existing[2],
                            existing[3].get("actual_ms"),
                            existing[3].get("raw_pts"),
                            existing[3].get("time_base"),
                            existing[3].get("timestamp_source"),
                            existing[3].get("width"),
                            existing[3].get("height"),
                        )
                        if existing_signature != signature:
                            conflicts.add(requested_raw)
                        continue
                    candidates[requested_raw] = (
                        cached_path,
                        expected_size,
                        expected_hash,
                        raw,
                    )
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue

        for requested_ms in conflicts:
            candidates.pop(requested_ms, None)
        if not candidates:
            return {}

        restored_by_request: dict[int, ExtractedFrame] = {}
        restore_jobs: list[tuple[int, Path, Path, int, str, dict[str, Any]]] = []
        for index, requested_ms in enumerate(ordered_times):
            candidate = candidates.get(requested_ms)
            if candidate is None:
                continue
            cached_path, expected_size, expected_hash, raw = candidate
            destination = output_dir / (
                f"F{index + 1:06d}__{format_frame_timestamp(int(raw['actual_ms']))}__full.png"
            )
            restore_jobs.append(
                (index, cached_path, destination, expected_size, expected_hash, raw)
            )

        def restore_one(
            job: tuple[int, Path, Path, int, str, dict[str, Any]],
        ) -> tuple[int, ExtractedFrame]:
            index, cached_path, destination, expected_size, expected_hash, raw = job
            # Prior-schedule frames have the same immutable-checkpoint
            # lifecycle as the exact schedule cache.  Hardlinking avoids a
            # second full PNG write on warm visual resumes while the atomic
            # metadata replacement below keeps the cache inode immutable.
            _link_or_copy_file_atomic(
                cached_path,
                destination,
                expected_sha256=expected_hash,
                expected_size=expected_size,
                durable=False,
            )
            return index, ExtractedFrame(
                frame_id=f"F{index + 1:06d}",
                path=destination.resolve(),
                requested_ms=int(raw["requested_ms"]),
                actual_ms=int(raw["actual_ms"]),
                raw_pts=int(raw["raw_pts"]),
                time_base=(str(raw["time_base"]) if raw.get("time_base") is not None else None),
                frame_index=(
                    int(raw["frame_index"]) if raw.get("frame_index") is not None else None
                ),
                offset_ms=int(raw["offset_ms"]),
                timestamp_source=str(raw["timestamp_source"]),
                width=int(raw["width"]),
                height=int(raw["height"]),
            )

        with _executor_context(
            worker_pool,
            max_workers=max_workers,
            thread_name_prefix="vsr-frame-prior-restore",
        ) as pool:
            futures = [pool.submit(restore_one, job) for job in restore_jobs]
            for future in futures:
                try:
                    index, frame = future.result()
                except (OSError, ValueError, TypeError, ValidationFailure):
                    continue
                restored_by_request[int(frame.requested_ms)] = frame
                publish_frame(frame)
        if restored_by_request:
            LOGGER.info(
                "Reused %d measured visual frames from prior schedules",
                len(restored_by_request),
            )
        return restored_by_request

    # Shared survey output is exact-safe for hard-cut and periodic/contextual
    # branches. Materialize those PNGs first, then decode any adaptive or
    # requests through the guarded exact path. This preserves the original
    # measured timing contract while avoiding a second full decode for the
    # safety scaffold on a cold run.
    shared_by_request: dict[int, Any] = {}
    for frame in shared_frames:
        if getattr(frame, "branch", None) not in {"hard", "periodic"}:
            continue
        requested_ms = getattr(frame, "requested_ms", None)
        path = Path(getattr(frame, "path", ""))
        if not isinstance(requested_ms, int) or requested_ms not in ordered_times:
            continue
        if requested_ms in shared_by_request or not path.is_file() or path.stat().st_size == 0:
            continue
        timing = getattr(frame, "timing", None)
        if timing is None or int(getattr(timing, "width", 0)) <= 0 or int(
            getattr(timing, "height", 0)
        ) <= 0:
            continue
        shared_by_request[requested_ms] = frame

    extracted_by_index: dict[int, ExtractedFrame] = {}
    missing: list[tuple[int, int]] = []
    shared_jobs: list[tuple[int, int, Any, Path]] = []
    for index, requested_ms in enumerate(ordered_times):
        shared = shared_by_request.get(requested_ms)
        if shared is None:
            missing.append((index, requested_ms))
            continue
        timing = shared.timing
        frame_id = f"F{index + 1:06d}"
        destination = output_dir / (
            f"{frame_id}__{format_frame_timestamp(int(timing.actual_ms))}__full.png"
        )
        shared_jobs.append((index, requested_ms, shared, destination))

    def materialize_shared(
        job: tuple[int, int, Any, Path],
    ) -> tuple[int, ExtractedFrame]:
        index, requested_ms, shared, destination = job
        timing = shared.timing
        shared_path = Path(shared.path)
        if destination.exists():
            if normalized_pixel_hash(destination) != normalized_pixel_hash(shared_path):
                raise ValidationFailure(
                    f"Existing evidence frame has conflicting pixels: {destination.name}"
                )
        else:
            # The shared survey directory is a run-local temporary tree and is
            # removed immediately after materialization.  Move within the same
            # volume instead of copying every full-resolution PNG twice; if a
            # caller places checkpoints and evidence on different volumes,
            # retain the guarded atomic-copy fallback.
            try:
                os.replace(shared_path, destination)
            except OSError:
                _copy_file_atomic(shared_path, destination, durable=False)
        frame_id = f"F{index + 1:06d}"
        return index, ExtractedFrame(
            frame_id=frame_id,
            path=destination.resolve(),
            requested_ms=requested_ms,
            actual_ms=int(timing.actual_ms),
            raw_pts=int(timing.raw_pts),
            time_base=(str(timing.time_base) if timing.time_base is not None else None),
            frame_index=None,
            offset_ms=int(timing.actual_ms) - requested_ms,
            timestamp_source=str(timing.timestamp_source),
            width=int(timing.width),
            height=int(timing.height),
        )

    if shared_jobs:
        with _executor_context(
            worker_pool,
            max_workers=max_workers,
            thread_name_prefix="vsr-shared-frame",
        ) as pool:
            for index, shared_frame in pool.map(materialize_shared, shared_jobs):
                extracted_by_index[index] = shared_frame
                publish_frame(shared_frame)

    prior_by_request = restore_prior_schedule_frames()
    for index, requested_ms in enumerate(ordered_times):
        if index in extracted_by_index:
            continue
        prior = prior_by_request.get(requested_ms)
        if prior is not None:
            extracted_by_index[index] = prior

    missing = [
        (index, requested_ms)
        for index, requested_ms in enumerate(ordered_times)
        if index not in extracted_by_index
    ]
    if missing:
        with tempfile.TemporaryDirectory(prefix=".vsr-missing-frames-", dir=output_dir) as name:
            pending_dir = Path(name)
            missing_times = [requested_ms for _index, requested_ms in missing]
            # Keep compatibility with lightweight test/custom extractors that
            # implement the historical keyword surface while the production
            # extractor receives the absolute FFmpeg path on Windows.
            extract_kwargs: dict[str, Any] = {
                "max_workers": max_workers,
                "batch": True,
                "timeout_seconds": 600.0,
            }
            extract_parameters: Mapping[str, inspect.Parameter]
            try:
                extract_parameters = inspect.signature(extract_frames).parameters
            except (TypeError, ValueError):
                extract_parameters = {}
            if "ffmpeg_bin" in extract_parameters:
                extract_kwargs["ffmpeg_bin"] = ffmpeg_path
            if worker_pool is not None and "worker_pool" in extract_parameters:
                extract_kwargs["worker_pool"] = worker_pool
            if on_frame is not None and "on_frame" in extract_parameters:
                extract_kwargs["on_frame"] = publish_frame
            decoded = extract_frames(source, missing_times, pending_dir, **extract_kwargs)
            for (index, _requested_ms), frame in zip(missing, decoded, strict=True):
                frame_id = f"F{index + 1:06d}"
                destination = output_dir / (
                    f"{frame_id}__{format_frame_timestamp(frame.actual_ms)}__full.png"
                )
                if destination.exists():
                    if normalized_pixel_hash(destination) != normalized_pixel_hash(frame.path):
                        raise ValidationFailure(
                            f"Existing evidence frame has conflicting pixels: {destination.name}"
                        )
                    Path(frame.path).unlink(missing_ok=True)
                else:
                    os.replace(frame.path, destination)
                extracted_by_index[index] = replace(
                    frame,
                    frame_id=frame_id,
                    path=destination.resolve(),
                )
    extracted = tuple(extracted_by_index[index] for index in range(len(ordered_times)))
    frame_sizes = [(frame, Path(frame.path).stat().st_size) for frame in extracted]
    total_bytes = sum(size for _frame, size in frame_sizes)
    # Keep the complete project-local checkpoint subject to its own budget,
    # but do not return before the optional bounded cross-project cache gets a
    # chance to retain a temporally spread subset.  Long recordings commonly
    # exceed the 512 MiB local raw-frame budget while still having room for a
    # useful shared acceleration receipt.
    local_checkpoint_enabled = cache_limit > 0 and total_bytes <= cache_limit
    if not local_checkpoint_enabled:
        LOGGER.info(
            "Skipping raw visual frame checkpoint: %d bytes exceeds budget %d",
            total_bytes,
            cache_limit,
        )
    records: list[dict[str, Any]] = []
    for frame, size_bytes in frame_sizes:
        frame_path = Path(frame.path)
        records.append(
            {
                "frame_id": frame.frame_id,
                "filename": frame_path.name,
                "requested_ms": frame.requested_ms,
                "actual_ms": frame.actual_ms,
                "raw_pts": frame.raw_pts,
                "time_base": frame.time_base,
                "frame_index": frame.frame_index,
                "offset_ms": frame.offset_ms,
                "timestamp_source": frame.timestamp_source,
                "width": frame.width,
                "height": frame.height,
                "size_bytes": size_bytes,
                "sha256": None,
            }
        )
    # When the full local schedule fits, retain the existing complete
    # checkpoint behavior.  If it does not fit, the evidence paths themselves
    # remain the immutable source for the bounded shared receipt below; no
    # duplicate local raw-frame tree is created.
    shared_source_dir = output_dir
    if local_checkpoint_enabled:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Checkpoint copies are independent, but the manifest remains committed
        # in deterministic frame order below. A bounded pool reduces the second
        # full read/write pass over high-resolution PNGs without changing the
        # recorded bytes or allowing unbounded disk pressure.
        with _executor_context(
            worker_pool,
            max_workers=max_workers,
            thread_name_prefix="vsr-frame-checkpoint",
        ) as pool:
            copy_futures = [
                pool.submit(
                    _copy_file_atomic,
                    Path(frame.path),
                    cache_dir / str(record["filename"]),
                )
                for frame, record in zip(extracted, records, strict=True)
            ]
            copied = [future.result() for future in copy_futures]
        for record, (digest, copied_size) in zip(records, copied, strict=True):
            if copied_size != int(record["size_bytes"]):
                raise ValidationFailure("raw visual frame changed while checkpointing")
            record["sha256"] = digest
        atomic_write_json(
            manifest_path,
            {
                "schema_version": "1.0",
                "cache_key": key,
                "source_sha256": source_digest,
                "ffmpeg_version": ffmpeg_version,
                "requested_times_ms": list(ordered_times),
                "cache_bytes": total_bytes,
                "frames": records,
            },
        )
        _prune_visual_frame_checkpoints(
            checkpoint_root,
            current_cache_dir=cache_dir,
            cache_limit=cache_limit,
        )
        shared_source_dir = cache_dir
    if (
        shared_checkpoint_root is not None
        and shared_cache_path is not None
        and shared_manifest_path is not None
        and shared_cache_limit > 0
    ):
        try:
            # A large schedule may exceed the bounded shared budget.  Keep a
            # deterministic, evenly spaced subset in that case so a future
            # project can reuse exact measured frames instead of paying the
            # full extraction cost again.  The project-local checkpoint remains
            # complete; only the optional cross-project receipt is partial.
            shared_records, shared_total_bytes = _bounded_shared_frame_records(
                records, shared_cache_limit
            )
            if not shared_records:
                LOGGER.info(
                    "Skipping visual shared checkpoint: no frame fits budget %d",
                    shared_cache_limit,
                )
                return extracted
            # A schedule that exceeded the project-local budget has no local
            # manifest (and therefore no precomputed SHA values). Hash only the
            # records selected for the bounded shared receipt, preserving the
            # same size/digest gates as a complete checkpoint without forcing a
            # second full read of every long-form frame.
            missing_hash_records = [
                record for record in shared_records if not isinstance(record.get("sha256"), str)
            ]
            if missing_hash_records:
                def hash_shared_record(record: Mapping[str, Any]) -> tuple[str, int]:
                    frame_path = shared_source_dir / str(record["filename"])
                    digest = sha256_file(frame_path)
                    return digest, frame_path.stat().st_size

                with _executor_context(
                    worker_pool,
                    max_workers=max_workers,
                    thread_name_prefix="vsr-shared-frame-hash",
                ) as pool:
                    hash_futures = [
                        pool.submit(hash_shared_record, record)
                        for record in missing_hash_records
                    ]
                    for record, future in zip(missing_hash_records, hash_futures, strict=True):
                        digest, observed_size = future.result()
                        if observed_size != int(record["size_bytes"]):
                            raise ValidationFailure(
                                "raw visual frame changed while hashing shared checkpoint"
                            )
                        record["sha256"] = digest
            # Shared visual checkpoints are acceleration-only and immutable.
            # Before copying a new schedule, index exact content already
            # present in sibling schedules.  Reusing an existing inode for an
            # identical SHA/size avoids another full PNG write while retaining
            # independent manifest paths; a stale/corrupt candidate is simply
            # rejected by the guarded materializer and falls back to the
            # current project checkpoint.
            shared_content_index: dict[tuple[str, int], Path] = {}
            try:
                current_shared_resolved = shared_cache_path.resolve()
                for prior_manifest in shared_checkpoint_root.glob("*/manifest.json"):
                    if prior_manifest.parent.resolve() == current_shared_resolved:
                        continue
                    try:
                        prior_payload = json.loads(prior_manifest.read_text(encoding="utf-8"))
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if not isinstance(prior_payload, Mapping):
                        continue
                    prior_frames = prior_payload.get("frames")
                    if not isinstance(prior_frames, list):
                        continue
                    for prior_raw in prior_frames:
                        if not isinstance(prior_raw, Mapping):
                            continue
                        prior_hash = prior_raw.get("sha256")
                        prior_size = prior_raw.get("size_bytes")
                        prior_filename = prior_raw.get("filename")
                        if (
                            not isinstance(prior_hash, str)
                            or len(prior_hash) != 64
                            or not isinstance(prior_size, int)
                            or prior_size <= 0
                            or not isinstance(prior_filename, str)
                            or Path(prior_filename).name != prior_filename
                        ):
                            continue
                        candidate = prior_manifest.parent / prior_filename
                        if candidate.is_symlink() or not candidate.is_file():
                            continue
                        shared_content_index.setdefault((prior_hash, prior_size), candidate)
            except OSError:
                # Shared-cache indexing is opportunistic; the current project
                # checkpoint remains a complete source for the write.
                shared_content_index = {}

            shared_checkpoint_root.mkdir(parents=True, exist_ok=True)
            shared_cache_path.mkdir(parents=True, exist_ok=True)

            def materialize_shared_checkpoint(record: Mapping[str, Any]) -> tuple[str, int]:
                expected_hash = str(record["sha256"])
                expected_size = int(record["size_bytes"])
                destination = shared_cache_path / str(record["filename"])
                candidate = shared_content_index.get((expected_hash, expected_size))
                if candidate is not None:
                    try:
                        return _link_or_copy_file_atomic(
                            candidate,
                            destination,
                            expected_sha256=expected_hash,
                            expected_size=expected_size,
                        )
                    except (OSError, ValueError):
                        # A manifest can outlive a manually removed or
                        # modified acceleration file.  Never fail the current
                        # run for that stale optimization hint.
                        pass
                return _link_or_copy_file_atomic(
                    shared_source_dir / str(record["filename"]),
                    destination,
                    expected_sha256=expected_hash,
                    expected_size=expected_size,
                )

            with _executor_context(
                worker_pool,
                max_workers=max_workers,
                thread_name_prefix="vsr-shared-frame-checkpoint",
            ) as pool:
                shared_copy_futures = [
                    pool.submit(materialize_shared_checkpoint, record)
                    for record in shared_records
                ]
                for future in shared_copy_futures:
                    future.result()
            atomic_write_json(shared_manifest_path, {
                "schema_version": "1.0",
                "cache_key": key,
                "source_sha256": source_digest,
                "ffmpeg_version": ffmpeg_version,
                "requested_times_ms": [int(record["requested_ms"]) for record in shared_records],
                "cache_bytes": shared_total_bytes,
                "frames": shared_records,
                "partial": len(shared_records) != len(records),
            })
            # A schedule may have been written under a larger budget in an
            # earlier run. Remove now-unreferenced raw files after the manifest
            # commit so a partial rewrite cannot leave hidden bytes outside the
            # advertised shared-cache budget. Never follow symlinks or a cache
            # directory that resolves outside its bounded parent.
            retained_names = {str(record["filename"]) for record in shared_records}
            if not shared_cache_path.is_symlink():
                resolved_shared_dir = shared_cache_path.resolve()
                if resolved_shared_dir.parent == shared_checkpoint_root.resolve():
                    for stale_path in shared_cache_path.glob("*.png"):
                        if (
                            stale_path.name not in retained_names
                            and not stale_path.is_symlink()
                            and stale_path.is_file()
                        ):
                            stale_path.unlink()
            _prune_visual_frame_checkpoints(
                shared_checkpoint_root,
                current_cache_dir=shared_cache_path,
                cache_limit=shared_cache_limit,
            )
        except (OSError, TypeError, ValueError, ValidationFailure) as exc:
            # Shared frames are acceleration state only. A cache write failure
            # must leave the project-local checkpoint and evidence untouched.
            LOGGER.warning("Unable to persist visual shared checkpoint: %s", exc)
    return extracted


def _visual_frame_workers() -> int:
    """Choose a bounded decoder pool without making hardware a correctness input."""

    override = os.environ.get("VSR_FRAME_EXTRACT_WORKERS", "").strip()
    if override:
        try:
            return max(1, min(8, int(override)))
        except ValueError:
            LOGGER.warning("Ignoring invalid VSR_FRAME_EXTRACT_WORKERS=%r", override)
    # Keep the automatic decoder pool conservative: FFmpeg does its own codec
    # threading, and same-source A/B runs showed no benefit from eight default
    # seeks.  ``extract_frames`` derives each process's thread count from this
    # pool size, so the aggregate decoder budget remains near the machine's
    # logical CPU count.  Hosts with measured characteristics can opt into up
    # to eight workers explicitly through VSR_FRAME_EXTRACT_WORKERS above.
    # This is scheduling only: requested PTS, measured timing, and pixel bytes
    # remain authoritative evidence fields.
    return max(1, min(4, (os.cpu_count() or 1) // 2))


def _visual_analysis_workers() -> int:
    """Choose a separate read/analysis pool for already-decoded PNG frames."""

    override = os.environ.get("VSR_FRAME_ANALYSIS_WORKERS", "").strip()
    if override:
        try:
            return max(1, min(8, int(override)))
        except ValueError:
            LOGGER.warning("Ignoring invalid VSR_FRAME_ANALYSIS_WORKERS=%r", override)
    # Analysis workers only hold a one-frame overlap per contiguous chunk; they
    # do not launch decoder subprocesses.  A wider bounded pool is therefore
    # safe on hosts where the FFmpeg extraction pool must remain conservative.
    return max(1, min(8, (os.cpu_count() or 1) // 2))


def _visual_crop_workers() -> int:
    """Choose a bounded pool for independent localized crop preparation.

    Crop preparation reopens each parent PNG, encodes a derived lossless PNG,
    and computes quality/hash fields.  It is independent of exact FFmpeg
    extraction, so sharing the four-worker decoder pool unnecessarily serialized
    this CPU/disk-heavy phase on hosts with more logical CPUs.  Keep the same
    conservative eight-worker ceiling as deterministic frame analysis, with an
    explicit override for measured hosts.  This setting is scheduling only and
    cannot change crop pixels, provenance, or validator semantics.
    """

    override = os.environ.get("VSR_CROP_PREP_WORKERS", "").strip()
    if override:
        try:
            return max(1, min(8, int(override)))
        except ValueError:
            LOGGER.warning("Ignoring invalid VSR_CROP_PREP_WORKERS=%r", override)
    return _visual_analysis_workers()


def _guarded_motion_only_enabled() -> bool:
    """Return whether conservative motion-only coalescing is enabled.

    This is deliberately opt-in.  A camera tile, presenter movement, or UI
    shimmer can exceed the ordinary pixel-change threshold without changing
    the visible slide/content state, but an overly broad downgrade could hide
    a subtle one-line edit.  Keeping the switch off by default preserves the
    historical quality-first behavior until a measured run explicitly opts
    into the guarded A/B path.
    """

    return os.environ.get("VSR_GUARDED_MOTION_DEDUP", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_guarded_motion_only(
    difference: Any,
    *,
    current_dhash: str,
    next_dhash: str | None,
    ocr_stable: bool,
    selection_reason: str,
    width: int,
    height: int,
    is_boundary: bool,
) -> bool:
    """Recognize only low-risk presenter/UI motion for optional coalescing.

    Every guard is intentionally conjunctive: OCR must be available and
    unchanged, the perceptual change must be small, and all changed regions
    must touch an outer overlay band.  Scene/transcript/periodic/sequence
    boundaries and first/last frames are never downgraded.  The next frame's
    dHash must remain close, which rejects one-frame content changes that are
    followed by a stable new slide.
    """

    if not _guarded_motion_only_enabled() or difference is None or not ocr_stable:
        return False
    if is_boundary:
        return False
    protected_tokens = (
        "periodic",
        "sequence",
        "transcript",
        "scene",
        "chapter",
        "ocr",
        "before",
        "action",
        "after",
        "deictic",
    )
    reason = selection_reason.casefold()
    if any(token in reason for token in protected_tokens):
        return False
    if not next_dhash or len(current_dhash) != len(next_dhash):
        return False
    try:
        current_value = int(current_dhash, 16)
        next_value = int(next_dhash, 16)
    except ValueError:
        return False
    if (current_value ^ next_value).bit_count() > 4:
        return False
    if (
        int(getattr(difference, "perceptual_hamming", 64)) > 8
        or float(getattr(difference, "changed_pixel_ratio", 1.0)) > 0.08
        or float(getattr(difference, "maximum_region_change", 1.0)) > 0.25
        or float(getattr(difference, "mean_pixel_difference", 1.0)) > 0.08
        or float(getattr(difference, "edge_difference", 1.0)) > 0.12
    ):
        return False
    if width <= 0 or height <= 0 or not getattr(difference, "regions", ()):
        return False
    # Presenter tiles, webcam strips, and browser chrome usually occupy an
    # outer band.  Requiring every changed region to touch one makes this
    # classifier reject central slide/text edits even when OCR misses them.
    for region in difference.regions:
        x, y, w, h = region.xywh
        touches_overlay = (
            x <= width * 0.18
            or y <= height * 0.18
            or x + w >= width * 0.82
            or y + h >= height * 0.82
        )
        if not touches_overlay:
            return False
    return True


def _visual_survey_ffmpeg_threads() -> int:
    """Bound the single full-duration survey decoder's codec threads.

    The shared hard-cut/adaptive survey is one long FFmpeg process rather than
    the independent exact-seek pool.  A small codec-thread bound avoids either
    FFmpeg's unbounded default or a second process storm, while retaining an
    explicit host override for codec/GPU-specific benchmarking.
    """

    override = os.environ.get("VSR_SURVEY_FFMPEG_THREADS", "").strip()
    if override:
        try:
            return max(1, min(8, int(override)))
        except ValueError:
            LOGGER.warning("Ignoring invalid VSR_SURVEY_FFMPEG_THREADS=%r", override)
    return max(1, min(4, (os.cpu_count() or 1) // 2))


# Off-by-default hardware-decode experiment for the combined visual survey.
# ``VSR_SURVEY_HWACCEL`` accepts only explicitly allowlisted FFmpeg hwaccel
# names; anything else is rejected before a decoder is ever invoked.  The
# effective mode is resolved per source through a guarded parity probe that
# decodes one deterministic frame twice (software and hardware) and compares
# normalized pixel hashes.  Probe outcomes are memoized under a lock and never
# retried: an unsupported, failing, or pixel-divergent accelerator falls back
# to software decode for the rest of the process.  The survey cache identity
# carries the effective decode mode so hardware receipts can never be reused
# as software evidence or vice versa.
_SURVEY_HWACCEL_ALLOWLIST = frozenset({"cuda"})
_SURVEY_HWACCEL_PROBE_LOCK = threading.Lock()
_SURVEY_HWACCEL_PROBES: dict[tuple[str, str], dict[str, str]] = {}


def _survey_hwaccel_request() -> str | None:
    """Return the normalized ``VSR_SURVEY_HWACCEL`` request, if any."""

    return os.environ.get("VSR_SURVEY_HWACCEL", "").strip().casefold() or None


def _probe_survey_hwaccel(mode: str, source: Path) -> dict[str, str]:
    """Compare one deterministic frame decoded by software and by ``mode``.

    The probe is deliberately minimal: two exact extractions at 0 ms and one
    normalized pixel-hash comparison.  Any extraction failure, timeout, or
    pixel divergence marks the mode unavailable instead of guessing.
    """

    from .frame_extract import extract_frame
    from .frame_quality import normalized_pixel_hash

    probe_ms = 0
    with tempfile.TemporaryDirectory(prefix=".vsr-hwaccel-probe-") as temporary:
        base = Path(temporary)
        try:
            software = extract_frame(source, probe_ms, base / "software.png", hwaccel=None)
            hardware = extract_frame(
                source,
                probe_ms,
                base / f"{safe_slug(mode, fallback='hwaccel')}.png",
                hwaccel=mode,
            )
        except (BlockedError, InputError, OSError, ValidationFailure) as exc:
            return {"status": "failed", "detail": str(exc)}
        if normalized_pixel_hash(software.path) != normalized_pixel_hash(hardware.path):
            return {
                "status": "mismatch",
                "detail": (
                    f"hwaccel {mode} frame at {probe_ms} ms differs from software decode"
                ),
            }
    return {"status": "verified", "detail": ""}


def _survey_hwaccel_effective_mode(
    source: Path,
    *,
    source_sha256: str | None = None,
) -> str | None:
    """Resolve the opt-in hardware survey decoder behind a guarded parity probe.

    Returns the allowlisted mode only when its memoized probe verified
    byte-identical pixels against the software decoder for this source digest.
    Every other outcome fails closed to ``None`` without retrying.
    """

    requested = _survey_hwaccel_request()
    if requested is None:
        return None
    if requested not in _SURVEY_HWACCEL_ALLOWLIST:
        LOGGER.warning(
            "Ignoring unsupported VSR_SURVEY_HWACCEL=%r; allowed values: %s",
            requested,
            ", ".join(sorted(_SURVEY_HWACCEL_ALLOWLIST)),
        )
        return None
    digest = source_sha256 or sha256_file(source)
    memo_key = (requested, digest)
    # Single-flight resolution: parallel survey workers must not race two
    # probes for the same source, and a decided source never probes again.
    with _SURVEY_HWACCEL_PROBE_LOCK:
        memo = _SURVEY_HWACCEL_PROBES.get(memo_key)
        if memo is None:
            memo = _probe_survey_hwaccel(requested, source)
            _SURVEY_HWACCEL_PROBES[memo_key] = memo
    if memo["status"] == "verified":
        LOGGER.info(
            "Survey hwaccel %r verified against software decode for source %s",
            requested,
            digest[:12],
        )
        return requested
    LOGGER.warning(
        "Survey hwaccel %r is unavailable (%s); using software decode without retry",
        requested,
        memo["detail"] or memo["status"],
    )
    return None


def _survey_hwaccel_telemetry() -> dict[str, Any]:
    """Summarize the current hwaccel policy and probe outcomes read-only."""

    requested = _survey_hwaccel_request()
    if requested is None:
        return {"requested": None, "effective": None, "status": "disabled", "detail": ""}
    if requested not in _SURVEY_HWACCEL_ALLOWLIST:
        return {
            "requested": requested,
            "effective": None,
            "status": "rejected",
            "detail": f"allowed values: {', '.join(sorted(_SURVEY_HWACCEL_ALLOWLIST))}",
        }
    with _SURVEY_HWACCEL_PROBE_LOCK:
        outcomes = [
            entry for (mode, _digest), entry in _SURVEY_HWACCEL_PROBES.items() if mode == requested
        ]
    failed = next((entry for entry in outcomes if entry["status"] != "verified"), None)
    if failed is not None:
        return {
            "requested": requested,
            "effective": None,
            "status": failed["status"],
            "detail": failed["detail"],
        }
    if not outcomes:
        return {"requested": requested, "effective": None, "status": "pending", "detail": ""}
    return {
        "requested": requested,
        "effective": requested,
        "status": "verified",
        "detail": "",
    }


def _visual_survey_timeout_seconds(duration_ms: int) -> float:
    """Return a duration-aware wall-clock bound for one survey decode pass.

    The combined hard-cut/adaptive/periodic survey decodes the complete source
    once, so the historical fixed 600-second ceiling killed long recordings
    before FFmpeg finished.  The computed bound keeps the 600-second budget
    that already covers up to one hour of media and then scales linearly past
    it (one budget second per six media seconds), capped at one day so the
    value stays bounded.  ``VSR_SURVEY_TIMEOUT_SECONDS`` remains an explicit
    bounded override; empty or invalid values fall back to the computed bound.
    """

    computed = min(86_400.0, max(600.0, duration_ms / 6_000.0))
    override = os.environ.get("VSR_SURVEY_TIMEOUT_SECONDS", "").strip()
    if not override:
        return computed
    try:
        value = float(override)
    except ValueError:
        LOGGER.warning("Ignoring invalid VSR_SURVEY_TIMEOUT_SECONDS=%r", override)
        return computed
    if not math.isfinite(value) or value <= 0:
        LOGGER.warning("Ignoring invalid VSR_SURVEY_TIMEOUT_SECONDS=%r", override)
        return computed
    if value > 86_400.0:
        LOGGER.warning("Clamping VSR_SURVEY_TIMEOUT_SECONDS=%s to 86400", value)
        return 86_400.0
    return value


def _ocr_workers() -> int:
    """Choose a separate bounded pool for independent OCR subprocesses.

    OCR adapters such as Tesseract launch one short-lived process per image and
    do not share FFmpeg's decoder state. Keep the pool separate and reserve
    three quarters of a large host's logical CPUs: the reference 16-thread
    FC101 workload measured a repeatable improvement at twelve workers, while
    sixteen workers only increased process and memory contention. Smaller
    hosts retain the conservative half-CPU policy. An explicit override still
    permits up to sixteen workers for hosts that have measured a benefit.
    """

    override = os.environ.get("VSR_OCR_WORKERS", "").strip()
    if override:
        try:
            return max(1, min(16, int(override)))
        except ValueError:
            LOGGER.warning("Ignoring invalid VSR_OCR_WORKERS=%r", override)
    logical_cpus = os.cpu_count() or 1
    if logical_cpus >= 16:
        return max(1, min(12, (logical_cpus * 3) // 4))
    return max(1, min(8, logical_cpus // 2))


def _asr_cpu_threads() -> int:
    """Choose efficient CPU threading for local Whisper without forcing GPUs.

    CTranslate2's ``0`` means backend auto-selection, which can use logical
    SMT threads and oversubscribe decode kernels on CPU-only hosts.  A bounded
    physical-core approximation is faster on the supported Windows/Linux
    targets while preserving an explicit environment override and leaving CUDA
    callers free to use their GPU runtime.
    """

    override = os.environ.get("VSR_ASR_CPU_THREADS", "").strip()
    if override:
        try:
            return max(0, min(32, int(override)))
        except ValueError:
            LOGGER.warning("Ignoring invalid VSR_ASR_CPU_THREADS=%r", override)
    if shutil.which("nvidia-smi"):
        return 0
    return max(1, min(8, (os.cpu_count() or 1) // 2))


def _faster_whisper_num_workers(*, duration_ms: int | None = None) -> int:
    """Choose bounded CTranslate2 workers for automatically resolved Whisper.

    A second worker is useful on short CUDA media with enough VRAM, but loading
    two large-v3 workers is needlessly risky on small GPUs and was slower on a
    representative ten-minute run. The automatic policy therefore opts into
    two workers only for media up to three minutes when the visible GPU reports
    at least 8 GiB; an explicit override remains available for benchmarking.
    """

    override = os.environ.get("VSR_FASTER_WHISPER_NUM_WORKERS", "").strip()
    if override:
        try:
            return max(1, min(8, int(override)))
        except ValueError:
            LOGGER.warning(
                "Ignoring invalid VSR_FASTER_WHISPER_NUM_WORKERS=%r; using host policy",
                override,
            )
    if duration_ms is None or duration_ms > 180_000:
        return 1
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return 1
    try:
        probe = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
        values = [
            int(line.strip())
            for line in str(probe.stdout).splitlines()
            if line.strip().isdigit()
        ]
    except (OSError, subprocess.SubprocessError, ValueError):
        return 1
    # The first visible device is the one CUDA normally selects for an
    # automatic adapter; stay conservative on heterogeneous hosts.
    return 2 if values and values[0] >= 8192 else 1


def _asr_shared_cache_dir() -> Path | None:
    """Resolve the local content-addressed ASR cache used across projects.

    The cache stores only validated, model-keyed transcript chunks and never
    performs network I/O. It is enabled for automatically resolved production
    adapters so a fresh output directory can reuse exact local work. Explicit
    adapters remain project-local for testability and caller isolation. Set
    ``VSR_DISABLE_ASR_SHARED_CACHE=1`` to keep all transcript state inside the
    project, or provide ``VSR_ASR_SHARED_CACHE_DIR`` for a controlled location.
    """

    disabled = os.environ.get("VSR_DISABLE_ASR_SHARED_CACHE", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return None
    configured = os.environ.get("VSR_ASR_SHARED_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "video-script-reconstructor" / "asr-cache").resolve()


def _asr_shared_cache_limit() -> int:
    """Return the total shared ASR-cache budget in bytes."""

    default_limit = 512 * 1024 * 1024
    raw_limit = os.environ.get("VSR_ASR_SHARED_CACHE_MAX_BYTES", "").strip()
    if not raw_limit:
        return default_limit
    try:
        return max(0, int(raw_limit))
    except ValueError:
        LOGGER.warning("Ignoring invalid VSR_ASR_SHARED_CACHE_MAX_BYTES=%r", raw_limit)
        return default_limit


def _visual_shared_cache_dir() -> Path | None:
    """Resolve the bounded local cache for exact source-keyed frame PNGs."""

    disabled = os.environ.get("VSR_DISABLE_VISUAL_SHARED_CACHE", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return None
    configured = os.environ.get("VSR_VISUAL_SHARED_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "video-script-reconstructor" / "visual-cache").resolve()


def _visual_shared_cache_limit() -> int:
    """Return the total shared visual-cache budget in bytes."""

    default_limit = 512 * 1024 * 1024
    raw_limit = os.environ.get("VSR_VISUAL_SHARED_CACHE_MAX_BYTES", "").strip()
    if not raw_limit:
        return default_limit
    try:
        return max(0, int(raw_limit))
    except ValueError:
        LOGGER.warning("Ignoring invalid VSR_VISUAL_SHARED_CACHE_MAX_BYTES=%r", raw_limit)
        return default_limit


def _visual_shared_survey_path(cache_key_value: str) -> Path | None:
    """Return the bounded cross-project path for a context-free survey receipt."""

    if _visual_shared_cache_limit() <= 0:
        return None
    root = _visual_shared_cache_dir()
    if root is None:
        return None
    path = root / "surveys" / f"{cache_key_value}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.warning("Visual shared survey cache unavailable at %s: %s", path.parent, exc)
        return None
    return path


def _keep_completed_checkpoints() -> bool:
    """Keep visual caches after a valid run when explicitly requested.

    Completed projects already have canonical evidence and can resume through
    their run-cache key.  Removing duplicate raw-frame/OCR caches by default
    prevents repeated long-form tests from accumulating large hidden trees;
    ``VSR_KEEP_COMPLETED_CHECKPOINTS=1`` preserves them for cache experiments.
    """

    value = os.environ.get("VSR_KEEP_COMPLETED_CHECKPOINTS", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


_ASR_PROGRESS_TIMING_LIMIT = 32


def _bounded_asr_progress_event(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Bound persisted ASR telemetry without changing the callback payload.

    Chunk checkpoints and the final ASR result retain every timing record for
    reproducibility.  The run manifest and the resumable progress envelope are
    operational telemetry, however, so retaining an ever-growing history in
    every chunk write creates unnecessary O(n²) JSON work on very long media.
    Keep the newest bounded window and report how many older records were
    intentionally omitted.  The caller's mapping is never mutated.
    """

    event = dict(payload)
    timings = event.get("chunk_timings")
    if isinstance(timings, list) and len(timings) > _ASR_PROGRESS_TIMING_LIMIT:
        omitted = len(timings) - _ASR_PROGRESS_TIMING_LIMIT
        event["chunk_timings"] = list(timings[-_ASR_PROGRESS_TIMING_LIMIT:])
        event["chunk_timings_omitted"] = omitted
    return event


def _parallel_visual_survey_enabled(
    *, duration_ms: int | None = None, automatic_adapter: bool = True
) -> bool:
    """Return whether detector work may overlap local ASR scheduling.

    FFmpeg survey work is CPU-bound while a production local ASR adapter can
    use CUDA independently.  The default ``auto`` policy therefore overlaps
    only long media (at least five minutes) on hosts with an NVIDIA runtime and
    twelve or more logical CPUs. Explicit ``0``/``1`` settings remain the
    escape hatch for hardware-specific benchmarking. An explicitly injected
    adapter never opts into ``auto`` because its device/resource contract is
    caller-owned. This is scheduling only: detector thresholds, measured PTS,
    evidence pixels, and cache identities are unchanged.
    """

    policy = os.environ.get("VSR_PARALLEL_VISUAL_SURVEY", "auto").strip().lower()
    if policy in {"0", "false", "no", "off"}:
        return False
    if policy in {"1", "true", "yes", "on"}:
        return True
    if policy not in {"", "auto"} or not automatic_adapter:
        return False
    if duration_ms is None or duration_ms < 5 * 60 * 1000:
        return False
    if (os.cpu_count() or 1) < 12:
        return False
    return shutil.which("nvidia-smi") is not None


def _parallel_visual_warmup_enabled(
    *, duration_ms: int | None = None, automatic_adapter: bool = True
) -> bool:
    """Return whether exact survey-frame warmup may overlap local ASR.

    Unlike the transcript-independent survey itself, exact frame warmup is an
    owner-tuned experiment.  It is therefore explicitly off by default: a
    caller must opt in after measuring CPU, disk, and ASR contention on the
    target host.  The warmup never changes the candidate set; it only writes
    bounded raw-frame checkpoints that the ordinary visual stage may reuse.
    """

    policy = os.environ.get("VSR_PARALLEL_VISUAL_WARMUP", "off").strip().lower()
    if policy in {"0", "false", "no", "off", ""}:
        return False
    if policy in {"1", "true", "yes", "on"}:
        return bool(automatic_adapter and (duration_ms is None or duration_ms >= 5 * 60 * 1000))
    if policy == "auto":
        # Keep auto deliberately conservative until an owner records a cold
        # matrix showing that the extra decoder work does not slow ASR or
        # exceed memory/disk limits.  Explicit ``on`` is the experiment gate.
        return False
    LOGGER.warning("Ignoring invalid VSR_PARALLEL_VISUAL_WARMUP=%r", policy)
    return False


def _parallel_visual_warmup_batch_size() -> int:
    """Return the bounded exact-frame warmup batch size."""

    raw = os.environ.get("VSR_PARALLEL_VISUAL_WARMUP_BATCH_SIZE", "64").strip()
    try:
        return max(1, min(128, int(raw)))
    except ValueError:
        LOGGER.warning(
            "Ignoring invalid VSR_PARALLEL_VISUAL_WARMUP_BATCH_SIZE=%r", raw
        )
        return 64


def _parallel_visual_warmup_max_frames() -> int:
    """Return the hard cap for transcript-independent warmup requests."""

    raw = os.environ.get("VSR_PARALLEL_VISUAL_WARMUP_MAX_FRAMES", "1024").strip()
    try:
        return max(0, min(4096, int(raw)))
    except ValueError:
        LOGGER.warning(
            "Ignoring invalid VSR_PARALLEL_VISUAL_WARMUP_MAX_FRAMES=%r", raw
        )
        return 1024


def _survey_candidate_point(candidate: Any, *, duration_ms: int) -> int | None:
    """Resolve a survey candidate exactly as the canonical visual stage does."""

    requested = getattr(candidate, "requested_ms", None)
    actual = getattr(candidate, "actual_ms", None)
    point_value = actual if actual is not None else requested
    if point_value is None:
        return None
    try:
        point = int(point_value)
    except (TypeError, ValueError):
        return None
    return min(max(point, 0), max(duration_ms - 1, 0))


def _transcript_independent_warmup_times(
    candidates: Sequence[Any],
    shared_frames: Sequence[Any],
    *,
    duration_ms: int,
) -> tuple[int, ...]:
    """Return exact survey points not already covered by emitted survey PNGs.

    The final visual stage resolves each survey candidate to the same measured
    ``actual_ms``/requested timestamp.  Prefetching those points through the
    guarded exact extractor is therefore byte-safe.  Existing shared survey
    PNGs are intentionally excluded because the final stage still owns and
    materializes those temporary files.
    """

    already_emitted: set[int] = set()
    for frame in shared_frames:
        branch = getattr(frame, "branch", None)
        requested = getattr(frame, "requested_ms", None)
        if branch in {"hard", "periodic"} and isinstance(requested, int):
            already_emitted.add(int(requested))
    points = {
        point
        for candidate in candidates
        if (point := _survey_candidate_point(candidate, duration_ms=duration_ms)) is not None
        and point not in already_emitted
    }
    ordered = tuple(sorted(points))
    limit = _parallel_visual_warmup_max_frames()
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    # This is an acceleration-only cap.  Keep a deterministic temporal spread
    # and leave all omitted timestamps to the canonical guarded extractor.
    return _bounded_shared_survey_emission_times(ordered, max_count=limit)


def _prefetch_transcript_independent_visual_frames(
    source: Path,
    project_dir: Path,
    *,
    duration_ms: int,
    candidates: Sequence[Any],
    shared_frames: Sequence[Any],
    frame_output_dir: Path,
    source_sha256: str | None,
) -> dict[str, Any]:
    """Warm exact survey timestamps into bounded schedule checkpoints.

    Each batch has its own schedule-bound checkpoint and disposable output
    directory.  A failed batch is recorded and ignored; the normal visual
    stage still extracts any missing timestamp through its authoritative path.
    """

    times = _transcript_independent_warmup_times(
        candidates,
        shared_frames,
        duration_ms=duration_ms,
    )
    prefetched_frame_count = 0
    prefetched_batch_count = 0
    prefetch_failed_batch_count = 0
    first_error: str | None = None
    if not times or _visual_frame_cache_limit() <= 0:
        return {
            "requested_count": len(times),
            "prefetched_frame_count": prefetched_frame_count,
            "prefetched_batch_count": prefetched_batch_count,
            "prefetch_failed_batch_count": prefetch_failed_batch_count,
            "prefetch_elapsed_seconds": 0.0,
            "prefetch_error": first_error,
        }

    started = time.perf_counter()
    warmup_root = frame_output_dir / "warmup"
    batch_size = _parallel_visual_warmup_batch_size()
    for batch_index, start in enumerate(range(0, len(times), batch_size)):
        batch_times = times[start : start + batch_size]
        output_dir = warmup_root / f"batch-{batch_index:06d}"
        try:
            extracted = _load_or_extract_visual_frames(
                source,
                project_dir,
                output_dir,
                batch_times,
                duration_ms=duration_ms,
                max_workers=_visual_frame_workers(),
                source_sha256=source_sha256,
                shared_frames=(),
                shared_cache_dir=None,
                worker_pool=None,
            )
        except Exception as exc:  # pragma: no cover - FFmpeg/filesystem specific
            prefetch_failed_batch_count += 1
            if first_error is None:
                first_error = str(exc)
            LOGGER.warning(
                "Transcript-independent visual warmup batch %d failed; canonical extraction will recover it",
                batch_index,
                exc_info=True,
            )
        else:
            prefetched_frame_count += len(extracted)
            prefetched_batch_count += 1
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
    return {
        "requested_count": len(times),
        "prefetched_frame_count": prefetched_frame_count,
        "prefetched_batch_count": prefetched_batch_count,
        "prefetch_failed_batch_count": prefetch_failed_batch_count,
        "prefetch_elapsed_seconds": round(time.perf_counter() - started, 6),
        "prefetch_error": first_error,
    }


def _scheduler_snapshot(*, duration_ms: int | None = None) -> dict[str, Any]:
    """Record the bounded scheduling decisions used by this run.

    These values are operational telemetry only. Keeping them beside stage
    timings makes CPU/GPU and OCR benchmark comparisons reproducible without
    putting scheduler settings into the evidence/cache identity; changing a
    worker count cannot change the selected transcript or pixels.
    """

    return {
        "frame_extract_workers": _visual_frame_workers(),
        "frame_analysis_workers": _visual_analysis_workers(),
        "crop_prepare_workers": _visual_crop_workers(),
        "survey_ffmpeg_threads": _visual_survey_ffmpeg_threads(),
        "ocr_workers": _ocr_workers(),
        "ocr_checkpoint_batch": _ocr_checkpoint_flush_interval(),
        "ocr_batch_size": _ocr_batch_size(),
        "paddle_ocr_workers": _paddle_ocr_batch_workers(),
        "asr_cpu_threads": _asr_cpu_threads(),
        "asr_num_workers": _faster_whisper_num_workers(duration_ms=duration_ms),
        "asr_shared_cache": _asr_shared_cache_dir() is not None,
        "visual_shared_cache": _visual_shared_cache_dir() is not None,
        "parallel_visual_survey": _parallel_visual_survey_enabled(duration_ms=duration_ms),
        "parallel_visual_warmup": _parallel_visual_warmup_enabled(duration_ms=duration_ms),
        # Read-only policy/probe summary. This never invokes the guarded
        # parity probe; it only reports already-decided outcomes.
        "survey_hwaccel": _survey_hwaccel_telemetry(),
        "guarded_motion_dedup": _guarded_motion_only_enabled(),
    }


def _precompute_visual_survey(
    source: Path,
    project_dir: Path,
    *,
    duration_ms: int,
    interval_seconds: float,
    strict: bool,
    scene_detection: bool,
    adaptive_detection: bool,
    source_sha256: str | None,
    prefetch_exact_frames: bool = False,
) -> _PrecomputedVisualSurvey:
    """Run the source survey independently of transcript/ASR decoding.

    The worker deliberately supplies no speech-reference points.  The visual
    stage adds those contextual candidates after ASR completes, then commits a
    final context-bound survey marker.  Structural hard-cut/adaptive candidates
    and exact-safe periodic frames are therefore reusable without guessing any
    transcript alignment.
    """

    from .scene_detection import periodic_candidate_times

    checkpoint_root = project_dir / ".state" / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    frame_output_dir = Path(
        tempfile.mkdtemp(prefix=".vsr-parallel-survey-frames-", dir=checkpoint_root)
    )
    periodic_times = periodic_candidate_times(
        duration_ms,
        interval_seconds=interval_seconds,
        strict=strict,
    )
    # Resolve the opt-in accelerator exactly once per source so every cache
    # identity and decode pass in this worker shares one decided mode.
    decode_mode = _survey_hwaccel_effective_mode(source, source_sha256=source_sha256)
    try:
        candidates, shared_frames = _load_or_run_visual_survey_with_frames(
            source,
            project_dir,
            duration_ms=duration_ms,
            interval_seconds=interval_seconds,
            strict=strict,
            scene_detection=scene_detection,
            adaptive_detection=adaptive_detection,
            speech_reference_times_ms=(),
            periodic_times_ms=periodic_times,
            frame_output_dir=frame_output_dir,
            source_sha256=source_sha256,
            decode_mode=decode_mode,
            hwaccel=decode_mode,
        )
    except Exception:
        shutil.rmtree(frame_output_dir, ignore_errors=True)
        raise
    prefetch_metrics: dict[str, Any] = {}
    if prefetch_exact_frames:
        prefetch_metrics = _prefetch_transcript_independent_visual_frames(
            source,
            project_dir,
            duration_ms=duration_ms,
            candidates=candidates,
            shared_frames=shared_frames,
            frame_output_dir=frame_output_dir,
            source_sha256=source_sha256,
        )
    return _PrecomputedVisualSurvey(
        candidates=tuple(candidates),
        shared_frames=tuple(shared_frames),
        shared_frame_dir=frame_output_dir,
        prefetched_frame_count=int(prefetch_metrics.get("prefetched_frame_count", 0)),
        prefetched_batch_count=int(prefetch_metrics.get("prefetched_batch_count", 0)),
        prefetch_failed_batch_count=int(
            prefetch_metrics.get("prefetch_failed_batch_count", 0)
        ),
        prefetch_elapsed_seconds=float(prefetch_metrics.get("prefetch_elapsed_seconds", 0.0)),
    )


def _extract_visual_evidence_legacy(
    source: Path,
    project_dir: Path,
    media_id: str,
    duration_ms: int,
    blocks: list[dict[str, Any]],
    *,
    survey_interval_seconds: float = 30.0,
    strict: bool = True,
    ocr_adapter: OCRAdapter | None = None,
    scene_detection_enabled: bool = True,
    frame_difference_enabled: bool = True,
    source_sha256: str | None = None,
    precomputed_survey: _PrecomputedVisualSurvey | None = None,
    worker_pool: ThreadPoolExecutor | None = None,
    frame_callback: Callable[[Any], None] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    from .image_metadata import embed_metadata_with_file_hash

    requested: list[tuple[int, str]] = _bounded_visual_block_points(
        blocks,
        duration_ms=duration_ms,
        spacing_ms=max(1_000, int((survey_interval_seconds if strict else 30.0) * 1000)),
    )
    from .scene_detection import periodic_candidate_times

    bounded_interval = (
        min(survey_interval_seconds, 30.0) if strict else survey_interval_seconds
    )
    initial_context_times = tuple(point for point, _ in requested)
    shared_checkpoint_root = project_dir / ".state" / "checkpoints"
    shared_checkpoint_root.mkdir(parents=True, exist_ok=True)
    # Resolve the opt-in accelerator exactly once so the survey loader and the
    # final context-bound cache marker share one decided decode mode.
    decode_mode = _survey_hwaccel_effective_mode(source, source_sha256=source_sha256)
    if precomputed_survey is None:
        shared_frame_dir = Path(
            tempfile.mkdtemp(prefix=".vsr-survey-frames-", dir=shared_checkpoint_root)
        )
        periodic_times = tuple(
            sorted(
                set(
                    periodic_candidate_times(
                        duration_ms,
                        interval_seconds=bounded_interval,
                        strict=strict,
                    )
                ).union(initial_context_times)
            )
        )
        survey_candidates, shared_frames = _load_or_run_visual_survey_with_frames(
            source,
            project_dir,
            duration_ms=duration_ms,
            interval_seconds=bounded_interval,
            strict=strict,
            scene_detection=scene_detection_enabled,
            adaptive_detection=frame_difference_enabled,
            # Use the same bounded transcript scaffold for contextual survey
            # points; passing every diarizer boundary would recreate the dense
            # per-segment extraction this stage is designed to avoid. These
            # points are part of the cache key so a different scaffold cannot
            # reuse a contextually incomplete survey.
            speech_reference_times_ms=initial_context_times,
            periodic_times_ms=periodic_times,
            frame_output_dir=shared_frame_dir,
            source_sha256=source_sha256,
            decode_mode=decode_mode,
            hwaccel=decode_mode,
        )
    else:
        from .scene_detection import contextual_candidates, merge_survey_candidates

        shared_frame_dir = precomputed_survey.shared_frame_dir
        survey_candidates = merge_survey_candidates(
            (
                precomputed_survey.candidates,
                contextual_candidates(speech_reference_times_ms=initial_context_times),
            )
        )
        shared_frames = precomputed_survey.shared_frames
        # The worker's cache key intentionally had no transcript context.  Write
        # the final context-bound marker now so later rebuilds remain exact and
        # deterministic even when the initial run used parallel scheduling.
        cache_path, cache_key_value, cache_source_digest, cache_ffmpeg_version = (
            _visual_survey_cache_identity(
                source,
                project_dir,
                duration_ms=duration_ms,
                interval_seconds=bounded_interval,
                strict=strict,
                scene_detection=scene_detection_enabled,
                adaptive_detection=frame_difference_enabled,
                speech_reference_times_ms=initial_context_times,
                source_sha256=source_sha256,
                # Match the worker's decided mode so this context-bound marker
                # can never collide with the software cache identity.
                decode_mode=decode_mode,
            )
        )
        _write_visual_survey_cache(
            cache_path,
            key=cache_key_value,
            source_digest=cache_source_digest,
            ffmpeg_version=cache_ffmpeg_version,
            candidates=survey_candidates,
        )
    survey_reasons_by_time: dict[int, tuple[str, ...]] = {}
    for candidate in survey_candidates:
        reasons = tuple(str(reason) for reason in candidate.reasons)
        for timestamp in (candidate.requested_ms, candidate.actual_ms):
            if timestamp is not None:
                existing = survey_reasons_by_time.get(int(timestamp), ())
                survey_reasons_by_time[int(timestamp)] = tuple(
                    dict.fromkeys((*existing, *reasons))
                )
        point = candidate.actual_ms if candidate.actual_ms is not None else candidate.requested_ms
        point = min(max(int(point), 0), max(duration_ms - 1, 0))
        has_speech_overlap = any(
            block.get("start_ms") is not None
            and block.get("end_ms") is not None
            and int(block["start_ms"]) <= point < int(block["end_ms"])
            for block in blocks
        )
        meaningful_change = bool(
            set(candidate.reasons)
            & {
                "scene_cut",
                "adaptive_frame_difference",
                "perceptual_change",
                "motion_change",
                "ocr_change",
                "post_motion_stable",
            }
        )
        if blocks and meaningful_change and not has_speech_overlap:
            next_block_number = (
                max(
                    (
                        int(str(block.get("block_id", "B0")).removeprefix("B"))
                        for block in blocks
                        if str(block.get("block_id", "")).startswith("B")
                        and str(block.get("block_id", "")).removeprefix("B").isdigit()
                    ),
                    default=0,
                )
                + 1
            )
            visual_block_id = f"B{next_block_number:06d}"
            blocks.append(
                {
                    "block_id": visual_block_id,
                    "chapter_id": "C001",
                    "start_ms": point,
                    "end_ms": point,
                    "speaker": "Visual event",
                    "spoken_text": "",
                    "visual_description": "[visual evidence retained; semantic description pending review]",
                    "on_screen_text": [],
                    "relevant_non_speech_audio": [],
                    "frame_ids": [],
                    "transcript_segment_ids": [],
                    "visual_event_ids": [],
                    "image_claim_ids": [],
                    "metadata_revision_ids": [],
                    "metadata_sufficiency_decision_ids": [],
                    "transformation_ids": [],
                    "fidelity_mode": "verbatim",
                    "confidence": 0.0,
                    "verification_status": "unverified",
                    "uncertainty": [
                        "Visual-only event is outside a spoken transcript interval; semantic analysis is pending."
                    ],
                    "residual_source_text": None,
                }
            )
            nearest = visual_block_id
        else:
            nearest = (
                min(
                    blocks,
                    key=lambda item: abs(int(item.get("start_ms") or 0) - point),
                )["block_id"]
                if blocks
                else "B000001"
            )
        requested.append((point, str(nearest)))
    if not blocks:
        blocks.append(
            {
                "block_id": "B000001",
                "chapter_id": "C001",
                "start_ms": 0,
                "end_ms": duration_ms,
                "speaker": "Visual event",
                "spoken_text": "",
                "visual_description": None,
                "on_screen_text": [],
                "relevant_non_speech_audio": [],
                "frame_ids": [],
                "transcript_segment_ids": [],
                "visual_event_ids": [],
                "image_claim_ids": [],
                "metadata_revision_ids": [],
                "metadata_sufficiency_decision_ids": [],
                "transformation_ids": [],
                "fidelity_mode": "verbatim",
                "confidence": 0.0,
                "verification_status": "unverified",
                "uncertainty": [],
                "residual_source_text": None,
            }
        )
        requested.append((0, "B000001"))
    dedup: dict[int, str] = {}
    for point, block_id in requested:
        dedup.setdefault(point, block_id)

    ordered_requests = sorted(dedup.items())
    try:
        extracted_frames = _load_or_extract_visual_frames(
            source,
            project_dir,
            project_dir / "evidence" / "full",
            [point for point, _ in ordered_requests],
            duration_ms=duration_ms,
            max_workers=_visual_frame_workers(),
            source_sha256=source_sha256,
            shared_frames=shared_frames,
            shared_cache_dir=_visual_shared_cache_dir(),
            worker_pool=worker_pool,
            on_frame=frame_callback,
        )
    finally:
        shutil.rmtree(shared_frame_dir, ignore_errors=True)

    frames: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    revisions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    blocks_by_id = {str(item["block_id"]): item for item in blocks}
    # Creation metadata is independent for every decoded frame. Prepare and
    # write those PNG envelopes in a bounded pool, then commit the canonical
    # lists below in timestamp/ID order so concurrency cannot affect IDs or
    # chronology.
    creation_specs: list[dict[str, Any]] = []
    for index, ((_point, block_id), extracted) in enumerate(
        zip(ordered_requests, extracted_frames, strict=True), 1
    ):
        frame_id = extracted.frame_id
        relative = extracted.path.relative_to(project_dir).as_posix()
        revision_id = f"MR{index:06d}"
        revision_seed = {
            "revision_id": revision_id,
            "image_id": frame_id,
            "revision_number": 1,
            "kind": "creation",
        }
        revision_digest = hashlib.sha256(
            json.dumps(revision_seed, sort_keys=True).encode()
        ).hexdigest()
        block = blocks_by_id[block_id]
        creation_specs.append(
            {
                "frame_id": frame_id,
                "extracted": extracted,
                "media_id": media_id,
                "relative_path": relative,
                "block_id": block_id,
                "segment_ids": list(block.get("transcript_segment_ids", [])),
                "revision_id": revision_id,
                "revision_digest": revision_digest,
            }
        )
    # Pixel normalization is independent per frame but can be expensive for
    # high-resolution PNGs. Hash in the same bounded pool used by the visual
    # stage, then retain the original two-phase metadata commit below so a
    # failed read cannot leave a partially enriched batch.
    with _executor_context(
        worker_pool,
        max_workers=_visual_frame_workers(),
        thread_name_prefix="vsr-creation-pixel-hash",
    ) as pool:
        hash_futures = {
            str(spec["frame_id"]): pool.submit(
                _creation_payload,
                spec["extracted"],
                media_id=spec["media_id"],
                relative_path=spec["relative_path"],
                block_id=spec["block_id"],
                segment_ids=spec["segment_ids"],
                revision_id=spec["revision_id"],
                revision_digest=spec["revision_digest"],
            )
            for spec in creation_specs
        }
        creation_payloads: dict[str, Any] = {
            frame_id: future.result() for frame_id, future in hash_futures.items()
        }
    creation_paths = {
        str(spec["frame_id"]): spec["extracted"].path for spec in creation_specs
    }
    with _executor_context(
        worker_pool,
        max_workers=_visual_frame_workers(),
        thread_name_prefix="vsr-creation-metadata",
    ) as pool:
        metadata_futures = {
            frame_id: pool.submit(
                embed_metadata_with_file_hash,
                creation_paths[frame_id],
                payload,
                # ``_creation_payload`` has already decoded and normalized the
                # source pixels for this payload.  The internal fast path
                # retains the IDAT/dimension/color/read-back invariants while
                # avoiding two more full PNG decodes per creation frame.
                verify_source_pixels=False,
                verify_decoded_pixels=False,
            )
            for frame_id, payload in creation_payloads.items()
        }
        prepared_creation: dict[str, tuple[Any, str]] = {
            frame_id: future.result() for frame_id, future in metadata_futures.items()
        }

    for index, ((point, block_id), extracted) in enumerate(
        zip(ordered_requests, extracted_frames, strict=True), 1
    ):
        frame_id = extracted.frame_id
        relative = extracted.path.relative_to(project_dir).as_posix()
        revision_id = f"MR{index:06d}"
        block = blocks_by_id[block_id]
        prepared, file_hash = prepared_creation[frame_id]
        payload_dict = prepared.model_dump(mode="json")
        survey_reason_text = "; ".join(
            dict.fromkeys(
                (
                    *survey_reasons_by_time.get(int(point), ()),
                    "whole-duration survey",
                    "measured decoder frame",
                )
            )
        )
        payloads.append(payload_dict)
        revision = {
            "revision_id": revision_id,
            "revision_number": 1,
            "image_id": frame_id,
            "base_revision_id": None,
            "observation_ids": [],
            "added_claim_ids": [],
            "confirmed_claim_ids": [],
            "narrowed_claim_ids": [],
            "disputed_claim_ids": [],
            "rejected_claim_ids": [],
            "superseded_claim_ids": [],
            "unresolved_claim_ids": [],
            "previous_payload_digest": None,
            "new_payload_digest": prepared.integrity.payload_digest,
            "reconciliation_method": "creation",
            "actor": "deterministic-pipeline",
            "stale_base_reconciled": False,
            "pixel_invariance_verified": True,
            "embedded_write_verified": True,
            "read_back_verified": True,
            "canonical_mirror_committed": True,
            "created_at_utc": _now(),
        }
        revisions.append(revision)
        frame = {
            "frame_id": frame_id,
            "requested_ms": extracted.requested_ms,
            "actual_ms": extracted.actual_ms,
            "pts": extracted.raw_pts,
            "time_base": extracted.time_base,
            "frame_index": extracted.frame_index,
            "offset_ms": extracted.offset_ms,
            "timestamp_source": extracted.timestamp_source,
            "timing_estimated": False,
            "full_frame_path": relative,
            "crops": [],
            "scene_id": None,
            "quality_scores": {},
            "perceptual_hashes": {},
            "region_hashes": {},
            "pixel_hash": payload_dict["image"]["pixel_hash"],
            "file_hash": file_hash,
            "metadata_payload_digest": prepared.integrity.payload_digest,
            "latest_revision_id": revision_id,
            "metadata_sufficiency_state": "semantic_observer_unavailable",
            "ocr_observation_ids": [],
            "selection_reason": survey_reason_text,
            "evidence_role": "context",
            "linked_event_ids": [f"V{index:06d}"],
            "linked_block_ids": [block_id],
            "verification_status": "unverified",
            "supported_claim_ids": [],
            "disputed_claim_ids": [],
            "unresolved_claim_ids": [],
            "description": "Visual evidence retained; semantic description pending review.",
            "path": relative,
            "final": True,
            "metadata": payload_dict,
        }
        frames.append(frame)
        block["frame_ids"].append(frame_id)
        block["metadata_revision_ids"].append(revision_id)
        block["visual_description"] = (
            "[visual evidence retained; semantic description pending review]"
        )
        block["uncertainty"].append("Semantic visual analysis is pending for the cited frame.")
        event_id = f"V{index:06d}"
        block["visual_event_ids"].append(event_id)
        events.append(
            {
                "event_id": event_id,
                "start_ms": extracted.actual_ms,
                "end_ms": extracted.actual_ms,
                "event_type": "semantic_pending",
                "scene_or_state_id": None,
                "evidence_frame_ids": [frame_id],
                "before_action_after_roles": {},
                "ocr_observation_ids": [],
                "factual_grounded_description": "[visual evidence retained; semantic description pending review]",
                "importance": "supporting",
                "confidence": 0.0,
                "uncertainty": ["Semantic visible content has not been analyzed."],
                "annotation_provider": None,
                "review_status": "review_required",
                "image_claim_ids": [],
                "metadata_revision_ids": [revision_id],
            }
        )
        review_id = _next_review_id(reviews)
        reviews.append(
            {
                "review_id": review_id,
                "severity": "medium",
                "category": "visual_semantic_annotation",
                "start_ms": extracted.actual_ms,
                "end_ms": extracted.actual_ms,
                "block_ids": [block_id],
                "segment_ids": list(block.get("transcript_segment_ids", [])),
                "event_ids": [event_id],
                "frame_ids": [frame_id],
                "ocr_observation_ids": [],
                "image_claim_ids": [],
                "metadata_revision_ids": [revision_id],
                "sufficiency_decision_ids": [],
                "problem": "A source frame exists, but no reliable semantic observer has described the meaningful visible state.",
                "alternatives": [],
                "required_action": "Read embedded metadata, inspect the full frame and adjacent evidence, then ingest a targeted host-agent observation.",
                "blocking": False,
                "decision": None,
                "reviewer": None,
                "decision_timestamp_utc": None,
                "rationale": None,
            }
        )
        packet = {
            "schema_version": "1.0",
            "packet_id": f"P{index:06d}",
            "event_id": event_id,
            "frame_ids": [frame_id],
            "frame_paths": [relative],
            "requested_ms": point,
            "actual_ms": extracted.actual_ms,
            "nearby_segment_ids": list(block.get("transcript_segment_ids", [])),
            "questions": [
                "What meaningful visible state or exact text is needed to understand this block?"
            ],
            "instructions_are_untrusted_evidence": True,
        }
        atomic_write_json(
            project_dir / ".state" / "vision" / "packets" / f"{event_id}.json", packet
        )
    atomic_write_json(
        project_dir / ".state" / "vision" / "image-observations.json",
        {
            "schema_version": "1.0",
            "payloads": payloads,
            "observations": [],
            "claims": [],
            "revisions": revisions,
        },
        compact=True,
    )
    return frames, payloads, revisions, events, reviews


def _extract_visual_evidence(
    source: Path,
    project_dir: Path,
    media_id: str,
    duration_ms: int,
    blocks: list[dict[str, Any]],
    *,
    survey_interval_seconds: float = 30.0,
    strict: bool = True,
    ocr_adapter: OCRAdapter | None = None,
    ocr_enabled: bool = True,
    scene_detection_enabled: bool = True,
    frame_difference_enabled: bool = True,
    protect_small_changes: bool = True,
    deduplicate: bool = True,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    source_sha256: str | None = None,
    ocr_cache_key: str = "auto-local-ocr",
    precomputed_survey: _PrecomputedVisualSurvey | None = None,
    worker_pool: ThreadPoolExecutor | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Run content-aware final selection and deterministic metadata enrichment.

    The legacy extractor creates measured, metadata-bearing candidates first. This
    pass then scores their real pixels, protects localized/OCR changes, moves
    rejected candidates under hidden state, and appends a deterministic metadata
    revision to every retained image. Semantic interpretation remains pending.
    """

    from .frame_quality import (
        PERCEPTUAL_DHASH_ALGORITHM,
        PERCEPTUAL_DHASH_VERIFIED,
        analyze_frame_sequence_with_hash,
        assess_frame_quality,
        perceptual_dhash,
    )
    from .frame_quality import normalized_pixel_hash as normalized_pixel_hash_image
    from .frame_selection import FrameCandidate, select_frames
    from .ids import evidence_filename, sequential_id
    from .image_metadata import (
        embed_metadata_with_file_hash,
        prepare_metadata_payload,
        read_embedded_metadata,
    )
    from .ocr import to_schema_observation
    from .schemas import EvidenceImageMetadata
    from .vision_packets import VisionPacket, create_vision_packet

    visual_started = time.perf_counter()

    def emit_progress(event: str, **metrics: Any) -> None:
        if progress_callback is None:
            return
        payload: dict[str, Any] = {
            "stage": "visual_evidence",
            "event": event,
            "updated_at_utc": _now(),
            "elapsed_seconds": round(time.perf_counter() - visual_started, 6),
            **metrics,
        }
        try:
            progress_callback(payload)
        except Exception:  # pragma: no cover - optional telemetry boundary
            LOGGER.warning("Visual progress callback failed", exc_info=True)

    adapter = ocr_adapter if ocr_enabled else None
    prefetch: _StreamingOCRPrefetch | None = None
    prefetched_by_frame_id: dict[str, Any] = {}
    prefetch_metrics: dict[str, Any] = {}
    if adapter is not None and callable(getattr(adapter, "recognize_many", None)):
        try:
            prefetch = _StreamingOCRPrefetch(
                adapter,
                batch_size=_ocr_batch_size(),
                worker_count=_paddle_ocr_batch_workers(adapter),
            )
        except InputError:
            LOGGER.info("Streaming OCR prefetch is unavailable for this adapter")

    decode_started = time.perf_counter()
    try:
        frames, payloads, revisions, events, reviews = _extract_visual_evidence_legacy(
            source,
            project_dir,
            media_id,
            duration_ms,
            blocks,
            survey_interval_seconds=survey_interval_seconds,
            strict=strict,
            ocr_adapter=ocr_adapter,
            scene_detection_enabled=scene_detection_enabled,
            frame_difference_enabled=frame_difference_enabled,
            source_sha256=source_sha256,
            precomputed_survey=precomputed_survey,
            worker_pool=worker_pool,
            frame_callback=prefetch.submit if prefetch is not None else None,
        )
    finally:
        if prefetch is not None:
            prefetch.finish()
            prefetched_by_frame_id = prefetch.observations
            prefetch_metrics = prefetch.metrics
    emit_progress(
        "frame_decode_completed",
        full_frame_count=len(frames),
        elapsed_seconds=round(time.perf_counter() - decode_started, 6),
        streaming_ocr_prefetch=prefetch_metrics,
    )
    if not frames:
        return frames, payloads, revisions, events, reviews, []

    ordered = sorted(frames, key=lambda item: (int(item["actual_ms"]), str(item["frame_id"])))
    quality_by_id: dict[str, Any] = {}
    difference_by_id: dict[str, Any] = {}
    raw_ocr_by_frame: dict[str, Any] = {}
    ocr_records: list[dict[str, Any]] = []
    candidates: list[Any] = []
    guarded_motion_downgrade_count = 0
    # OCR and deterministic frame analysis are independent reads of the same
    # already-materialized evidence PNGs. When an OCR adapter is actually
    # configured, overlap their bounded worker pools so a slow OCR backend does
    # not serialize the image-quality pass. The no-adapter path stays direct to
    # avoid creating orchestration threads for the common lightweight case.
    try:
        if adapter is None:
            ocr_by_frame, ocr_metrics = _load_or_run_ocr(
                source,
                project_dir,
                ordered,
                adapter=None,
                adapter_key=ocr_cache_key,
                source_sha256=source_sha256,
                shared_cache_dir=_visual_shared_cache_dir(),
                prefetched_by_frame_id=prefetched_by_frame_id,
                prefetch_metrics=prefetch_metrics,
            )
            sequence_results = analyze_frame_sequence_with_hash(
                [project_dir / str(frame["full_frame_path"]) for frame in ordered],
                max_workers=_visual_analysis_workers(),
            )
        else:
            with ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="vsr-visual-preanalysis"
            ) as phase_pool:
                ocr_future = phase_pool.submit(
                    _load_or_run_ocr,
                    source,
                    project_dir,
                    ordered,
                    adapter=adapter,
                    adapter_key=ocr_cache_key,
                    source_sha256=source_sha256,
                    shared_cache_dir=_visual_shared_cache_dir(),
                    prefetched_by_frame_id=prefetched_by_frame_id,
                    prefetch_metrics=prefetch_metrics,
                )
                analysis_future = phase_pool.submit(
                    analyze_frame_sequence_with_hash,
                    [project_dir / str(frame["full_frame_path"]) for frame in ordered],
                    max_workers=_visual_analysis_workers(),
                )
                ocr_by_frame, ocr_metrics = ocr_future.result()
                sequence_results = analysis_future.result()
    finally:
        close_adapter = getattr(adapter, "close", None)
        if callable(close_adapter):
            close_adapter()
    emit_progress("ocr_completed", **ocr_metrics)

    # Keep the canonical analysis ordering deterministic even though the two
    # independent pre-analysis phases may finish in either order.
    analysis_by_id = {
        str(frame["frame_id"]): result
        for frame, result in zip(ordered, sequence_results, strict=True)
    }
    # Keep the precomputed dHash attached to its own frame.  The enrichment
    # pass below commits frames in a separate loop; reusing the loop-local
    # ``dhash`` variable there would silently stamp every frame with the last
    # frame's hash, disabling perceptual deduplication and corrupting the
    # canonical similarity metadata.
    dhash_by_id = {
        frame_id: result[2] for frame_id, result in analysis_by_id.items()
    }

    for index, frame in enumerate(ordered):
        frame_id = str(frame["frame_id"])
        path = project_dir / str(frame["full_frame_path"])
        quality, difference, dhash = analysis_by_id[frame_id]
        quality_by_id[frame_id] = quality
        if frame_id in ocr_by_frame:
            ocr_observation = ocr_by_frame[frame_id]
        else:
            ocr_observation = None
        if ocr_observation is not None:
            raw_ocr_by_frame[frame_id] = ocr_observation
            ocr_records.append(to_schema_observation(ocr_observation).model_dump(mode="json"))
        if difference is not None:
            difference_by_id[frame_id] = difference
        ocr_text = ocr_observation.normalized_interpretation if ocr_observation else None
        pixel_changed = bool(
            difference
            and (difference.maximum_region_change >= 0.05 or difference.changed_pixel_ratio >= 0.01)
        )
        ocr_changed = False
        if index and ocr_text:
            prior = raw_ocr_by_frame.get(str(ordered[index - 1]["frame_id"]))
            ocr_changed = bool(prior and prior.normalized_interpretation != ocr_text)
        changed = pixel_changed or ocr_changed
        next_dhash = None
        if index + 1 < len(ordered):
            next_dhash = dhash_by_id.get(str(ordered[index + 1]["frame_id"]))
        frame_metadata = frame.get("metadata", {})
        image_metadata = (
            frame_metadata.get("image", {}) if isinstance(frame_metadata, Mapping) else {}
        )
        if not isinstance(image_metadata, Mapping):
            image_metadata = {}
        frame_width = int(image_metadata.get("width") or frame.get("width") or 0)
        frame_height = int(image_metadata.get("height") or frame.get("height") or 0)
        selection_reason = str(frame.get("selection_reason", ""))
        motion_only = _is_guarded_motion_only(
            difference,
            current_dhash=dhash,
            next_dhash=next_dhash,
            ocr_stable=bool(
                ocr_text
                and index
                and raw_ocr_by_frame.get(str(ordered[index - 1]["frame_id"])) is not None
                and not ocr_changed
            ),
            selection_reason=selection_reason,
            width=frame_width,
            height=frame_height,
            is_boundary=index in {0, len(ordered) - 1},
        )
        if motion_only:
            guarded_motion_downgrade_count += 1
            changed = False
        novelty = (
            min(
                1.0,
                difference.perceptual_hamming / 16.0
                + difference.changed_pixel_ratio
                + difference.maximum_region_change / 2.0,
            )
            if difference
            else 1.0
        )
        candidates.append(
            FrameCandidate(
                frame_id=frame_id,
                path=path,
                actual_ms=int(frame["actual_ms"]),
                requested_ms=int(frame["requested_ms"]),
                relevance=0.85,
                importance=0.9 if changed else 0.55,
                temporal_proximity=0.9,
                stability=max(0.0, 1.0 - quality.transition_risk),
                novelty=novelty,
                ocr_readability=ocr_observation.confidence or 0.0
                if ocr_observation is not None
                else 0.0,
                full_state_completeness=1.0,
                transition_risk=quality.transition_risk,
                evidence_role="after" if changed else "context",
                ocr_text=ocr_text,
                consequential_change=changed,
                mandatory=index in {0, len(ordered) - 1} or changed,
                quality=quality,
                pixel_hash=(
                    str(frame["pixel_hash"].get("value"))
                    if isinstance(frame.get("pixel_hash"), Mapping)
                    and frame["pixel_hash"].get("value")
                    else None
                ),
                # Sequence analysis has already decoded this PNG and returned
                # its dHash.  Reuse that value during selection instead of
                # reopening every candidate for a second perceptual pass.
                perceptual_hash=dhash,
                reasons=tuple(
                    dict.fromkeys(
                        (
                            *tuple(
                                reason.strip()
                                for reason in selection_reason.split(";")
                                if reason.strip()
                            ),
                            *( ("guarded_motion_only",) if motion_only else () ),
                            "whole-duration survey",
                            "measured decoder frame",
                        )
                    )
                ),
            )
        )

    emit_progress(
        "analysis_completed",
        full_frame_count=len(ordered),
        ocr_observation_count=len(ocr_records),
        guarded_motion_downgrade_count=guarded_motion_downgrade_count,
    )
    selection = select_frames(
        candidates,
        duration_ms=duration_ms,
        protect_small_changes=protect_small_changes,
        deduplicate=deduplicate,
    )
    selected_ids = {item.candidate.frame_id for item in selection.selected}
    candidate_frames: list[dict[str, Any]] = []
    # Candidate processing only replaces one payload at a time and appends
    # immutable revision records.  Copy the list containers, not every nested
    # envelope, so long-form selection avoids a second full JSON tree walk.
    candidate_payloads = list(payloads)
    candidate_revisions = list(revisions)
    candidate_revision_counter = max(
        (
            int(str(item.get("revision_id", "MR0")).removeprefix("MR"))
            for item in candidate_revisions
            if str(item.get("revision_id", "")).startswith("MR")
            and str(item.get("revision_id", "")).removeprefix("MR").isdigit()
        ),
        default=0,
    )
    hidden_rejected = project_dir / ".state" / "candidates" / "rejected-frames"
    hidden_rejected.mkdir(parents=True, exist_ok=True)
    payload_by_image_id = {
        str(payload.get("image", {}).get("image_id")): payload for payload in payloads
    }
    for frame in frames:
        frame_id = str(frame["frame_id"])
        if frame_id in selected_ids:
            continue
        path = project_dir / str(frame["full_frame_path"])
        if path.is_file():
            destination = hidden_rejected / path.name
            if destination.exists():
                destination.unlink()
            shutil.move(str(path), str(destination))
            candidate_frame = json.loads(json.dumps(frame))
            candidate_relative = destination.relative_to(project_dir).as_posix()
            candidate_frame["full_frame_path"] = candidate_relative
            candidate_frame["path"] = candidate_relative
            candidate_frame["final"] = False
            # A rejected survey frame is still a generated evidence artifact,
            # but its portable origin must say candidate rather than implying
            # that it was selected as final evidence.  Record this as an
            # append-only metadata revision and mirror the current payload in
            # the hidden canonical ledger.
            # The creation pool already validated and read back every current
            # envelope. Reuse that canonical mirror instead of reopening the
            # rejected PNG solely to recover the same metadata; retain the
            # guarded embedded-read fallback for legacy/missing mirrors.
            candidate_payload = payload_by_image_id.get(frame_id)
            if candidate_payload is not None:
                candidate_current = EvidenceImageMetadata.model_validate(candidate_payload)
            else:
                candidate_current = read_embedded_metadata(destination)
            candidate_revision_counter += 1
            candidate_revision_id = f"MR{candidate_revision_counter:06d}"
            candidate_raw = candidate_current.model_dump(mode="json")
            candidate_raw["image"]["origin"] = "candidate"
            candidate_raw["image"]["derivation"] = {
                "method": "duration-survey-candidate-retained-after-selection",
                "transformation_ids": [],
            }
            candidate_raw["links"]["candidate_ids"] = [f"VC{frame_id[1:]}"]
            candidate_raw["analysis"].update(
                {
                    "enrichment_level": "deterministic",
                    "semantic_status": "deterministic_only",
                    "latest_revision_id": candidate_revision_id,
                    "revision_number": candidate_current.analysis.revision_number + 1,
                }
            )
            candidate_raw["integrity"].update(
                {
                    "previous_revision_id": candidate_current.analysis.latest_revision_id,
                    "previous_payload_digest": candidate_current.integrity.payload_digest,
                    "canonical_revision_locator": (
                        f".state/vision/image-observations.json#{candidate_revision_id}"
                    ),
                    "canonical_revision_digest": hashlib.sha256(
                        json.dumps(
                            {
                                "revision_id": candidate_revision_id,
                                "image_id": frame_id,
                                "kind": "candidate",
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest(),
                }
            )
            candidate_prepared, candidate_file_hash = embed_metadata_with_file_hash(
                destination, candidate_raw, verify_source_pixels=False
            )
            candidate_prepared_dict = candidate_prepared.model_dump(mode="json")
            candidate_payloads = [
                candidate_prepared_dict
                if str(payload.get("image", {}).get("image_id")) == frame_id
                else payload
                for payload in candidate_payloads
            ]
            candidate_revisions.append(
                {
                    "revision_id": candidate_revision_id,
                    "revision_number": candidate_current.analysis.revision_number + 1,
                    "image_id": frame_id,
                    "base_revision_id": candidate_current.analysis.latest_revision_id,
                    "observation_ids": [],
                    "added_claim_ids": [],
                    "confirmed_claim_ids": [],
                    "narrowed_claim_ids": [],
                    "disputed_claim_ids": [],
                    "rejected_claim_ids": [],
                    "superseded_claim_ids": [],
                    "unresolved_claim_ids": [],
                    "previous_payload_digest": candidate_current.integrity.payload_digest,
                    "new_payload_digest": candidate_prepared.integrity.payload_digest,
                    "reconciliation_method": "candidate-origin-enrichment",
                    "actor": "deterministic-visual-pipeline",
                    "stale_base_reconciled": False,
                    "pixel_invariance_verified": True,
                    "embedded_write_verified": True,
                    "read_back_verified": True,
                    "canonical_mirror_committed": True,
                    "created_at_utc": _now(),
                }
            )
            candidate_frame.update(
                {
                    "latest_revision_id": candidate_revision_id,
                    "metadata_payload_digest": candidate_prepared.integrity.payload_digest,
                    "metadata_sufficiency_state": candidate_prepared.analysis.sufficiency.status,
                    "file_hash": candidate_file_hash,
                    "metadata": candidate_prepared_dict,
                }
            )
            candidate_frames.append(candidate_frame)

    frames = [frame for frame in frames if str(frame["frame_id"]) in selected_ids]
    payloads = [
        payload
        for payload in payloads
        if str(payload.get("image", {}).get("image_id")) in selected_ids
    ]
    revisions = [
        revision for revision in revisions if str(revision.get("image_id")) in selected_ids
    ]
    events = [
        event
        for event in events
        if any(str(frame_id) in selected_ids for frame_id in event.get("evidence_frame_ids", []))
    ]
    kept_event_ids = {str(event["event_id"]) for event in events}
    reviews = [
        review
        for review in reviews
        if not review.get("frame_ids")
        or any(str(frame_id) in selected_ids for frame_id in review.get("frame_ids", []))
    ]
    kept_revision_ids = {str(revision["revision_id"]) for revision in revisions}
    for block in blocks:
        block["frame_ids"] = [
            frame_id for frame_id in block.get("frame_ids", []) if str(frame_id) in selected_ids
        ]
        block["visual_event_ids"] = [
            event_id
            for event_id in block.get("visual_event_ids", [])
            if str(event_id) in kept_event_ids
        ]
        block["metadata_revision_ids"] = [
            revision_id
            for revision_id in block.get("metadata_revision_ids", [])
            if str(revision_id) in kept_revision_ids
        ]
        block["uncertainty"] = list(dict.fromkeys(block.get("uncertainty", [])))

    frames.sort(key=lambda item: (int(item["actual_ms"]), str(item["frame_id"])))
    block_by_id = {str(block["block_id"]): block for block in blocks}
    emit_progress(
        "selection_completed",
        selected_full_frame_count=len(frames),
        candidate_frame_count=len(candidate_frames),
    )

    # A consequential pixel/OCR change is a sequence, not an isolated still.
    # The survey may not decode a distinct interaction frame, so represent the
    # strongest safe relationship as before -> after using the nearest retained
    # full frame. Keep it in event state, block links, frame records, and the
    # embedded envelope so a later agent can inspect the group directly.
    full_frames = [frame for frame in frames if not frame.get("parent_full_frame_id")]
    selected_by_id = {item.candidate.frame_id: item for item in selection.selected}
    event_by_id = {str(event.get("event_id")): event for event in events}
    event_group_by_frame: dict[str, dict[str, list[str]]] = {}
    group_id_by_frame: dict[str, str] = {}
    for index, frame in enumerate(full_frames):
        frame_id = str(frame["frame_id"])
        scored = selected_by_id.get(frame_id)
        changed = bool(scored and scored.candidate.consequential_change)
        if not changed or index == 0:
            continue
        before_id = str(full_frames[index - 1]["frame_id"])
        event_ids = [str(value) for value in frame.get("linked_event_ids", [])]
        if not event_ids:
            continue
        event_id = event_ids[0]
        roles = {"before": [before_id], "after": [frame_id]}
        event_group_by_frame[frame_id] = roles
        event_group_by_frame[before_id] = roles
        group_id_by_frame[before_id] = event_id
        group_id_by_frame[frame_id] = event_id
        frame["evidence_role"] = "after"
        previous = full_frames[index - 1]
        # A frame can be the after-state of one change and the before-state of
        # another. Preserve the stronger change role when that occurs.
        if previous.get("evidence_role") != "after":
            previous["evidence_role"] = "before"
        previous.setdefault("linked_event_ids", [])
        if event_id not in previous["linked_event_ids"]:
            previous["linked_event_ids"].append(event_id)
        event = event_by_id.get(event_id)
        if event is None:
            continue
        event["evidence_frame_ids"] = list(
            dict.fromkeys([*event.get("evidence_frame_ids", []), before_id, frame_id])
        )
        event["before_action_after_roles"] = roles
        event["start_ms"] = int(previous["actual_ms"])
        event["end_ms"] = int(frame["actual_ms"])
        for linked_block_id in {
            *[str(value) for value in previous.get("linked_block_ids", [])],
            *[str(value) for value in frame.get("linked_block_ids", [])],
        }:
            linked_block = block_by_id.get(linked_block_id)
            if linked_block is None:
                continue
            linked_block.setdefault("visual_event_ids", [])
            if event_id not in linked_block["visual_event_ids"]:
                linked_block["visual_event_ids"].append(event_id)
        for review in reviews:
            if event_id not in [str(value) for value in review.get("event_ids", [])]:
                continue
            review.setdefault("frame_ids", [])
            if before_id not in review["frame_ids"]:
                review["frame_ids"].append(before_id)

    payload_history: list[dict[str, Any]] = []
    # Candidate-origin revisions were allocated before selected-frame
    # deterministic revisions so every global MR identifier remains unique.
    next_revision_number = max(
        candidate_revision_counter,
        max((int(str(item["revision_id"])[2:]) for item in revisions), default=0),
    )
    enrichment_jobs: list[dict[str, Any]] = []
    payload_index_by_image_id = {
        str(payload.get("image", {}).get("image_id")): index
        for index, payload in enumerate(payloads)
    }
    for index, frame in enumerate(frames):
        frame_id = str(frame["frame_id"])
        path = project_dir / str(frame["full_frame_path"])
        # The legacy creation/candidate transaction already validated the
        # canonical payload during its metadata write and stored that exact
        # mirror on the frame.  Re-validate the immutable mirror in memory for
        # this same transaction; legacy callers without it retain the guarded
        # embedded-read fallback.
        cached_payload = frame.get("metadata")
        current = (
            prepare_metadata_payload(cached_payload)
            if isinstance(cached_payload, Mapping)
            else read_embedded_metadata(path)
        )
        payload_history.append(current.model_dump(mode="json"))
        next_revision_number += 1
        revision_id = f"MR{next_revision_number:06d}"
        neighbor_ids = [
            str(frames[position]["frame_id"])
            for position in (index - 1, index + 1)
            if 0 <= position < len(frames)
        ]
        difference_regions: list[dict[str, Any]] = []
        difference = difference_by_id.get(frame_id)
        if difference is not None and index > 0:
            previous_id = str(frames[index - 1]["frame_id"])
            difference_regions = [
                {
                    "neighbor_image_id": previous_id,
                    "xywh": list(region.xywh),
                    "changed_ratio": region.changed_ratio,
                    "mean_difference": region.mean_difference,
                }
                for region in difference.regions[:8]
                if region.changed_ratio > 0 or region.mean_difference > 0
            ]
        selected = next(item for item in selection.selected if item.candidate.frame_id == frame_id)
        ocr_observation = raw_ocr_by_frame.get(frame_id)
        ocr_ids = [ocr_observation.observation_id] if ocr_observation is not None else []
        updated = current.model_dump(mode="json")
        updated["links"]["neighbor_image_ids"] = neighbor_ids
        updated["links"]["candidate_ids"] = [f"VC{frame_id[1:]}"]
        updated["links"]["ocr_observation_ids"] = ocr_ids
        updated["links"]["visual_event_ids"] = [
            str(value) for value in frame.get("linked_event_ids", [])
        ]
        updated["image"]["role"] = str(frame.get("evidence_role") or "context")
        updated["knowledge"]["selection_reason"] = selected.selected_reason
        updated["knowledge"]["why_it_matters"] = (
            "Retained by duration-aware scoring after pixel, quality, novelty, and OCR-change checks."
        )
        group_id = group_id_by_frame.get(frame_id)
        group_roles = event_group_by_frame.get(frame_id)
        if group_id is not None and group_roles is not None:
            updated["knowledge"]["before_action_after"] = {
                "group_id": group_id,
                "before_image_id": (
                    group_roles["before"][0] if group_roles.get("before") else None
                ),
                "action_image_ids": [],
                "after_image_ids": list(group_roles.get("after", [])),
                "supported_change_claim_ids": [],
            }
        updated["analysis"].update(
            {
                "enrichment_level": "deterministic",
                "semantic_status": "deterministic_only",
                "latest_revision_id": revision_id,
                "revision_number": 2,
                "frame_quality": asdict(quality_by_id[frame_id]),
                "scene_relationships": ["chronological-neighbor"] if neighbor_ids else [],
                "difference_regions": difference_regions,
                "ocr_observation_ids": ocr_ids,
                "neighbor_image_ids": neighbor_ids,
                "before_action_after_membership": str(frame.get("evidence_role") or "context"),
            }
        )
        updated["integrity"].update(
            {
                "previous_revision_id": current.analysis.latest_revision_id,
                "previous_payload_digest": current.integrity.payload_digest,
                "canonical_revision_locator": (
                    f".state/vision/image-observations.json#{revision_id}"
                ),
                "canonical_revision_digest": hashlib.sha256(
                    json.dumps(
                        {"revision_id": revision_id, "image_id": frame_id, "kind": "deterministic"},
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            }
        )
        enrichment_jobs.append(
            {
                "frame": frame,
                "frame_id": frame_id,
                "path": path,
                "current": current,
                "revision_id": revision_id,
                "updated": updated,
                "selected": selected,
                "ocr_observation": ocr_observation,
                "ocr_ids": ocr_ids,
                "difference_regions": difference_regions,
            }
        )

    # PNG metadata writes preserve the original pixel stream and perform a
    # read-back verification. They are independent across images, so run the
    # expensive byte-copy/read-back work in a bounded pool, then commit all
    # canonical payloads, revisions, and links in deterministic frame order.
    with _executor_context(
        worker_pool,
        max_workers=_visual_frame_workers(),
        thread_name_prefix="vsr-metadata",
    ) as pool:
        futures = [
            pool.submit(
                embed_metadata_with_file_hash,
                job["path"],
                job["updated"],
                verify_source_pixels=False,
                verify_decoded_pixels=False,
            )
            for job in enrichment_jobs
        ]
        for job, future in zip(enrichment_jobs, futures, strict=True):
            frame = cast(dict[str, Any], job["frame"])
            frame_id = str(job["frame_id"])
            path = cast(Path, job["path"])
            current = job["current"]
            revision_id = str(job["revision_id"])
            selected = job["selected"]
            ocr_observation = job["ocr_observation"]
            ocr_ids = cast(list[str], job["ocr_ids"])
            difference_regions = cast(list[dict[str, Any]], job["difference_regions"])
            prepared, file_hash = future.result()
            prepared_dict = prepared.model_dump(mode="json")
            payload_index = payload_index_by_image_id.get(frame_id)
            if payload_index is None:
                payloads.append(prepared_dict)
                payload_index_by_image_id[frame_id] = len(payloads) - 1
            else:
                payloads[payload_index] = prepared_dict
            revisions.append(
                {
                    "revision_id": revision_id,
                    "revision_number": 2,
                    "image_id": frame_id,
                    "base_revision_id": current.analysis.latest_revision_id,
                    "observation_ids": [],
                    "added_claim_ids": [],
                    "confirmed_claim_ids": [],
                    "narrowed_claim_ids": [],
                    "disputed_claim_ids": [],
                    "rejected_claim_ids": [],
                    "superseded_claim_ids": [],
                    "unresolved_claim_ids": [],
                    "previous_payload_digest": current.integrity.payload_digest,
                    "new_payload_digest": prepared.integrity.payload_digest,
                    "reconciliation_method": "deterministic-enrichment",
                    "actor": "deterministic-visual-pipeline",
                    "stale_base_reconciled": False,
                    "pixel_invariance_verified": current.image.pixel_hash
                    == prepared.image.pixel_hash,
                    "embedded_write_verified": True,
                    "read_back_verified": True,
                    "canonical_mirror_committed": True,
                    "created_at_utc": _now(),
                }
            )
            frame.update(
                {
                    "quality_scores": asdict(quality_by_id[frame_id]),
                    "perceptual_hashes": {
                        "dhash-8": dhash_by_id[frame_id],
                        "dhash-8-algorithm": PERCEPTUAL_DHASH_ALGORITHM,
                        "dhash-8-verified": PERCEPTUAL_DHASH_VERIFIED,
                    },
                    "region_hashes": {
                        f"difference-{position + 1}": hashlib.sha256(
                            json.dumps(region, sort_keys=True).encode()
                        ).hexdigest()
                        for position, region in enumerate(difference_regions)
                    },
                    "metadata_payload_digest": prepared.integrity.payload_digest,
                    "latest_revision_id": revision_id,
                    "ocr_observation_ids": ocr_ids,
                    "selection_reason": selected.selected_reason,
                    "evidence_role": str(
                        frame.get("evidence_role") or selected.candidate.evidence_role
                    ),
                    "file_hash": file_hash,
                    "metadata": prepared_dict,
                }
            )
            for block_id in frame.get("linked_block_ids", []):
                linked_block = block_by_id.get(str(block_id))
                if linked_block is None:
                    continue
                linked_block.setdefault("metadata_revision_ids", []).append(revision_id)
                if ocr_observation is not None and ocr_observation.normalized_interpretation:
                    value = ocr_observation.normalized_interpretation
                    if value not in linked_block.setdefault("on_screen_text", []):
                        linked_block["on_screen_text"].append(value)
                    if ocr_observation.uncertain_characters:
                        linked_block.setdefault("uncertainty", []).append(
                            f"OCR {ocr_observation.observation_id} contains uncertain characters."
                    )

    emit_progress("metadata_completed", retained_frame_count=len(frames))
    event_by_frame: dict[str, dict[str, Any]] = {}
    frame_by_id = {str(item["frame_id"]): item for item in frames}
    for event in events:
        evidence_frame_ids = [str(value) for value in event.get("evidence_frame_ids", [])]
        event_ocr_ids: list[str] = []
        revision_ids: list[str] = []
        for frame_id in evidence_frame_ids:
            event_by_frame[frame_id] = event
            event_frame = frame_by_id.get(frame_id)
            if event_frame is None:
                continue
            event_ocr_ids.extend(str(value) for value in event_frame.get("ocr_observation_ids", []))
            latest_revision_id = event_frame.get("latest_revision_id")
            if latest_revision_id is not None:
                revision_ids.append(str(latest_revision_id))
        event["ocr_observation_ids"] = list(dict.fromkeys(event_ocr_ids))
        event["metadata_revision_ids"] = list(dict.fromkeys(revision_ids))

    packet_dir = project_dir / ".state" / "vision" / "packets"
    written_packet_event_ids: set[str] = set()
    packet_payloads: dict[str, dict[str, Any]] = {}
    for index, frame in enumerate(frames):
        frame_id = str(frame["frame_id"])
        event = event_by_frame.get(frame_id)
        if event is None:
            continue
        event_id = str(event["event_id"])
        # Full frames precede derived crops in ``frames``.  A packet is keyed
        # by event, so rewriting it for every crop (or every before/after frame
        # in the same event) only repeats an fsync-heavy atomic write. Crop
        # records have already been appended to the packet during crop
        # creation; keep the first complete full-frame packet as canonical.
        if event_id in written_packet_event_ids:
            continue
        written_packet_event_ids.add(event_id)
        packet_frames: list[dict[str, Any]] = []
        for position in (index - 1, index, index + 1):
            if not 0 <= position < len(frames):
                continue
            neighbor = frames[position]
            if abs(int(neighbor["actual_ms"]) - int(frame["actual_ms"])) > 15_000:
                continue
            role = "focus"
            if position < index:
                role = "before"
            elif position > index:
                role = "after"
            neighbor_id = str(neighbor["frame_id"])
            packet_frames.append(
                {
                    "frame_id": neighbor_id,
                    "path": str(neighbor["full_frame_path"]),
                    "role": role,
                    "requested_ms": int(neighbor["requested_ms"]),
                    "actual_ms": int(neighbor["actual_ms"]),
                    "raw_pts": neighbor.get("pts"),
                    "time_base": neighbor.get("time_base"),
                    "metadata_revision_id": neighbor.get("latest_revision_id"),
                    "difference_regions": [],
                }
            )
        # Keep every packet inside the schema's bounded 15-second evidence
        # window.  A neighboring frame can be within 15 seconds of the focus
        # frame while the before+after pair spans more than 15 seconds.
        if packet_frames:
            focus_ms = int(frame["actual_ms"])
            packet_frames = _bounded_packet_frames(packet_frames, focus_ms=focus_ms)
            if not any(item["frame_id"] == frame_id for item in packet_frames):
                packet_frames.append(
                    {
                        "frame_id": frame_id,
                        "path": str(frame["full_frame_path"]),
                        "role": "focus",
                        "requested_ms": int(frame["requested_ms"]),
                        "actual_ms": focus_ms,
                        "raw_pts": frame.get("pts"),
                        "time_base": frame.get("time_base"),
                        "metadata_revision_id": frame.get("latest_revision_id"),
                        "difference_regions": [],
                    }
                )
        nearby = [
            block_by_id[block_id]
            for block_id in frame.get("linked_block_ids", [])
            if block_id in block_by_id
        ]
        packet_ocr = []
        for packet_frame in packet_frames:
            observation = raw_ocr_by_frame.get(str(packet_frame["frame_id"]))
            if observation is None:
                continue
            packet_ocr.append(
                {
                    "observation_id": observation.observation_id,
                    "frame_id": observation.frame_id,
                    "crop_id": observation.crop_id,
                    "raw_engine_text": observation.raw_engine_text,
                    "normalized_interpretation": observation.normalized_interpretation,
                    "confidence": observation.confidence,
                    "bounding_region": observation.bounding_region,
                    "uncertain_characters": list(observation.uncertain_characters),
                }
            )
        packet = create_vision_packet(
            candidate_id=event_id,
            frames=packet_frames,
            questions=[
                "What meaningful visible state or exact text is needed to understand this block?",
                "Does the before/focus/after sequence show a consequential state change?",
            ],
            nearby_transcript=nearby,
            raw_ocr=packet_ocr,
            scene_motion_metadata={
                "quality": frame.get("quality_scores", {}),
                "selection_reason": frame.get("selection_reason"),
            },
            prior_event_context=events[index - 1] if index > 0 else None,
            next_event_context=events[index + 1] if index + 1 < len(events) else None,
        )
        packet_payloads[event_id] = packet.model_dump(mode="json")
        atomic_write_json(packet_dir / f"{event_id}.json", packet_payloads[event_id])

    emit_progress("packet_completed", packet_count=len(written_packet_event_ids))

    # A localized, consequential difference justifies a content-aware crop. The
    # full frame remains canonical context and the crop receives its own embedded
    # identity, provenance, revision chain, and public CLI addressable record.
    from PIL import Image

    crop_frames: list[dict[str, Any]] = []
    updated_packet_event_ids: set[str] = set()
    crop_specs: list[dict[str, Any]] = []
    for parent in list(frames):
        parent_id = str(parent["frame_id"])
        difference = difference_by_id.get(parent_id)
        if difference is None or difference.maximum_region_change < 0.20:
            continue
        changed_regions = [
            item
            for item in difference.regions
            if item.changed_ratio >= 0.05 or item.mean_difference >= 0.02
        ]
        region = changed_regions[0] if changed_regions else None
        if region is None:
            continue
        crop_id = f"{parent_id}-C01"
        crop_relative = f"evidence/crops/{evidence_filename(crop_id, int(parent['actual_ms']))}"
        crop_specs.append(
            {
                "parent": parent,
                "parent_id": parent_id,
                "parent_path": project_dir / str(parent["full_frame_path"]),
                "crop_id": crop_id,
                "crop_relative": crop_relative,
                "changed_regions": changed_regions,
                "region": region,
            }
        )

    def prepare_crop(spec: Mapping[str, Any]) -> dict[str, Any] | None:
        """Decode, write, and analyze one crop without touching canonical state."""

        parent = cast(dict[str, Any], spec["parent"])
        parent_path = cast(Path, spec["parent_path"])
        changed_regions = cast(Sequence[Any], spec["changed_regions"])
        union_left = min(int(item.xywh[0]) for item in changed_regions)
        union_top = min(int(item.xywh[1]) for item in changed_regions)
        union_right = max(int(item.xywh[0] + item.xywh[2]) for item in changed_regions)
        union_bottom = max(int(item.xywh[1] + item.xywh[3]) for item in changed_regions)
        x, y, width, height = (
            union_left,
            union_top,
            union_right - union_left,
            union_bottom - union_top,
        )
        crop_path = project_dir / str(spec["crop_relative"])
        with Image.open(parent_path) as image:
            image.load()
            padding = max(8, min(image.width, image.height) // 40)
            left = max(0, x - padding)
            top = max(0, y - padding)
            right = min(image.width, x + width + padding)
            bottom = min(image.height, y + height + padding)
            crop_xywh = (left, top, right - left, bottom - top)
            if _crop_covers_full_frame(crop_xywh, image.width, image.height):
                # A crop that covers the complete parent is not localized
                # evidence. Keep the authoritative parent frame and avoid
                # writing a duplicate with a misleading crop ID.
                return None
            crop_image = image.crop((left, top, right, bottom))
            try:
                crop_image.save(crop_path, "PNG")
                # The crop is already decoded in memory. Reuse it for
                # deterministic quality/hash fields instead of decoding the
                # just-written PNG several additional times.
                crop_pixel_hash = normalized_pixel_hash_image(crop_image)
                crop_quality = assess_frame_quality(crop_image)
                crop_dhash = perceptual_dhash(crop_image)
            finally:
                crop_image.close()
        return {
            "parent": parent,
            "parent_id": str(spec["parent_id"]),
            "parent_path": parent_path,
            "crop_id": str(spec["crop_id"]),
            "crop_relative": str(spec["crop_relative"]),
            "crop_xywh": crop_xywh,
            "region": spec["region"],
            "crop_pixel_hash": crop_pixel_hash,
            "crop_quality": crop_quality,
            "crop_dhash": crop_dhash,
        }

    # Crops are independent reads/encodes of already-materialized evidence.
    # Use their own bounded analysis pool instead of the decoder pool supplied
    # by the caller: the latter is intentionally capped at four workers to keep
    # FFmpeg seeks from oversubscribing codec threads, while crop work benefits
    # from the wider deterministic-analysis budget.  The separate context is
    # shut down before canonical writes continue, keeping peak memory bounded
    # and preventing thread leakage across pipeline runs.
    with ThreadPoolExecutor(
        max_workers=_visual_crop_workers(),
        thread_name_prefix="vsr-crop-prep",
    ) as pool:
        crop_prepare_futures = [pool.submit(prepare_crop, spec) for spec in crop_specs]
        crop_preparations: list[dict[str, Any]] = []
        for crop_future in crop_prepare_futures:
            crop_preparation = crop_future.result()
            if crop_preparation is not None:
                crop_preparations.append(crop_preparation)
    skipped_full_frame_crop_count = len(crop_specs) - len(crop_preparations)
    emit_progress(
        "crop_preparation_completed",
        crop_count=len(crop_preparations),
        skipped_full_frame_crop_count=skipped_full_frame_crop_count,
    )

    for crop_preparation in crop_preparations:
        parent = cast(dict[str, Any], crop_preparation["parent"])
        parent_id = str(crop_preparation["parent_id"])
        parent_path = cast(Path, crop_preparation["parent_path"])
        crop_id = str(crop_preparation["crop_id"])
        crop_relative = str(crop_preparation["crop_relative"])
        crop_path = project_dir / crop_relative
        crop_xywh = tuple(
            int(value) for value in cast(Sequence[Any], crop_preparation["crop_xywh"])
        )
        region = crop_preparation["region"]
        crop_pixel_hash = str(crop_preparation["crop_pixel_hash"])
        crop_quality = crop_preparation["crop_quality"]
        crop_dhash = str(crop_preparation["crop_dhash"])

        transformation_id = sequential_id("transformation", len(crop_frames) + 1)
        next_revision_number += 1
        creation_revision_id = f"MR{next_revision_number:06d}"
        # The selected full-frame record already carries the canonical payload
        # produced by the deterministic enrichment pass above. Reuse that
        # validated in-memory mirror instead of reparsing the parent PNG for
        # every localized crop; fall back to an embedded read only for callers
        # that provide legacy frame records without the mirror.
        parent_payload = parent.get("metadata")
        if isinstance(parent_payload, Mapping):
            creation_payload = json.loads(json.dumps(parent_payload))
        else:
            creation_payload = read_embedded_metadata(parent_path).model_dump(mode="json")
        creation_payload["image"].update(
            {
                "image_id": crop_id,
                "parent_full_frame_id": parent_id,
                "origin": "derived_crop",
                "derivation": {
                    "method": "localized-pixel-difference-region-with-context-padding",
                    "transformation_ids": [transformation_id],
                },
                "crop_xywh": list(crop_xywh),
                "width": crop_xywh[2],
                "height": crop_xywh[3],
                "role": "detail",
                "pixel_hash": {
                    "algorithm": "sha256-rgba8-srgb-v1",
                    "value": crop_pixel_hash,
                },
            }
        )
        creation_payload["links"].update(
            {
                "neighbor_image_ids": [parent_id],
                "ocr_observation_ids": [],
                "candidate_ids": [f"VC{parent_id[1:]}-C01"],
            }
        )
        creation_payload["knowledge"].update(
            {
                "selection_reason": (
                    "Localized changed region retained with padding; the linked full frame preserves context."
                ),
                "why_it_matters": (
                    "Makes a consequential small state change inspectable without discarding full-frame evidence."
                ),
                "current_factual_description": None,
                "claims": [],
                "supported_claim_ids": [],
                "disputed_claim_ids": [],
                "rejected_claim_ids": [],
                "unresolved_claim_ids": [],
                "explicit_unknowns": ["Semantic meaning of the changed region is pending review."],
                "before_action_after": None,
            }
        )
        creation_payload["analysis"].update(
            {
                "enrichment_level": "creation",
                "semantic_status": "unobserved",
                "latest_revision_id": creation_revision_id,
                "revision_number": 1,
                "observation_history": [],
                "frame_quality": {},
                "scene_relationships": [],
                "difference_regions": [],
                "ocr_observation_ids": [],
                "neighbor_image_ids": [parent_id],
                "before_action_after_membership": "detail",
            }
        )
        creation_payload["integrity"].update(
            {
                "previous_revision_id": None,
                "previous_payload_digest": None,
                "canonical_revision_locator": (
                    f".state/vision/image-observations.json#{creation_revision_id}"
                ),
                "canonical_revision_digest": hashlib.sha256(
                    json.dumps(
                        {
                            "revision_id": creation_revision_id,
                            "image_id": crop_id,
                            "kind": "creation",
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            }
        )
        # Keep the creation envelope and revision in the append-only ledger,
        # but write only the final deterministic envelope to disk. The previous
        # implementation wrote the transient creation payload and immediately
        # overwrote it, doubling PNG metadata I/O for every crop.
        created = prepare_metadata_payload(creation_payload)
        payload_history.append(created.model_dump(mode="json"))
        revisions.append(
            {
                "revision_id": creation_revision_id,
                "revision_number": 1,
                "image_id": crop_id,
                "base_revision_id": None,
                "observation_ids": [],
                "added_claim_ids": [],
                "confirmed_claim_ids": [],
                "narrowed_claim_ids": [],
                "disputed_claim_ids": [],
                "rejected_claim_ids": [],
                "superseded_claim_ids": [],
                "unresolved_claim_ids": [],
                "previous_payload_digest": None,
                "new_payload_digest": created.integrity.payload_digest,
                "reconciliation_method": "creation",
                "actor": "deterministic-visual-pipeline",
                "stale_base_reconciled": False,
                "pixel_invariance_verified": True,
                "embedded_write_verified": True,
                "read_back_verified": True,
                "canonical_mirror_committed": True,
                "created_at_utc": _now(),
            }
        )
        next_revision_number += 1
        deterministic_revision_id = f"MR{next_revision_number:06d}"
        enriched_payload = created.model_dump(mode="json")
        enriched_payload["analysis"].update(
            {
                "enrichment_level": "deterministic",
                "semantic_status": "deterministic_only",
                "latest_revision_id": deterministic_revision_id,
                "revision_number": 2,
                "frame_quality": asdict(crop_quality),
                "scene_relationships": [f"crop-of:{parent_id}"],
                "difference_regions": [
                    {
                        "neighbor_image_id": parent_id,
                        "xywh": [0, 0, crop_xywh[2], crop_xywh[3]],
                        "changed_ratio": region.changed_ratio,
                        "mean_difference": region.mean_difference,
                    }
                ],
            }
        )
        enriched_payload["integrity"].update(
            {
                "previous_revision_id": creation_revision_id,
                "previous_payload_digest": created.integrity.payload_digest,
                "canonical_revision_locator": (
                    f".state/vision/image-observations.json#{deterministic_revision_id}"
                ),
                "canonical_revision_digest": hashlib.sha256(
                    json.dumps(
                        {
                            "revision_id": deterministic_revision_id,
                            "image_id": crop_id,
                            "kind": "deterministic",
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            }
        )
        enriched, crop_file_hash = embed_metadata_with_file_hash(
            crop_path,
            enriched_payload,
            verify_source_pixels=False,
            verify_decoded_pixels=False,
        )
        revisions.append(
            {
                "revision_id": deterministic_revision_id,
                "revision_number": 2,
                "image_id": crop_id,
                "base_revision_id": creation_revision_id,
                "observation_ids": [],
                "added_claim_ids": [],
                "confirmed_claim_ids": [],
                "narrowed_claim_ids": [],
                "disputed_claim_ids": [],
                "rejected_claim_ids": [],
                "superseded_claim_ids": [],
                "unresolved_claim_ids": [],
                "previous_payload_digest": created.integrity.payload_digest,
                "new_payload_digest": enriched.integrity.payload_digest,
                "reconciliation_method": "deterministic-enrichment",
                "actor": "deterministic-visual-pipeline",
                "stale_base_reconciled": False,
                "pixel_invariance_verified": created.image.pixel_hash == enriched.image.pixel_hash,
                "embedded_write_verified": True,
                "read_back_verified": True,
                "canonical_mirror_committed": True,
                "created_at_utc": _now(),
            }
        )
        payloads.append(enriched.model_dump(mode="json"))
        crop_record = {
            "crop_id": crop_id,
            "parent_full_frame_id": parent_id,
            "crop_xywh": list(crop_xywh),
            "path": crop_relative,
            "reason": "localized consequential pixel difference",
            "method": "difference-region-with-padding",
        }
        parent.setdefault("crops", []).append(crop_record)
        crop_frame = {
            "frame_id": crop_id,
            "requested_ms": parent["requested_ms"],
            "actual_ms": parent["actual_ms"],
            "pts": parent.get("pts"),
            "time_base": parent.get("time_base"),
            "frame_index": parent.get("frame_index"),
            "offset_ms": parent.get("offset_ms"),
            "timestamp_source": parent.get("timestamp_source"),
            "timing_estimated": parent.get("timing_estimated", False),
            "full_frame_path": crop_relative,
            "parent_full_frame_id": parent_id,
            "crop_xywh": list(crop_xywh),
            "crops": [],
            "scene_id": parent.get("scene_id"),
            "quality_scores": asdict(crop_quality),
            "perceptual_hashes": {
                "dhash-8": crop_dhash,
                "dhash-8-algorithm": PERCEPTUAL_DHASH_ALGORITHM,
                "dhash-8-verified": PERCEPTUAL_DHASH_VERIFIED,
            },
            "region_hashes": {},
            "pixel_hash": enriched.image.pixel_hash.model_dump(mode="json"),
            "file_hash": crop_file_hash,
            "metadata_payload_digest": enriched.integrity.payload_digest,
            "latest_revision_id": deterministic_revision_id,
            "metadata_sufficiency_state": "semantic_observer_unavailable",
            "ocr_observation_ids": [],
            "selection_reason": crop_record["reason"],
            "evidence_role": "detail",
            "linked_event_ids": list(parent.get("linked_event_ids", [])),
            "linked_block_ids": list(parent.get("linked_block_ids", [])),
            "verification_status": "unverified",
            "supported_claim_ids": [],
            "disputed_claim_ids": [],
            "unresolved_claim_ids": [],
            "description": "Changed region detail; semantic description pending review.",
            "path": crop_relative,
            "final": False,
            "metadata": enriched.model_dump(mode="json"),
        }
        crop_frames.append(crop_frame)
        for block_id in crop_frame["linked_block_ids"]:
            linked_block = block_by_id.get(str(block_id))
            if linked_block is not None:
                linked_block.setdefault("metadata_revision_ids", []).extend(
                    [creation_revision_id, deterministic_revision_id]
                )
                linked_block.setdefault("transformation_ids", []).append(transformation_id)
        for event_id in crop_frame["linked_event_ids"]:
            event = next((item for item in events if item["event_id"] == event_id), None)
            if event is not None:
                event.setdefault("evidence_frame_ids", []).append(crop_id)
                event.setdefault("metadata_revision_ids", []).append(deterministic_revision_id)
            packet_path = packet_dir / f"{event_id}.json"
            packet_data = packet_payloads.get(event_id)
            if packet_data is None and packet_path.is_file():
                packet_data = json.loads(packet_path.read_text(encoding="utf-8"))
            if packet_data is not None:
                packet_data.setdefault("frames", []).append(
                    {
                        "frame_id": crop_id,
                        "path": crop_relative,
                        "role": "focus",
                        "requested_ms": int(parent["requested_ms"]),
                        "actual_ms": int(parent["actual_ms"]),
                        "raw_pts": parent.get("pts"),
                        "time_base": parent.get("time_base"),
                        "metadata_revision_id": deterministic_revision_id,
                        "difference_regions": [
                            {
                                "xywh": [0, 0, crop_xywh[2], crop_xywh[3]],
                                "changed_ratio": region.changed_ratio,
                                "description": "localized changed-state crop",
                            }
                        ],
                    }
                )
                packet_data["frames"] = _bounded_packet_frames(
                    packet_data.get("frames", []),
                    focus_ms=int(parent["actual_ms"]),
                    max_span_ms=int(packet_data.get("max_span_ms", 15_000)),
                )
                allowed_packet_frame_ids = {
                    str(item["frame_id"]) for item in packet_data["frames"]
                }
                packet_data["raw_ocr"] = [
                    observation
                    for observation in packet_data.get("raw_ocr", [])
                    if str(observation.get("frame_id")) in allowed_packet_frame_ids
                ]
                # Keep all crop mutations in memory and validate/write once per
                # changed event after the preparation/commit loop. This avoids
                # repeated PNG-adjacent packet reads and fsyncs while preserving
                # the exact append-and-bound order for every crop.
                packet_payloads[event_id] = packet_data
                updated_packet_event_ids.add(event_id)
        for review in reviews:
            if parent_id in review.get("frame_ids", []):
                review["frame_ids"].append(crop_id)
                review["metadata_revision_ids"].append(deterministic_revision_id)
        # One justified crop per changed parent is sufficient; further regions
        # remain available in deterministic metadata for targeted escalation.

    frames.extend(crop_frames)

    for event_id in sorted(updated_packet_event_ids):
        packet_path = packet_dir / f"{event_id}.json"
        validated_packet = VisionPacket.model_validate(packet_payloads[event_id])
        packet_payloads[event_id] = validated_packet.model_dump(mode="json")
        atomic_write_json(packet_path, packet_payloads[event_id])

    # The legacy extractor can leave a compact handoff object at an event path,
    # and shared evidence frames can cause a later event to overwrite the
    # frame-to-event lookup used above. Ensure every retained event has its own
    # complete, schema-valid semantic packet before a provider is invoked.
    all_frame_by_id = {str(item["frame_id"]): item for item in frames}
    ordered_events = sorted(events, key=lambda item: (int(item["start_ms"]), str(item["event_id"])))
    for event_index, event in enumerate(ordered_events):
        event_id = str(event["event_id"])
        packet_path = packet_dir / f"{event_id}.json"
        try:
            existing_packet = json.loads(packet_path.read_text(encoding="utf-8"))
            VisionPacket.model_validate(existing_packet)
        except (OSError, json.JSONDecodeError, ValueError):
            evidence_ids = [
                str(value)
                for value in event.get("evidence_frame_ids", [])
                if str(value) in all_frame_by_id
            ]
            if not evidence_ids:
                continue
            focus_id = evidence_ids[0]
            focus_frame = all_frame_by_id[focus_id]
            role_by_id: dict[str, str] = {}
            for role, role_ids in event.get("before_action_after_roles", {}).items():
                for role_id in role_ids:
                    role_by_id[str(role_id)] = str(role)
            packet_frame_ids = list(evidence_ids)
            full_frames = [item for item in frames if not item.get("parent_full_frame_id")]
            focus_position = next(
                (
                    position
                    for position, candidate in enumerate(full_frames)
                    if str(candidate["frame_id"]) == focus_id
                ),
                None,
            )
            if focus_position is not None:
                for position in (focus_position - 1, focus_position + 1):
                    if not 0 <= position < len(full_frames):
                        continue
                    neighbor = full_frames[position]
                    if abs(int(neighbor["actual_ms"]) - int(focus_frame["actual_ms"])) <= 15_000:
                        neighbor_id = str(neighbor["frame_id"])
                        if neighbor_id not in packet_frame_ids:
                            packet_frame_ids.append(neighbor_id)
                            role_by_id[neighbor_id] = (
                                "before" if position < focus_position else "after"
                            )
            packet_frames = []
            for frame_id in packet_frame_ids:
                packet_frame = all_frame_by_id[frame_id]
                packet_frames.append(
                    {
                        "frame_id": frame_id,
                        "path": str(packet_frame["full_frame_path"]),
                        "role": role_by_id.get(
                            frame_id, "focus" if frame_id == focus_id else "context"
                        ),
                        "requested_ms": int(packet_frame["requested_ms"]),
                        "actual_ms": int(packet_frame["actual_ms"]),
                        "raw_pts": packet_frame.get("pts"),
                        "time_base": packet_frame.get("time_base"),
                        "metadata_revision_id": packet_frame.get("latest_revision_id"),
                        "difference_regions": [],
                    }
                )
            # The packet schema bounds the complete before/focus/after window,
            # not each neighbor independently. Keep stale-packet rebuilding
            # subject to the same bounded evidence contract as the primary
            # packet path.
            focus_actual_ms = int(focus_frame["actual_ms"])
            packet_frames = _bounded_packet_frames(
                packet_frames, focus_ms=focus_actual_ms
            )
            packet_frame_ids = [str(item["frame_id"]) for item in packet_frames]
            nearby = [
                block
                for block in blocks
                if event_id in block.get("visual_event_ids", [])
                or any(frame_id in block.get("frame_ids", []) for frame_id in evidence_ids)
            ]
            packet_ocr = []
            for frame_id in packet_frame_ids:
                observation = raw_ocr_by_frame.get(frame_id)
                if observation is None:
                    continue
                packet_ocr.append(
                    {
                        "observation_id": observation.observation_id,
                        "frame_id": observation.frame_id,
                        "crop_id": observation.crop_id,
                        "raw_engine_text": observation.raw_engine_text,
                        "normalized_interpretation": observation.normalized_interpretation,
                        "confidence": observation.confidence,
                        "bounding_region": observation.bounding_region,
                        "uncertain_characters": list(observation.uncertain_characters),
                    }
                )
            packet = create_vision_packet(
                candidate_id=event_id,
                frames=packet_frames,
                questions=[
                    "What meaningful visible state or exact text is needed to understand this block?",
                    "Does the before/focus/after sequence show a consequential state change?",
                ],
                nearby_transcript=nearby,
                raw_ocr=packet_ocr,
                scene_motion_metadata={"event_type": event.get("event_type")},
                prior_event_context=(ordered_events[event_index - 1] if event_index > 0 else None),
                next_event_context=(
                    ordered_events[event_index + 1]
                    if event_index + 1 < len(ordered_events)
                    else None
                ),
            )
            atomic_write_json(packet_path, packet.model_dump(mode="json"))

    atomic_write_json(
        project_dir / ".state" / "vision" / "image-observations.json",
        {
            "schema_version": "1.0",
            "payloads": payloads,
            "payload_history": payload_history,
            "observations": [],
            "claims": [],
            "revisions": revisions,
            "candidate_frames": candidate_frames,
            "candidate_payloads": [
                payload
                for payload in candidate_payloads
                if str(payload.get("image", {}).get("image_id")) not in selected_ids
            ],
            "candidate_revisions": [
                revision
                for revision in candidate_revisions
                if str(revision.get("image_id")) not in selected_ids
            ],
            "selection": {
                "target_budget": selection.target_budget,
                "selected_frame_ids": sorted(selected_ids),
                "duplicate_frame_ids": list(selection.duplicate_frame_ids),
                "low_score_frame_ids": list(selection.low_score_frame_ids),
                # Keep the importance-tier/representative ledger beside the
                # historical selection fields.  This is deliberately an
                # additive receipt: existing readers can ignore it, while
                # audits can prove why a frame was retained or covered by a
                # low-importance representative without reopening pixels.
                "selection_audit": selection.audit,
            },
        },
        compact=True,
    )
    emit_progress(
        "completed",
        retained_frame_count=len(frames),
        crop_count=len(crop_frames),
        skipped_full_frame_crop_count=skipped_full_frame_crop_count,
        packet_count=len(written_packet_event_ids),
    )
    return frames, payloads, revisions, events, reviews, ocr_records


def _media_identity(
    source: Path, kind: str, *, content_hash: str | None = None
) -> tuple[dict[str, Any], Any | None]:
    from .ids import media_id

    resolved_content_hash = content_hash or sha256_file(source)
    stable_media_id = media_id(resolved_content_hash)
    if kind in {"video", "audio"}:
        from .media_probe import probe_media

        probe = probe_media(source)
        video = probe.video_streams[0] if probe.video_streams else None
        identity = {
            "schema_version": "1.0",
            "media_id": stable_media_id,
            "original_source_reference": str(source),
            "local_preserved_reference": None,
            "content_hash": resolved_content_hash,
            "byte_size": source.stat().st_size,
            "duration_ms": probe.duration_ms,
            "container": probe.container,
            "video_streams": [
                {
                    "index": item.index,
                    "codec": item.codec_name,
                    "language": item.language,
                    "disposition": {k: bool(v) for k, v in item.disposition.items()},
                    "metadata": dict(item.tags),
                }
                for item in probe.video_streams
            ],
            "audio_streams": [
                {
                    "index": item.index,
                    "codec": item.codec_name,
                    "language": item.language,
                    "disposition": {k: bool(v) for k, v in item.disposition.items()},
                    "metadata": dict(item.tags),
                }
                for item in probe.audio_streams
            ],
            "subtitle_streams": [
                {
                    "index": item.index,
                    "codec": item.codec_name,
                    "language": item.language,
                    "disposition": {k: bool(v) for k, v in item.disposition.items()},
                    "metadata": dict(item.tags),
                }
                for item in probe.subtitle_streams
            ],
            "frame_rate": video.r_frame_rate if video else None,
            "average_frame_rate": video.avg_frame_rate if video else None,
            "time_base": video.time_base if video else None,
            "variable_frame_rate": probe.variable_frame_rate,
            "resolution": [video.width, video.height]
            if video and video.width and video.height
            else None,
            "sample_aspect_ratio": video.sample_aspect_ratio if video else None,
            "rotation": video.rotation if video else None,
            "chapters": [
                {
                    "chapter_id": f"C{index + 1:03d}",
                    "title": chapter.title,
                    "start_ms": chapter.start_ms,
                    "end_ms": chapter.end_ms,
                }
                for index, chapter in enumerate(probe.chapters)
            ],
            "source_metadata": dict(probe.source_metadata),
            "acquisition_provenance": {"kind": "local_file", "network": False},
        }
        return identity, probe
    return (
        {
            "schema_version": "1.0",
            "media_id": stable_media_id,
            "original_source_reference": str(source),
            "local_preserved_reference": None,
            "content_hash": resolved_content_hash,
            "byte_size": source.stat().st_size,
            "duration_ms": None,
            "container": source.suffix.lstrip(".").casefold(),
            "video_streams": [],
            "audio_streams": [],
            "subtitle_streams": [],
            "frame_rate": None,
            "average_frame_rate": None,
            "time_base": None,
            "variable_frame_rate": None,
            "resolution": None,
            "sample_aspect_ratio": None,
            "rotation": None,
            "chapters": [],
            "source_metadata": {},
            "acquisition_provenance": {"kind": "local_file", "network": False},
        },
        None,
    )


def _reuse_visual_state(
    prior_project: dict[str, Any],
    blocks: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Reuse source-pixel visual state when only transcript inputs changed.

    This keeps semantic observations, claims, crops, and revision history append-only
    while the transcript stage recomputes its own candidate/repair decisions.
    """

    # The prior project was loaded from JSON, so all values here are already
    # JSON-compatible.  Copy the six immutable visual collections through one
    # encoder/decoder pass instead of serializing each large list separately;
    # long-form projects otherwise pay six redundant traversal/serialization
    # costs every time transcript-only invalidation reuses visual state.
    visual_state = json.loads(
        json.dumps(
            {
                "frames": prior_project.get("frames", []),
                "payloads": prior_project.get("evidence_image_metadata", []),
                "revisions": prior_project.get("metadata_revisions", []),
                "events": prior_project.get("visual_events", []),
                "reviews": prior_project.get("review_items", []),
                "ocr_observations": prior_project.get("ocr_observations", []),
            }
        )
    )
    frames = visual_state["frames"]
    payloads = visual_state["payloads"]
    revisions = visual_state["revisions"]
    events = visual_state["events"]
    reviews = visual_state["reviews"]
    ocr_observations = visual_state["ocr_observations"]
    old_blocks = {
        str(item.get("block_id")): item
        for item in prior_project.get("script_blocks", [])
        if isinstance(item, dict)
    }
    fields_to_copy = (
        "visual_description",
        "on_screen_text",
        "frame_ids",
        "visual_event_ids",
        "image_claim_ids",
        "metadata_revision_ids",
        "metadata_sufficiency_decision_ids",
        "transformation_ids",
        "uncertainty",
    )
    for block in blocks:
        previous = old_blocks.get(str(block.get("block_id")))
        if previous is None:
            continue
        # Batch-copy the selected block fields in one JSON pass.  This keeps
        # the returned block independent from the prior canonical object while
        # avoiding one encoder/decoder traversal per field.
        selected_fields = {
            field: previous[field] for field in fields_to_copy if field in previous
        }
        block.update(json.loads(json.dumps(selected_fields)))
    return frames, payloads, revisions, events, reviews, ocr_observations


def _ensure_sufficiency_decisions(
    project_dir: Path,
    *,
    frames: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    revisions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    existing_decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Record a scoped stopping decision for every retained image.

    Creation and deterministic analysis cannot answer semantic questions.  They
    still must leave a machine-readable decision so a later observer knows exactly
    what remains open.  The decision is also mirrored into the image envelope in
    its own append-only revision; this keeps the canonical and portable evidence
    layers synchronized rather than treating the image sidecar as optional.
    """

    from .ids import sequential_id
    from .image_metadata import embed_metadata_with_file_hash, read_embedded_metadata
    from .metadata_reconcile import evaluate_sufficiency
    from .schemas import EvidenceQuestion

    decisions = json.loads(json.dumps(existing_decisions))

    def greatest_id(values: list[str], prefix: str) -> int:
        greatest = 0
        for value in values:
            if not value.startswith(prefix):
                continue
            suffix = value[len(prefix) :]
            if suffix.isdigit():
                greatest = max(greatest, int(suffix))
        return greatest

    decision_ids = [str(item.get("decision_id")) for item in decisions]
    question_ids = [
        str(question.get("question_id"))
        for item in decisions
        for question in item.get("questions", [])
        if isinstance(question, dict)
    ]
    next_decision_number = greatest_id(decision_ids, "MS")
    next_question_number = greatest_id(question_ids, "Q")
    revision_number = max(
        (int(str(item.get("revision_id", "MR0")).removeprefix("MR")) for item in revisions),
        default=0,
    )
    block_by_id = {str(block.get("block_id")): block for block in blocks}
    event_by_id = {str(event.get("event_id")): event for event in events}
    # Review links are queried for every retained frame and again when each
    # independent metadata job commits.  Index them once while preserving the
    # source list order; this avoids an otherwise quadratic frame×review scan
    # on long-form projects without changing review ordering or IDs.
    review_by_id = {
        str(review.get("review_id")): review
        for review in reviews
        if review.get("review_id") is not None
    }
    review_ids_by_frame: dict[str, list[str]] = {}
    for review in reviews:
        review_id = review.get("review_id")
        if review_id is None:
            continue
        review_id_text = str(review_id)
        for frame_id in review.get("frame_ids", []):
            review_ids_by_frame.setdefault(str(frame_id), []).append(review_id_text)
    existing_images = {
        str(image_id) for decision in decisions for image_id in decision.get("image_ids", [])
    }
    payload_history: list[dict[str, Any]] = []
    ledger_path = project_dir / ".state" / "vision" / "image-observations.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        ledger = {"schema_version": "1.0", "payload_history": []}
    if not isinstance(ledger, dict):
        ledger = {"schema_version": "1.0", "payload_history": []}
    ledger.setdefault("payload_history", [])
    sufficiency_jobs: list[dict[str, Any]] = []
    payload_index_by_image_id = {
        str(payload.get("image", {}).get("image_id")): index
        for index, payload in enumerate(payloads)
    }

    for frame in sorted(
        frames, key=lambda item: (int(item.get("actual_ms", 0)), str(item.get("frame_id")))
    ):
        image_id = str(frame.get("frame_id") or frame.get("image_id"))
        if image_id in existing_images:
            continue
        block_ids = [str(value) for value in frame.get("linked_block_ids", [])]
        event_ids = [str(value) for value in frame.get("linked_event_ids", [])]
        linked_events = [event_by_id[event_id] for event_id in event_ids if event_id in event_by_id]
        importance: Literal["incidental", "supporting", "consequential", "high_impact"] = (
            "consequential"
            if any(
                str(event.get("importance")) in {"consequential", "high_impact"}
                for event in linked_events
            )
            else "supporting"
        )
        next_question_number += 1
        question_id = sequential_id("question", next_question_number)
        question = EvidenceQuestion(
            question_id=question_id,
            question="What meaningful visible state or exact text supports the linked reconstruction?",
            importance=importance,
            required_precision="observable visible state and consequential text",
            modality="visual",
            candidate_claim_ids=[],
        )

        image_path = project_dir / str(frame.get("full_frame_path") or frame.get("path"))
        current = read_embedded_metadata(image_path)
        stored_file_hash = frame.get("file_hash")
        # The deterministic stages already recorded a whole-file digest.  A
        # matching digest is a cryptographic precondition for the internal
        # metadata-only fast path; if it is absent or stale, retain the public
        # independent pixel-decode verification below.
        fast_metadata_path = bool(
            isinstance(stored_file_hash, str)
            and stored_file_hash
            and sha256_file(image_path) == stored_file_hash
        )
        revision_number += 1
        revision_id = sequential_id("metadata_revision", revision_number)
        decision = evaluate_sufficiency(
            decision_id=sequential_id("metadata_sufficiency", next_decision_number + 1),
            questions=[question],
            claims=[],
            observations=[],
            image_ids=[image_id],
            metadata_revision_ids=[revision_id],
            visual_event_ids=event_ids,
            script_block_ids=block_ids,
            payload_current_and_valid=True,
            unattempted_evidence_actions=[
                "Inspect the original-resolution full frame, relevant crop, and adjacent before/action/after evidence."
            ],
            semantic_observer_available=False,
            decided_by="deterministic-sufficiency-rule-v1",
            now_utc=_now(),
        )
        next_decision_number += 1
        decision_dict = decision.model_dump(mode="json")

        linked_blocks = [block_by_id[block_id] for block_id in block_ids if block_id in block_by_id]
        segment_ids = sorted(
            {
                str(segment_id)
                for block in linked_blocks
                for segment_id in block.get("transcript_segment_ids", [])
            }
        )
        chapter_ids = sorted(
            {str(block.get("chapter_id")) for block in linked_blocks if block.get("chapter_id")}
        )
        review_ids = list(review_ids_by_frame.get(image_id, []))
        raw = current.model_dump(mode="json")
        payload_history.append(raw)
        raw["links"].update(
            {
                "chapter_ids": chapter_ids,
                "block_ids": block_ids,
                "segment_ids": segment_ids,
                "visual_event_ids": event_ids,
                "review_item_ids": review_ids,
            }
        )
        raw["analysis"]["sufficiency"] = {
            "status": decision.status,
            "evaluated_question_ids": [question_id],
            "answered_question_ids": [],
            "unanswered_questions": [f"{question_id}: {question.question}"],
            "recommended_next_action": decision.recommended_next_action,
        }
        raw["analysis"]["latest_revision_id"] = revision_id
        raw["analysis"]["revision_number"] = current.analysis.revision_number + 1
        raw["analysis"]["semantic_status"] = "deterministic_only"
        raw["integrity"].update(
            {
                "previous_revision_id": current.analysis.latest_revision_id,
                "previous_payload_digest": current.integrity.payload_digest,
                "canonical_revision_locator": f".state/vision/image-observations.json#{revision_id}",
                "canonical_revision_digest": hashlib.sha256(
                    json.dumps(
                        {"revision_id": revision_id, "image_id": image_id, "kind": "sufficiency"},
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            }
        )
        sufficiency_jobs.append(
            {
                "frame": frame,
                "image_id": image_id,
                "image_path": image_path,
                "current": current,
                "revision_id": revision_id,
                "decision": decision,
                "decision_dict": decision_dict,
                "raw": raw,
                "linked_blocks": linked_blocks,
                "linked_events": linked_events,
                "review_ids": review_ids,
                "fast_metadata_path": fast_metadata_path,
            }
        )

    # Each sufficiency revision is independent. Perform the verified PNG
    # rewrite/read-back concurrently, then apply all canonical links in the
    # deterministic sorted frame order used to allocate IDs above. A matching
    # pre-write whole-file hash permits the internal IDAT-preserving fast path;
    # missing/stale hashes deliberately use the public decoded-pixel path.
    with ThreadPoolExecutor(
        max_workers=_visual_frame_workers(), thread_name_prefix="vsr-sufficiency"
    ) as pool:
        futures = [
            pool.submit(
                embed_metadata_with_file_hash,
                job["image_path"],
                job["raw"],
                verify_source_pixels=not bool(job["fast_metadata_path"]),
                verify_decoded_pixels=not bool(job["fast_metadata_path"]),
            )
            for job in sufficiency_jobs
        ]
        for job, future in zip(sufficiency_jobs, futures, strict=True):
            frame = cast(dict[str, Any], job["frame"])
            image_id = str(job["image_id"])
            image_path = cast(Path, job["image_path"])
            current = job["current"]
            revision_id = str(job["revision_id"])
            decision = job["decision"]
            prepared, file_hash = future.result()
            prepared_dict = prepared.model_dump(mode="json")
            payload_index = payload_index_by_image_id.get(image_id)
            if payload_index is None:
                payloads.append(prepared_dict)
                payload_index_by_image_id[image_id] = len(payloads) - 1
            else:
                payloads[payload_index] = prepared_dict
            revisions.append(
                {
                    "revision_id": revision_id,
                    "revision_number": current.analysis.revision_number + 1,
                    "image_id": image_id,
                    "base_revision_id": current.analysis.latest_revision_id,
                    "observation_ids": [],
                    "added_claim_ids": [],
                    "confirmed_claim_ids": [],
                    "narrowed_claim_ids": [],
                    "disputed_claim_ids": [],
                    "rejected_claim_ids": [],
                    "superseded_claim_ids": [],
                    "unresolved_claim_ids": [],
                    "previous_payload_digest": current.integrity.payload_digest,
                    "new_payload_digest": prepared.integrity.payload_digest,
                    "reconciliation_method": "deterministic-sufficiency-decision",
                    "actor": "deterministic-sufficiency-rule-v1",
                    "stale_base_reconciled": False,
                    "pixel_invariance_verified": True,
                    "embedded_write_verified": True,
                    "read_back_verified": True,
                    "canonical_mirror_committed": True,
                    "created_at_utc": _now(),
                }
            )
            frame.update(
                {
                    "latest_revision_id": revision_id,
                    "metadata_payload_digest": prepared.integrity.payload_digest,
                    "metadata_sufficiency_state": decision.status,
                    "file_hash": file_hash,
                    "metadata": prepared_dict,
                }
            )
            for block in job["linked_blocks"]:
                block.setdefault("metadata_sufficiency_decision_ids", []).append(
                    decision.decision_id
                )
                if revision_id not in block.setdefault("metadata_revision_ids", []):
                    block["metadata_revision_ids"].append(revision_id)
            for event in job["linked_events"]:
                if revision_id not in event.setdefault("metadata_revision_ids", []):
                    event["metadata_revision_ids"].append(revision_id)
            for review_id in job["review_ids"]:
                review_record = review_by_id.get(str(review_id))
                if review_record is None:
                    continue
                review_record.setdefault("sufficiency_decision_ids", []).append(
                    decision.decision_id
                )
                if revision_id not in review_record.setdefault("metadata_revision_ids", []):
                    review_record["metadata_revision_ids"].append(revision_id)
            decisions.append(cast(dict[str, Any], job["decision_dict"]))
            existing_images.add(image_id)

    ledger["payload_history"].extend(payload_history)
    ledger["payloads"] = payloads
    ledger["revisions"] = revisions
    ledger["claims"] = ledger.get("claims", [])
    ledger["observations"] = ledger.get("observations", [])
    atomic_write_json(ledger_path, ledger, compact=True)
    return cast(list[dict[str, Any]], decisions)


def run_pipeline(
    input_value: str | Path,
    *,
    output_root: Path | None = None,
    subtitles: Sequence[Path] = (),
    transcript: Path | None = None,
    preset: str = "strict",
    config_path: Path | None = None,
    subtitle_mode: str = "auto",
    language: str | None = None,
    fidelity_mode: str = "verbatim",
    vision_mode: str = "host-agent",
    asr_chunk_seconds: int | None = None,
    asr_overlap_seconds: int | None = None,
    semantic_max_packets: int | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    resume: bool = True,
    offline: bool = True,
    allow_remote_download: bool = False,
    allow_external_ai: bool = False,
    asr_adapter: ASRAdapter | None = None,
    ocr_adapter: OCRAdapter | None = None,
    vision_provider: VisionProvider | None = None,
) -> RunResult:
    kind = classify_input(input_value)
    # Preserve the public ``auto`` spelling for older callers without allowing
    # it to silently invoke the legacy local Qwen/VLM route.
    if vision_mode == "auto":
        vision_mode = "host-agent"
    if kind == "remote_media":
        if not allow_remote_download or offline:
            raise InputError(
                "Remote input requires explicit --allow-remote-download and non-offline configuration"
            )
        raise InputError(
            "Direct URL input is conditional and no tested downloader adapter is installed"
        )
    source = Path(input_value).expanduser().resolve(strict=True)
    project_dir, colocated = _single_output_dir(source, output_root)
    project_dir.parent.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    _prepare_tree(project_dir)
    markdown_path = project_dir / _markdown_filename(source, colocated=colocated)
    overrides: dict[str, Any] = {
        "script": {"fidelity_mode": fidelity_mode},
        "privacy": {
            "offline": offline,
            "allow_remote_download": allow_remote_download,
            "allow_external_ai": allow_external_ai,
        },
    }
    asr_overrides: dict[str, int] = {}
    if asr_chunk_seconds is not None:
        asr_overrides["chunk_seconds"] = asr_chunk_seconds
    if asr_overlap_seconds is not None:
        asr_overrides["overlap_seconds"] = asr_overlap_seconds
    if asr_overrides:
        overrides["asr"] = asr_overrides
    # Hashing a long source is independent of configuration/model/tool probes.
    # Run it concurrently with those deterministic setup checks while retaining
    # the same immutable digest as the cache-key and media-identity authority.
    source_hash_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="vsr-source-hash"
    )
    source_hash_future = source_hash_executor.submit(sha256_file, source)
    try:
        config = load_config(config_path or preset, overrides)
        # OCR is a visual-only stage. Defer optional OCR imports/model resolution
        # for transcript and audio inputs so lightweight runs never pay for a
        # capability they cannot execute.
        if ocr_adapter is None and kind == "video" and config.visual.ocr:
            ocr_adapter = cast(Any, _auto_ocr_adapter())
        cfg_hash = config_digest(config)
        visual_reuse_config_hash = _visual_reuse_config_digest(config)
        sidecar_hashes = [sha256_file(Path(path).resolve(strict=True)) for path in subtitles]
        if transcript:
            sidecar_hashes.append(sha256_file(transcript.resolve(strict=True)))
        # Compute the immutable source digest once and thread it through
        # cache-key, manifest, media-identity, and checkpoint stages. Reusing
        # this completed future avoids re-hashing at each later boundary.
        source_sha256 = source_hash_future.result()
    finally:
        source_hash_executor.shutdown(wait=True)

    def adapter_key(adapter: Any, fallback: str) -> str:
        if adapter is None:
            return fallback
        base = f"{adapter.__class__.__module__}.{adapter.__class__.__qualname__}"
        identity = getattr(adapter, "cache_identity", None)
        if identity is None:
            identity = getattr(adapter, "backend_name", None)
        if identity is None:
            identity = getattr(adapter, "model_name_or_path", None)
        descriptor = getattr(adapter, "descriptor", None)
        if descriptor is not None:
            identity = "|".join(
                str(value)
                for value in (
                    getattr(descriptor, "provider_id", None),
                    getattr(descriptor, "route", None),
                    getattr(descriptor, "model", None),
                    getattr(descriptor, "model_version", None),
                )
                if value is not None
            )
        version = getattr(adapter, "adapter_version", None)
        suffix = [str(value) for value in (identity, version) if value is not None]
        return base if not suffix else base + "|" + "|".join(suffix)

    def managed_revision(name: str) -> str:
        from .model_store import MANIFEST_NAME, model_directory

        if name == "faster-whisper-large-v3":
            external = _configured_faster_whisper_model()
            if external is not None:
                try:
                    signature = "|".join(
                        f"{filename}:{(external / filename).stat().st_size}:"
                        f"{(external / filename).stat().st_mtime_ns}"
                        for filename in _FASTER_WHISPER_REQUIRED_FILES
                    )
                except OSError:
                    signature = "unstatable"
                return f"external:{external}:{signature}"
        try:
            payload = json.loads(
                (model_directory(name) / MANIFEST_NAME).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return "unavailable"
        return str(payload.get("revision") or "unavailable")

    language_hint_policy = os.environ.get("VSR_ASR_LANGUAGE_HINT", "").strip().lower()
    language_hint_mode = "on" if language_hint_policy in {"1", "true", "yes", "on"} else "off"
    if kind in {"video", "audio"}:
        automatic_asr_identity = (
            "auto-whisper-large-v3:"
            + managed_revision("faster-whisper-large-v3")
        )
        if str(os.environ.get("VSR_ALLOW_LEGACY_LOCAL_MODELS", "")).casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            automatic_asr_identity += ":legacy=" + ":".join(
                (
                    f"qwen={managed_revision('qwen3-asr-1.7b')}",
                    f"aligner={managed_revision('qwen3-forced-aligner-0.6b')}",
                    f"moss={managed_revision('moss-transcribe-diarize-0.9b')}",
                )
            )
    else:
        # Transcript-only runs cannot instantiate an automatic speech adapter;
        # avoid reading four optional model manifests just to build an unused
        # cache-key component.
        automatic_asr_identity = "auto-local-accuracy:unused"
    if language_hint_mode == "on" and kind in {"video", "audio"}:
        automatic_asr_identity = "auto-local-accuracy:language-hint-on:" + automatic_asr_identity.removeprefix(
            "auto-local-accuracy:"
        )
    asr_adapter_key = adapter_key(asr_adapter, automatic_asr_identity)
    ocr_adapter_key = adapter_key(ocr_adapter, "auto-local-ocr")
    managed_vision_revision: str | None = None
    # Semantic vision is also video-only. Keep non-video cache keys free of
    # unrelated vision-manifest reads and avoid touching the optional model
    # store during transcript/audio reconstruction.
    if kind == "video" and vision_provider is None and vision_mode == "local":
        from .model_store import MANIFEST_NAME, model_directory

        manifest_path = model_directory("qwen3-vl-4b-q4") / MANIFEST_NAME
        try:
            model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            managed_vision_revision = str(model_manifest.get("revision") or "") or None
        except (OSError, json.JSONDecodeError):
            managed_vision_revision = None
    default_vision_identity = (
        f"{vision_mode}:qwen3-vl-4b-q4:{managed_vision_revision or 'unavailable'}"
        if vision_mode == "local"
        else f"{vision_mode}:codex-subagent:offline"
    )
    vision_adapter_key = adapter_key(vision_provider, default_vision_identity)
    tool_inputs = _tool_versions_for_cache_key()
    run_key = cache_key(
        source_sha256,
        sidecar_hashes,
        cfg_hash,
        subtitle_mode,
        language,
        fidelity_mode,
        vision_mode,
        asr_adapter_key,
        ocr_adapter_key,
        vision_adapter_key,
        tool_inputs,
        __version__,
    )
    existing_manifest = project_dir / ".state" / "run-manifest.json"
    existing_canonical = project_dir / ".state" / "canonical-project.json"
    old_manifest: dict[str, Any] | None = None
    old_project: dict[str, Any] | None = None

    def _refresh_blocked_semantic_cache(
        status: str, validation: ValidationResult
    ) -> tuple[str, ValidationResult]:
        """Repair stale semantic block links before returning a cache hit.

        Semantic ingestion can finish its packet ledger while an incremental
        evidence update leaves a block description as a placeholder.  A
        normal unchanged-run cache hit must not hide that stale audit state,
        but it also must not start the local VLM again.  Reconcile only blocked
        video projects with no pending packets; all other cache hits retain the
        existing O(1) receipt path.
        """

        if status != "blocked" or kind != "video" or vision_mode not in {
            "local",
            "external",
        }:
            return status, validation
        try:
            from .semantic_pipeline import pending_packet_count, refresh_semantic_state

            if pending_packet_count(project_dir):
                return status, validation
            refresh_semantic_state(project_dir)
            refreshed_project = json.loads(
                existing_canonical.read_text(encoding="utf-8")
            )
            refreshed_validation = validate_project(
                project_dir, use_cached_file_hash=True
            )
            refreshed_status = str(refreshed_project.get("project_status", status))
            return refreshed_status, refreshed_validation
        except (OSError, TypeError, ValueError, ValidationFailure):
            # Preserve the original cache result if reconciliation cannot be
            # proven safe; the next explicit resume still has the full state.
            return status, validation

    if resume and existing_manifest.is_file() and existing_canonical.is_file():
        try:
            old_manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt/incomplete prior run is not a cache hit.  Keep the
            # partially written files for diagnosis and rebuild the affected
            # state atomically below.
            old_manifest = None
        else:
            # A current receipt binds the final status and all generated-file
            # invariants, so an unchanged resume can return without parsing the
            # potentially multi-megabyte canonical project.  A missing/legacy
            # status falls through to the complete migration path below.
            if old_manifest.get("run_cache_key") == run_key:
                receipt_validation = read_trusted_validation_receipt(
                    project_dir,
                    None,
                    run_cache_key=run_key,
                )
                if receipt_validation is not None and receipt_validation.project_status:
                    status = receipt_validation.project_status
                    status, receipt_validation = _refresh_blocked_semantic_cache(
                        status, receipt_validation
                    )
                    return RunResult(
                        project_dir,
                        markdown_path,
                        status,
                        0
                        if status in {"automatically_checked", "human_reviewed", "fully_verified"}
                        else (3 if status == "review_required" else 4),
                        receipt_validation,
                    )
            try:
                old_project = json.loads(existing_canonical.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                old_project = None
        if old_manifest is not None and old_project is not None:
            # Older projects may predate the mandatory per-image sufficiency
            # ledger.  Re-enter the pipeline once to migrate those artifacts
            # instead of returning a stale cache hit.
            cache_contract_complete = not old_project.get("frames") or bool(
                old_project.get("sufficiency_decisions")
            )
            # New runs publish a processing/finalizing marker before the
            # canonical transaction.  Never treat a matching partial tree as
            # a cache hit until the marker says finalization completed.  A
            # missing marker is retained only for one-time legacy migration.
            finalization_complete = old_manifest.get("run_state") in {
                None,
                "completed",
            }
            if (
                old_manifest.get("run_cache_key") == run_key
                and cache_contract_complete
                and finalization_complete
            ):
                validation = read_trusted_validation_receipt(
                    project_dir,
                    old_project,
                    run_cache_key=run_key,
                )
                receipt_trusted = validation is not None
                if validation is None:
                    render_to_path(old_project, markdown_path)
                    validation = validate_project(project_dir, use_cached_file_hash=True)
                status = old_project.get("project_status", "blocked")
                if validation.valid:
                    status, validation = _refresh_blocked_semantic_cache(status, validation)
                    if status != "blocked":
                        if (
                            not receipt_trusted
                            and isinstance(old_project.get("manifest"), dict)
                        ):
                            # Upgrade a pre-receipt project after its ordinary
                            # cache-hit proof.  The receipt is added before a
                            # telemetry refresh so output byte/file parity stays
                            # authoritative for the migrated project as well.
                            write_validation_receipt(
                                project_dir,
                                old_project,
                                run_cache_key=run_key,
                                validation=validation,
                            )
                            old_manifest_payload = old_project["manifest"]
                            _publish_resource_telemetry(
                                project_dir,
                                old_project,
                                old_manifest_payload,
                            )
                            refresh_validation_receipt_signature(project_dir)
                            if (
                                read_trusted_validation_receipt(
                                    project_dir,
                                    old_project,
                                    run_cache_key=run_key,
                                )
                                is None
                            ):
                                (project_dir / ".state" / "validation-receipt.json").unlink(
                                    missing_ok=True
                                )
                        return RunResult(
                            project_dir,
                            markdown_path,
                            status,
                            0
                            if status in {"automatically_checked", "human_reviewed", "fully_verified"}
                            else (3 if status == "review_required" else 4),
                            validation,
                        )
                    # A blocked project has no trusted receipt by design. Do
                    # not treat a valid stale snapshot as a cache hit: a
                    # retryable prerequisite may now be available, and resume
                    # must re-enter the stages while preserving checkpoints.

    manifest = ManifestBuilder(
        run_id="RUN" + hashlib.sha256((run_key + _now()).encode()).hexdigest()[:12].upper(),
        input_identity={"path": str(source), "sha256": source_sha256, "kind": kind},
        config_hash=cfg_hash,
    )
    manifest.performance["scheduling"] = _scheduler_snapshot()
    manifest.performance["stage_config_hashes"] = {
        "visual": visual_reuse_config_hash,
    }
    run_manifest_path = project_dir / ".state" / "run-manifest.json"
    asr_progress_path = project_dir / ".state" / "asr-progress.json"
    manifest.start("identity")
    media, probe = _media_identity(source, kind, content_hash=source_sha256)
    manifest.finish("identity", "completed")
    measured_duration_ms = media.get("duration_ms")
    duration_ms_value = (
        int(measured_duration_ms) if isinstance(measured_duration_ms, int) else None
    )
    # The media duration is not available when the manifest builder is created.
    # Refresh scheduling telemetry now so the auto-overlap decision is visible
    # before transcript work begins.
    manifest.performance["scheduling"] = _scheduler_snapshot(duration_ms=duration_ms_value)

    transcript_candidates: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    repair_records: list[dict[str, Any]] = []
    transcript_disagreements: list[dict[str, Any]] = []
    transcript_timing_adjustments: list[dict[str, Any]] = []
    source_decision = ""
    blockers: list[str] = []
    asr_completed_without_segments = False
    asr_produced_segments = False
    asr_had_failure = False
    visual_only_no_speech_fallback = False
    precomputed_visual_survey: _PrecomputedVisualSurvey | None = None
    parallel_survey_future: Future[_PrecomputedVisualSurvey] | None = None
    parallel_survey_executor: ThreadPoolExecutor | None = None
    parallel_survey_started_at: float | None = None
    manifest.start("transcript")
    manifest.write(run_manifest_path)

    def persist_asr_progress(payload: Mapping[str, Any]) -> None:
        """Write resumable ASR telemetry without making it part of evidence."""

        event = dict(payload)
        event.setdefault("stage", "transcript")
        event.setdefault("updated_at_utc", _now())
        if event.get("event") == "chunk_heartbeat":
            # Heartbeats may arrive from the decoder-observer thread while a
            # native call is blocked. Keep them out of the mutable manifest so
            # they cannot race the parallel visual-survey writer; the atomic
            # progress receipt is the durable stall-observation surface.
            atomic_write_json(
                asr_progress_path,
                {"schema_version": "1.0", **_bounded_asr_progress_event(event)},
            )
            if progress_callback is not None:
                try:
                    progress_callback(event)
                except Exception:  # pragma: no cover - optional UI telemetry boundary
                    LOGGER.warning("ASR progress callback failed", exc_info=True)
            return
        persisted_event = _bounded_asr_progress_event(event)
        manifest.update_performance("asr", persisted_event)
        if ".state/asr-progress.json" not in manifest.artifacts:
            manifest.artifacts.append(".state/asr-progress.json")
        atomic_write_json(
            asr_progress_path,
            {"schema_version": "1.0", **persisted_event},
        )
        # A chunk-start event is already durable through the prior checkpoint;
        # writing the full manifest for it only duplicates work.  Completed
        # chunks still refresh the manifest, so an interruption never loses a
        # valid checkpoint's progress accounting.
        if event.get("event") != "chunk_started":
            manifest.write(run_manifest_path)
        if progress_callback is not None:
            try:
                progress_callback(event)
            except Exception:  # pragma: no cover - optional UI telemetry boundary
                LOGGER.warning("ASR progress callback failed", exc_info=True)

    def persist_visual_progress(payload: Mapping[str, Any]) -> None:
        """Persist bounded visual substage telemetry in the run manifest."""

        event = dict(payload)
        event.setdefault("stage", "visual_evidence")
        event.setdefault("updated_at_utc", _now())
        visual_history = manifest.performance.setdefault("visual_events", [])
        if isinstance(visual_history, list):
            visual_history.append(event)
            # Keep telemetry bounded even if a provider emits additional
            # progress events in a future implementation.
            del visual_history[:-32]
        manifest.update_performance("visual", event)
        manifest.write(run_manifest_path)

    candidate_paths: list[tuple[Path, str] | _CandidatePath] = []
    if kind == "transcript":
        candidate_paths.append(
            (
                source,
                "user_human_transcript"
                if source.suffix.casefold() in {".json", ".txt"}
                else "user_subtitle",
            )
        )
    if transcript:
        candidate_paths.append((transcript.resolve(strict=True), "user_human_transcript"))
    candidate_paths.extend((Path(path).resolve(strict=True), "user_subtitle") for path in subtitles)
    if subtitle_mode == "force-asr":
        candidate_paths = []
    try:
        embedded_issues: list[str] = []
        if (
            kind in {"video", "audio"}
            and probe is not None
            and subtitle_mode in {"auto", "compare-all"}
        ):
            from .subtitle_sources import (
                EmbeddedSubtitleError,
                discover_embedded_subtitle_tracks,
                extract_embedded_subtitle_track,
            )

            embedded_tracks = discover_embedded_subtitle_tracks(source, probe=probe)
            for track in embedded_tracks:
                if not track.supported:
                    embedded_issues.append(
                        f"stream {track.stream_index}: {track.unsupported_reason}"
                    )
                    continue
                try:
                    extracted = extract_embedded_subtitle_track(
                        source,
                        track,
                        project_dir / ".state" / "candidates",
                    )
                except EmbeddedSubtitleError as exc:
                    embedded_issues.append(f"stream {track.stream_index}: {exc}")
                    continue
                active_dispositions = sorted(
                    key for key, value in track.disposition.items() if value
                )
                source_track = (
                    f"embedded stream {track.stream_index}; codec={track.codec_name}; "
                    f"language={track.language or 'und'}; disposition="
                    f"{','.join(active_dispositions) or 'none'}"
                )
                candidate_paths.append(
                    _CandidatePath(
                        path=extracted.path,
                        source_type=track.source_type,
                        origin=f"{source.name}#stream={track.stream_index}",
                        language=track.language,
                        authorship=track.authorship,
                        source_track=source_track,
                    )
                )
        transcript_candidates, segments, source_decision = _parse_candidates(
            candidate_paths,
            project_dir,
            media_duration_ms=media.get("duration_ms"),
            expected_language=language,
        )
        if embedded_issues:
            source_decision += " Embedded subtitle diagnostics: " + "; ".join(embedded_issues)
        selected_candidate = next(
            (
                item
                for item in transcript_candidates
                if item.get("decision_rationale", "").startswith("Selected as")
            ),
            None,
        )
        segments, transcript_timing_adjustments = _clip_source_transcript_bounds_to_media(
            segments, media.get("duration_ms")
        )
        if transcript_timing_adjustments:
            if selected_candidate is not None:
                selected_candidate["segments"] = segments
                metrics = selected_candidate.setdefault("quality_metrics", {})
                if isinstance(metrics, dict):
                    metrics["timing_adjustment_count"] = len(transcript_timing_adjustments)
            source_decision += (
                " Clipped "
                f"{len(transcript_timing_adjustments)} subtitle cue end(s) to the measured "
                "media duration; original source timing remains available for review."
            )
        if (
            segments
            and selected_candidate is not None
            and selected_candidate.get("unreliable_intervals")
            and asr_adapter is not None
            and config.transcript.selective_repair
            and kind in {"video", "audio"}
        ):
            from .subtitle_validate import validate_segments
            from .transcript_repair import repair_suspect_intervals

            suspect_intervals = [
                (int(interval["start_ms"]), int(interval["end_ms"]))
                for interval in selected_candidate["unreliable_intervals"]
                if interval.get("start_ms") is not None
                and interval.get("end_ms") is not None
                and interval["end_ms"] > interval["start_ms"]
            ]
            if suspect_intervals:
                repair_outcome = repair_suspect_intervals(
                    source,
                    segments,
                    suspect_intervals,
                    cast(Any, asr_adapter),
                    media_duration_ms=media.get("duration_ms"),
                    language=language,
                    work_dir=project_dir / ".state" / "checkpoints" / "repair-audio",
                )
                segments = [_record(segment) for segment in repair_outcome.segments]
                repair_records = [_record(record) for record in repair_outcome.records]
                repaired_report = validate_segments(
                    segments, media_duration_ms=media.get("duration_ms")
                )
                selected_candidate["quality_metrics"]["post_repair_quality_score"] = (
                    repaired_report.quality_score
                )
                source_decision += (
                    f" Selectively repaired {len(repair_records)} segment interval(s); "
                    f"post-repair quality={repaired_report.quality_score:.3f}."
                )
        # ``compare-all`` deliberately runs Whisper even when a supplied or
        # embedded subtitle candidate parsed successfully.  The candidates
        # are then ranked together by ``_select_asr_candidate_set`` so the
        # selected transcript remains evidence-driven while preserving an
        # independent ASR corroboration/disagreement record.  ``auto`` keeps
        # the historical fast path and trusts a usable supplied candidate;
        # ``provided-only`` never invokes ASR.
        run_asr = (
            kind in {"video", "audio"}
            and subtitle_mode != "provided-only"
            and (not segments or subtitle_mode == "compare-all")
        )
        if run_asr:
            adapters = (
                [asr_adapter]
                if asr_adapter is not None
                else _auto_asr_adapters(
                    language=language,
                    compare_candidates=config.transcript.compare_candidates,
                    duration_ms=duration_ms_value,
                )
            )
            if not adapters:
                raise ValueError(
                    "No installed hash-verified faster-whisper large-v3 ASR worker is available; "
                    "install it explicitly or inject an ASR adapter"
                )
            if (
                kind == "video"
                and probe is not None
                and probe.video_streams
                and _parallel_visual_survey_enabled(
                    duration_ms=duration_ms_value,
                    automatic_adapter=asr_adapter is None,
                )
                and not (old_project is not None and old_project.get("frames"))
            ):
                parallel_survey_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="vsr-parallel-visual-survey",
                )
                parallel_survey_started_at = time.perf_counter()
                parallel_survey_future = parallel_survey_executor.submit(
                    _precompute_visual_survey,
                    source,
                    project_dir,
                    duration_ms=int(media.get("duration_ms") or 1),
                    interval_seconds=float(config.visual.survey_interval_seconds),
                    strict=config.preset == "strict",
                    scene_detection=config.visual.scene_detection,
                    adaptive_detection=config.visual.frame_difference,
                    source_sha256=source_sha256,
                    prefetch_exact_frames=_parallel_visual_warmup_enabled(
                        duration_ms=duration_ms_value,
                        automatic_adapter=asr_adapter is None,
                    ),
                )
                persist_visual_progress(
                    {
                        "event": "survey_parallel_started",
                        "reason": "overlapped source survey with local ASR",
                        "exact_frame_warmup": _parallel_visual_warmup_enabled(
                            duration_ms=duration_ms_value,
                            automatic_adapter=asr_adapter is None,
                        ),
                        "elapsed_seconds": 0.0,
                    }
                )
            asr_failures: list[str] = []
            for adapter in adapters:
                backend = str(getattr(adapter, "backend_name", adapter.__class__.__name__))
                try:
                    candidate_segments = _asr_segments(
                        cast(ASRAdapter, adapter),
                        source,
                        language,
                        duration_ms=media.get("duration_ms"),
                        checkpoint_dir=(
                            project_dir / ".state" / "checkpoints" / "asr" / safe_slug(backend)
                        ),
                        chunk_ms=int(config.asr.chunk_seconds * 1000),
                        overlap_ms=int(config.asr.overlap_seconds * 1000),
                        media_sha256=source_sha256,
                        shared_cache_dir=(
                            _asr_shared_cache_dir() if asr_adapter is None else None
                        ),
                        progress_callback=persist_asr_progress,
                    )
                    if not candidate_segments:
                        asr_completed_without_segments = True
                    else:
                        asr_produced_segments = True
                    runtime_settings = {
                        key: getattr(adapter, key)
                        for key in ("device", "compute_type", "cpu_threads", "num_workers")
                        if getattr(adapter, key, None) is not None
                    }
                    if runtime_settings:
                        runtime_by_backend = manifest.performance.setdefault(
                            "asr_runtime", {}
                        )
                        if isinstance(runtime_by_backend, dict):
                            runtime_by_backend[backend] = runtime_settings
                        manifest.write(run_manifest_path)
                    _record_asr_candidate(
                        transcript_candidates,
                        candidate_segments,
                        cast(ASRAdapter, adapter),
                        language=language,
                        media_duration_ms=media.get("duration_ms"),
                    )
                    # Mixed Filipino/English recordings are often auto-labeled
                    # as English when the opening seconds contain a short
                    # courtesy phrase or silence. If the first large-v3 pass
                    # is rejected and its segment labels clearly lean
                    # Filipino, retry once with Whisper's canonical ``fil``
                    # hint. The retry is automatic-backend-only and gets its
                    # own language-keyed checkpoints, so it is resumable and
                    # never overwrites the original evidence candidate.
                    if (
                        asr_adapter is None
                        and backend == "faster-whisper"
                        and language is None
                        and not transcript_candidates[-1]
                        .get("quality_metrics", {})
                        .get("usable")
                        and _infer_primary_language(candidate_segments) == "tl"
                    ):
                        try:
                            retry_language = "fil"
                            retry_segments = _asr_segments(
                                cast(ASRAdapter, adapter),
                                source,
                                retry_language,
                                duration_ms=media.get("duration_ms"),
                                checkpoint_dir=(
                                    project_dir
                                    / ".state"
                                    / "checkpoints"
                                    / "asr"
                                    / safe_slug(backend)
                                ),
                                chunk_ms=int(config.asr.chunk_seconds * 1000),
                                overlap_ms=int(config.asr.overlap_seconds * 1000),
                                media_sha256=source_sha256,
                                shared_cache_dir=_asr_shared_cache_dir(),
                                progress_callback=persist_asr_progress,
                            )
                            _record_asr_candidate(
                                transcript_candidates,
                                retry_segments,
                                cast(ASRAdapter, adapter),
                                language=retry_language,
                                media_duration_ms=media.get("duration_ms"),
                            )
                            source_decision += (
                                " Retried faster-whisper with Filipino language hint "
                                "after the auto-detected candidate failed validation."
                            )
                        except Exception as retry_exc:
                            asr_failures.append(
                                f"{backend} Filipino-language retry: {retry_exc}"
                            )
                    # ``compare_candidates=false`` is an explicit performance
                    # choice: stop after the first usable local candidate while
                    # retaining a fallback if that candidate fails validation.
                    if (
                        not config.transcript.compare_candidates
                        and transcript_candidates[-1].get("quality_metrics", {}).get("usable")
                    ):
                        break
                except Exception as exc:
                    asr_had_failure = True
                    asr_failures.append(f"{backend}: {exc}")
                    if asr_adapter is not None:
                        raise
                finally:
                    if asr_adapter is None:
                        _release_asr_adapter(adapter)
            if not transcript_candidates:
                raise ValueError("; ".join(asr_failures) or "No ASR candidate was produced")
            segments, source_decision, transcript_disagreements = _select_asr_candidate_set(
                transcript_candidates,
                media_duration_ms=media.get("duration_ms"),
                expected_language=language,
                preferred_backend=(
                    "faster-whisper"
                    if (
                        str(os.environ.get("VSR_PREFER_WHISPER", "")).casefold()
                        in {"1", "true", "yes"}
                        or (
                            language is not None
                            and language.casefold().split("-")[0] in {"fil", "tl"}
                        )
                    )
                    else None
                ),
                prefer_local_asr=subtitle_mode == "compare-all",
            )
            if language is None:
                inferred_language = _infer_primary_language(segments)
                if inferred_language is not None:
                    language = inferred_language
                    source_decision += f" Inferred primary language={inferred_language} from dominant segment labels."
            for candidate in transcript_candidates:
                if not candidate.get("language"):
                    candidate_language = _infer_primary_language(candidate.get("segments", []))
                    if candidate_language is not None:
                        candidate["language"] = candidate_language
            if asr_adapter is not None and not bool(getattr(asr_adapter, "is_production", False)):
                source_decision += (
                    " Used an explicitly injected model-independent ASR adapter; this does "
                    "not prove production-model accuracy."
                )
            if asr_failures:
                source_decision += " Backend degradations: " + "; ".join(asr_failures)
    except Exception as exc:
        if _can_use_visual_only_fallback(
            kind=kind,
            segments=segments,
            asr_completed_without_segments=asr_completed_without_segments,
            asr_produced_segments=asr_produced_segments,
            asr_had_failure=asr_had_failure,
        ):
            visual_only_no_speech_fallback = True
            source_decision = (
                "No recoverable spoken segments were produced by the completed local ASR "
                "pass. Continuing in explicit visual-only mode; no dialogue is inferred."
            )
        else:
            blockers.append(f"Transcript stage blocked: {exc}")
            source_decision = f"No safe transcript could be selected: {exc}"
    except BaseException as exc:
        # KeyboardInterrupt/SystemExit must not leave a durable manifest saying
        # that transcript work is still running.  The partial checkpoints and
        # progress receipt remain intentionally intact for the next explicit
        # resume, while the stage record makes the interrupted boundary clear
        # to a human operator and to diagnostics.
        detail = f"Interrupted during transcript stage: {type(exc).__name__}"
        if str(exc):
            detail += f": {exc}"
        manifest.finish("transcript", "failed", detail)
        try:
            manifest.write(run_manifest_path)
        except Exception:  # pragma: no cover - defensive persistence boundary
            LOGGER.warning("Unable to persist interrupted transcript stage", exc_info=True)
        raise
    finally:
        if parallel_survey_future is not None:
            try:
                precomputed_visual_survey = parallel_survey_future.result()
                persist_visual_progress(
                    {
                        "event": "survey_parallel_completed",
                        "shared_frame_count": len(precomputed_visual_survey.shared_frames),
                        "candidate_count": len(precomputed_visual_survey.candidates),
                        "prefetched_frame_count": precomputed_visual_survey.prefetched_frame_count,
                        "prefetched_batch_count": precomputed_visual_survey.prefetched_batch_count,
                        "prefetch_failed_batch_count": precomputed_visual_survey.prefetch_failed_batch_count,
                        "prefetch_elapsed_seconds": precomputed_visual_survey.prefetch_elapsed_seconds,
                        "elapsed_seconds": round(
                            time.perf_counter() - (parallel_survey_started_at or time.perf_counter()),
                            6,
                        ),
                    }
                )
            except Exception as exc:  # pragma: no cover - backend-specific fallback
                LOGGER.warning("Parallel visual survey unavailable; using sequential fallback", exc_info=True)
                persist_visual_progress(
                    {
                        "event": "survey_parallel_failed",
                        "error": str(exc),
                        "elapsed_seconds": round(
                            time.perf_counter() - (parallel_survey_started_at or time.perf_counter()),
                            6,
                        ),
                    }
                )
                precomputed_visual_survey = None
            finally:
                if parallel_survey_executor is not None:
                    parallel_survey_executor.shutdown(wait=True)
                parallel_survey_future = None
                parallel_survey_executor = None
                parallel_survey_started_at = None
    if segments:
        from .diarization import apply_explicit_identity_evidence

        segments, identity_claims = apply_explicit_identity_evidence(segments)
        if identity_claims:
            selected_candidate = next(
                (
                    item
                    for item in transcript_candidates
                    if str(item.get("decision_rationale", "")).startswith("Selected as")
                ),
                None,
            )
            if selected_candidate is not None:
                selected_candidate["segments"] = segments
            identity_summary = "; ".join(
                f"{claim.name} ({claim.segment_id or 'untimed'})" for claim in identity_claims
            )
            source_decision += (
                " Explicit spoken self-identification evidence was retained and applied "
                f"only to exact transcript segment(s): {identity_summary}."
            )
    if kind in {"video", "audio"} and not segments and not visual_only_no_speech_fallback:
        blockers.append(
            "Complete spoken reconstruction requires a usable transcript candidate or local large-v3 ASR evidence."
        )
    manifest.finish(
        "transcript", "blocked" if blockers else "completed", "; ".join(blockers) or None
    )

    blocks = _build_blocks(segments, fidelity_mode)
    chapters = _derive_chapters(blocks, media.get("duration_ms"))
    reuse_visual_state = bool(
        resume
        and kind == "video"
        and probe is not None
        and old_manifest is not None
        and old_project is not None
        and isinstance(old_manifest.get("performance"), Mapping)
        and isinstance(old_manifest["performance"].get("stage_config_hashes"), Mapping)
        and old_manifest["performance"]["stage_config_hashes"].get("visual")
        == visual_reuse_config_hash
        and old_manifest.get("input_identity", {}).get("sha256") == media.get("content_hash")
        and old_project.get("frames")
        and _visual_reuse_transcript_compatible(old_project, blocks)
    )
    if reuse_visual_state and old_manifest is not None:
        old_reproducibility = old_manifest.get("reproducibility", {})
        old_adapters = old_reproducibility.get("adapter_keys", {})
        old_tools = old_reproducibility.get("tool_versions", old_manifest.get("tool_versions", {}))
        reuse_visual_state = bool(
            old_manifest.get("code_version") == __version__
            and old_adapters.get("ocr") == ocr_adapter_key
            and old_manifest.get("model_versions", {}).get("vision_mode") == vision_mode
            and old_tools
            == {name: value for name, value in tool_inputs.items() if value is not None}
        )
    frames: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    revisions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    ocr_observations: list[dict[str, Any]] = []
    visual_observations: list[dict[str, Any]] = []
    image_claims: list[dict[str, Any]] = []
    sufficiency_decisions: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    final_signoffs: list[dict[str, Any]] = []
    if kind == "video" and probe and probe.video_streams:
        manifest.start("visual_evidence")
        try:
            if reuse_visual_state and old_project is not None:
                (
                    frames,
                    payloads,
                    revisions,
                    events,
                    reviews,
                    ocr_observations,
                ) = _reuse_visual_state(old_project, blocks)
                # Keep the additional append-only visual collections detached
                # from the prior canonical object, but clone them in one JSON
                # pass to avoid another five independent traversals on every
                # transcript-only visual-state reuse.
                reused_append_only_state = json.loads(
                    json.dumps(
                        {
                            "visual_observations": old_project.get("visual_observations", []),
                            "image_claims": old_project.get("image_claims", []),
                            "sufficiency_decisions": old_project.get(
                                "sufficiency_decisions", []
                            ),
                            "corrections": old_project.get("corrections", []),
                            "final_signoffs": old_project.get("final_signoffs", []),
                        }
                    )
                )
                visual_observations = reused_append_only_state["visual_observations"]
                image_claims = reused_append_only_state["image_claims"]
                sufficiency_decisions = reused_append_only_state["sufficiency_decisions"]
                corrections = reused_append_only_state["corrections"]
                final_signoffs = reused_append_only_state["final_signoffs"]
                manifest.finish(
                    "visual_evidence",
                    "completed",
                    "Reused source-pixel visual state; transcript-only invalidation.",
                )
            else:
                _rotate_incomplete_visual_state(project_dir)
                visual_pool = ThreadPoolExecutor(
                    max_workers=_visual_frame_workers(),
                    thread_name_prefix="vsr-visual-stage",
                )
                try:
                    visual_result = _extract_visual_evidence(
                        source,
                        project_dir,
                        str(media["media_id"]),
                        int(media.get("duration_ms") or 1),
                        blocks,
                        survey_interval_seconds=float(config.visual.survey_interval_seconds),
                        strict=config.preset == "strict",
                        ocr_adapter=ocr_adapter,
                        ocr_enabled=config.visual.ocr,
                        scene_detection_enabled=config.visual.scene_detection,
                        frame_difference_enabled=config.visual.frame_difference,
                        protect_small_changes=config.visual.protect_small_changes,
                        deduplicate=config.visual.deduplicate,
                        progress_callback=persist_visual_progress,
                        source_sha256=str(media.get("content_hash") or "") or None,
                        ocr_cache_key=ocr_adapter_key,
                        precomputed_survey=precomputed_visual_survey,
                        worker_pool=visual_pool,
                    )
                finally:
                    visual_pool.shutdown(wait=True)
                (
                    frames,
                    payloads,
                    revisions,
                    events,
                    reviews,
                    ocr_observations,
                ) = visual_result
                manifest.finish("visual_evidence", "completed")
        except Exception as exc:
            blockers.append(f"Visual evidence stage blocked: {exc}")
            manifest.finish("visual_evidence", "blocked", str(exc))
        except BaseException as exc:
            # A process-level interruption during FFmpeg or a visual worker
            # must leave the durable stage boundary explicit before the
            # interruption is re-raised.  The partial frame/checkpoint state
            # remains available for the next explicit resume, while a running
            # visual stage can never be mistaken for a completed project.
            detail = f"Interrupted during visual evidence stage: {type(exc).__name__}"
            if str(exc):
                detail += f": {exc}"
            manifest.finish("visual_evidence", "failed", detail)
            try:
                manifest.write(run_manifest_path)
            except Exception:  # pragma: no cover - defensive persistence boundary
                LOGGER.warning(
                    "Unable to persist interrupted visual stage", exc_info=True
                )
            raise
    elif kind != "video":
        for block in blocks:
            block["visual_description"] = "[no visual source available]"

    for adjustment in transcript_timing_adjustments:
        segment_id = str(adjustment["segment_id"])
        adjusted_block_ids = [
            str(block["block_id"])
            for block in blocks
            if segment_id in {str(value) for value in block.get("transcript_segment_ids", [])}
        ]
        reviews.append(
            {
                "review_id": _next_review_id(reviews),
                "severity": "medium",
                "category": "transcript_timing_clipped_to_media",
                "start_ms": adjustment["start_ms"],
                "end_ms": adjustment["clipped_end_ms"],
                "block_ids": adjusted_block_ids,
                "segment_ids": [segment_id],
                "event_ids": [],
                "frame_ids": [],
                "ocr_observation_ids": [],
                "image_claim_ids": [],
                "metadata_revision_ids": [],
                "sufficiency_decision_ids": [],
                "problem": (
                    "A subtitle cue extends beyond the measured media duration; its canonical "
                    "end was clipped to the playable boundary."
                ),
                "alternatives": [],
                "required_action": (
                    "Confirm the source subtitle tail against the media and retain the clipped "
                    "boundary if the source was trimmed."
                ),
                "blocking": False,
                "decision": None,
                "reviewer": None,
                "decision_timestamp_utc": None,
                "rationale": None,
            }
        )

    for disagreement in transcript_disagreements:
        high_impact = bool(
            disagreement.get("selected_high_impact_missing_from_other")
            or disagreement.get("other_high_impact_added")
        )
        reviews.append(
            {
                "review_id": _next_review_id(reviews),
                "severity": "critical" if high_impact else "high",
                "category": (
                    "high_impact_transcript_candidate_disagreement"
                    if high_impact
                    else "transcript_candidate_disagreement"
                ),
                "start_ms": 0,
                "end_ms": media.get("duration_ms"),
                "block_ids": [str(block["block_id"]) for block in blocks],
                "segment_ids": [str(segment["segment_id"]) for segment in segments],
                "event_ids": [],
                "frame_ids": [],
                "ocr_observation_ids": [],
                "image_claim_ids": [],
                "metadata_revision_ids": [],
                "sufficiency_decision_ids": [],
                "problem": (
                    "Independent local ASR candidates disagree: "
                    + json.dumps(disagreement, ensure_ascii=False, sort_keys=True)
                ),
                "alternatives": [
                    str(disagreement["selected_candidate_id"]),
                    str(disagreement["other_candidate_id"]),
                ],
                "required_action": (
                    "Inspect the bounded audio and forced-alignment evidence; preserve exact "
                    "high-impact wording if the evidence does not decide it."
                ),
                "blocking": True,
                "decision": None,
                "reviewer": None,
                "decision_timestamp_utc": None,
                "rationale": None,
            }
        )

    # Scene/state candidates can reveal a meaningful visual-only event in a
    # transcript gap.  Keep chapter membership bidirectional after that late
    # block insertion.
    chapter_by_id: dict[str, dict[str, Any]] = {
        str(chapter.get("chapter_id")): chapter for chapter in chapters
    }
    if not chapter_by_id:
        chapters.append(
            {
                "chapter_id": "C001",
                "title": "Incomplete reconstruction (navigational)",
                "start_ms": 0,
                "end_ms": media.get("duration_ms") or 0,
                "block_ids": [],
                "source_authored": False,
            }
        )
        chapter_by_id["C001"] = chapters[-1]
    for block in blocks:
        start_ms = block.get("start_ms")
        if start_ms is not None:
            bucket = max(0, int(start_ms)) // 600_000
            while len(chapters) <= bucket:
                index = len(chapters)
                duration_value = (
                    int(media["duration_ms"]) if media.get("duration_ms") is not None else None
                )
                new_chapter: dict[str, Any] = {
                    "chapter_id": f"C{index + 1:03d}",
                    "title": f"Navigational chapter {index + 1}",
                    "start_ms": index * 600_000,
                    "end_ms": min((index + 1) * 600_000, duration_value)
                    if duration_value is not None
                    else None,
                    "block_ids": [],
                    "source_authored": False,
                }
                chapters.append(new_chapter)
                chapter_by_id[str(new_chapter["chapter_id"])] = new_chapter
        chapter_id = str(block.get("chapter_id") or "C001")
        if (
            start_ms is not None
            and not block.get("transcript_segment_ids")
            and int(start_ms) // 600_000 > 0
        ):
            chapter_id = f"C{int(start_ms) // 600_000 + 1:03d}"
        selected_chapter = chapter_by_id.get(chapter_id)
        if selected_chapter is None and start_ms is not None:
            selected_chapter = chapter_by_id.get(f"C{int(start_ms) // 600_000 + 1:03d}")
        selected_chapter = selected_chapter or chapter_by_id["C001"]
        block["chapter_id"] = str(selected_chapter["chapter_id"])
        block_ids = selected_chapter.setdefault("block_ids", [])
        if block["block_id"] not in block_ids:
            block_ids.append(block["block_id"])

    # Visual-only events are discovered after the transcript scaffold and are
    # appended during the survey. Reorder the shared block list before timeline
    # construction and Markdown rendering so chronology remains deterministic.
    blocks.sort(
        key=lambda item: (
            int(item.get("start_ms") or 0),
            int(item.get("end_ms") or item.get("start_ms") or 0),
            str(item.get("block_id", "")),
        )
    )

    if frames:
        try:
            sufficiency_decisions = _ensure_sufficiency_decisions(
                project_dir,
                frames=frames,
                payloads=payloads,
                revisions=revisions,
                events=events,
                blocks=blocks,
                reviews=reviews,
                existing_decisions=sufficiency_decisions,
            )
        except Exception as exc:
            blockers.append(f"Metadata sufficiency stage blocked: {exc}")

    timeline_records: list[dict[str, Any]] = []
    timeline_report: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "timed_count": 0,
        "untimed_count": 0,
    }
    candidate_snapshots: list[dict[str, Any]] = []
    candidate_metadata_count = 0
    candidate_semantic_count = 0
    manifest.start("timeline")
    try:
        from .timeline import build_timeline, validate_timeline

        ledger_path = project_dir / ".state" / "vision" / "image-observations.json"
        if ledger_path.is_file():
            try:
                ledger_data = json.loads(ledger_path.read_text(encoding="utf-8"))
                if isinstance(ledger_data, dict):
                    candidate_snapshots = [
                        item
                        for item in ledger_data.get("candidate_frames", [])
                        if isinstance(item, dict)
                    ]
                    candidate_metadata_count = sum(
                        1
                        for item in ledger_data.get("candidate_payloads", [])
                        if isinstance(item, dict)
                    )
                    candidate_semantic_count = sum(
                        1
                        for item in ledger_data.get("candidate_payloads", [])
                        if isinstance(item, dict)
                        and item.get("analysis", {}).get("semantic_status") == "observed"
                    )
            except (OSError, json.JSONDecodeError):
                candidate_snapshots = []
        timeline_items = build_timeline(
            segments,
            chapters=chapters,
            visual_events=events,
            ocr_observations=ocr_observations,
            snapshots=[*frames, *candidate_snapshots],
            review_decisions=reviews,
            media_duration_ms=media.get("duration_ms"),
        )
        checked_timeline = validate_timeline(
            timeline_items, media_duration_ms=media.get("duration_ms")
        )
        timeline_report = {
            "valid": checked_timeline.valid,
            "errors": list(checked_timeline.errors),
            "warnings": list(checked_timeline.warnings),
            "timed_count": checked_timeline.timed_count,
            "untimed_count": checked_timeline.untimed_count,
        }
        timeline_records = [item.model_dump() for item in timeline_items]
        atomic_write_json(
            project_dir / ".state" / "timeline" / "timeline.json",
            {"schema_version": "1.0", "items": timeline_records, "validation": timeline_report},
            compact=True,
        )
        if not checked_timeline.valid:
            blockers.append("Timeline stage blocked: " + "; ".join(checked_timeline.errors))
            manifest.finish("timeline", "blocked", "; ".join(checked_timeline.errors))
        else:
            manifest.finish("timeline", "completed")
    except Exception as exc:
        blockers.append(f"Timeline stage blocked: {exc}")
        manifest.finish("timeline", "blocked", str(exc))

    if blockers:
        reviews.append(
            {
                "review_id": _next_review_id(reviews),
                "severity": "critical",
                "category": "blocked_prerequisite",
                "start_ms": 0,
                "end_ms": media.get("duration_ms"),
                "block_ids": [],
                "segment_ids": [],
                "event_ids": [],
                "frame_ids": [],
                "ocr_observation_ids": [],
                "image_claim_ids": [],
                "metadata_revision_ids": [],
                "sufficiency_decision_ids": [],
                "problem": "; ".join(blockers),
                "alternatives": [],
                "required_action": "Resolve the exact blocker and resume the preserved project.",
                "blocking": True,
                "decision": None,
                "reviewer": None,
                "decision_timestamp_utc": None,
                "rationale": None,
            }
        )
    if visual_only_no_speech_fallback:
        reviews.append(
            {
                "review_id": _next_review_id(reviews),
                "severity": "high",
                "category": "no_speech_visual_only_fallback",
                "start_ms": 0,
                "end_ms": media.get("duration_ms"),
                "block_ids": [str(block.get("block_id")) for block in blocks],
                "segment_ids": [],
                "event_ids": [],
                "frame_ids": [],
                "ocr_observation_ids": [],
                "image_claim_ids": [],
                "metadata_revision_ids": [],
                "sufficiency_decision_ids": [],
                "problem": (
                    "The completed local large-v3 pass produced no reliable spoken segments. "
                    "This reconstruction contains visual evidence only."
                ),
                "alternatives": [],
                "required_action": (
                    "Confirm whether the source contains recoverable speech; if it does, "
                    "supply a verified transcript or a reviewed ASR configuration."
                ),
                "blocking": False,
                "decision": None,
                "reviewer": None,
                "decision_timestamp_utc": None,
                "rationale": None,
            }
        )

    project: dict[str, Any] = {
        "schema_version": "1.0",
        "source_title": source.stem,
        "project_status": "processing",
        "status_reason": "Mandatory audits have not yet completed.",
        "generated_at_utc": _now(),
        "fidelity_mode": fidelity_mode,
        # Only spoken segment/word labels establish a primary language. An
        # explicit ASR hint must not label a visual-only/no-speech artifact.
        "primary_language": language if segments else "und",
        "visual_source_available": kind == "video",
        "input_reference": source.name,
        "media": media,
        "transcript_candidates": transcript_candidates,
        "transcript_segments": segments,
        "repairs": repair_records,
        "frames": frames,
        "evidence_image_metadata": payloads,
        "ocr_observations": ocr_observations,
        "visual_observations": visual_observations,
        "image_claims": image_claims,
        "metadata_revisions": revisions,
        "sufficiency_decisions": sufficiency_decisions,
        "visual_events": events,
        "chapters": chapters,
        "script_blocks": blocks,
        "review_items": reviews,
        "transcript_source_decision": source_decision,
        "corrections": corrections,
        "final_signoffs": final_signoffs,
        "timeline": timeline_records,
        "state_metadata": {
            "timeline_validation": timeline_report,
            "candidate_image_count": len(candidate_snapshots),
            "candidate_metadata_image_count": candidate_metadata_count,
            "candidate_semantically_analyzed_image_count": candidate_semantic_count,
            "reconstruction_mode": (
                "visual_only_no_speech" if visual_only_no_speech_fallback else "spoken"
            ),
            "speech_recovery": {
                "asr_completed_without_segments": asr_completed_without_segments,
                "asr_had_failure": asr_had_failure,
                "dialogue_inferred": False if visual_only_no_speech_fallback else None,
            },
        },
        "tools_models_summary": "Deterministic Python/FFmpeg pipeline; see run manifest for exact availability and degradations.",
        "config_hash": cfg_hash,
        "code_version": __version__,
        "state_transitions": [],
    }
    audit = audit_project(project)
    if blockers:
        audit["final_project_status"] = "blocked"
        if "blocked_prerequisite" not in audit["blocking_failures"]:
            audit["blocking_failures"].append("blocked_prerequisite")
    project["audit"] = audit
    project["project_status"] = audit["final_project_status"]
    project["status_reason"] = (
        "; ".join(blockers)
        if blockers
        else (
            "Consequential visual or wording uncertainty remains in the review queue."
            if project["project_status"] == "review_required"
            else "All mandatory deterministic audits passed; no human verification is implied."
        )
    )
    project["state_transitions"].append(
        {
            "from": "processing",
            "to": project["project_status"],
            "at_utc": _now(),
            "reason": project["status_reason"],
        }
    )

    # Mark the canonical transaction before its first write.  If a process
    # stops after canonical JSON exists but before the remaining state and
    # validation receipt settle, resume must rebuild rather than trust a
    # partially committed tree as an unchanged cache hit.
    manifest.run_state = "finalizing"
    manifest.write(run_manifest_path)
    manifest.artifacts = [
        markdown_path.name,
        ".state/canonical-project.json",
        ".state/run-manifest.json",
        ".state/audit.json",
        ".state/review-queue.json",
        ".state/timeline/timeline.json",
        *([".state/asr-progress.json"] if asr_progress_path.is_file() else []),
        *[frame["full_frame_path"] for frame in frames],
    ]
    manifest_dict = manifest.as_dict()
    manifest_dict["run_cache_key"] = run_key
    manifest_dict["tool_versions"] = {
        name: value for name, value in tool_inputs.items() if value is not None
    }
    selected_transcript_candidate = next(
        (
            item
            for item in transcript_candidates
            if str(item.get("decision_rationale", "")).startswith("Selected as")
        ),
        None,
    )
    if selected_transcript_candidate is None:
        selected_asr_model = "unavailable" if not transcript_candidates else "not_used"
    elif selected_transcript_candidate.get("source_type") != "local_asr":
        selected_asr_model = (
            "not_used:" + str(selected_transcript_candidate.get("source_type") or "transcript")
        )
    else:
        selected_origin = str(selected_transcript_candidate.get("origin") or "unknown")
        model_name_by_origin = {
            "faster-whisper": "faster-whisper-large-v3",
            "qwen3-asr": "qwen3-asr-1.7b",
            "moss-transcribe-diarize": "moss-transcribe-diarize-0.9b",
        }
        managed_name = model_name_by_origin.get(selected_origin)
        selected_asr_model = (
            f"{selected_origin}:{managed_revision(managed_name)}"
            if managed_name is not None
            else selected_origin
        )
    manifest_dict["model_versions"] = {
        "asr": selected_asr_model,
        "asr_config": config.asr.model,
        "vision_mode": vision_mode,
        "vision_provider": vision_adapter_key,
        "ocr_adapter": ocr_adapter_key,
    }
    manifest_dict["provider_usage"] = [
        {"route": "local", "provider": ocr_adapter_key, "purpose": "optional-ocr"}
        if ocr_adapter is not None
        else {"route": "none", "provider": "semantic-observer-unavailable", "purpose": "visual"}
    ]
    manifest_dict["reproducibility"] = {
        **dict(manifest_dict.get("reproducibility", {})),
        "run_cache_key": run_key,
        "adapter_keys": {
            "asr": asr_adapter_key,
            "ocr": ocr_adapter_key,
            "vision": vision_adapter_key,
        },
        "tool_versions": manifest_dict["tool_versions"],
    }
    project["manifest"] = manifest_dict
    canonical_path = project_dir / ".state" / "canonical-project.json"
    atomic_write_json(
        canonical_path,
        project,
        compact=canonical_compact_for_payload(canonical_path, project),
    )
    atomic_write_json(project_dir / ".state" / "audit.json", audit)
    atomic_write_json(project_dir / ".state" / "review-queue.json", reviews)
    atomic_write_json(project_dir / ".state" / "corrections.json", [])
    atomic_write_json(project_dir / ".state" / "run-manifest.json", manifest_dict)
    render_to_path(project, markdown_path)
    # The first pass is a structural/output-contract preflight.  Per-image
    # verification is reserved for the independent final proof after audit and
    # compaction; any final metadata failure still blocks the returned project.
    # This avoids decoding or hashing every evidence image twice on long runs
    # without weakening the final contract.
    validation = validate_project(
        project_dir,
        use_cached_file_hash=True,
        verify_metadata=False,
    )
    if not validation.valid:
        if project["project_status"] != "blocked":
            project["project_status"] = "blocked"
            project["status_reason"] = "Output validation failed: " + "; ".join(
                validation.errors
            )
            project["audit"]["final_project_status"] = "blocked"
            project["audit"]["blocking_failures"].append("output_validation")
            atomic_write_json(
                project_dir / ".state" / "canonical-project.json",
                project,
                compact=canonical_compact_for_payload(
                    project_dir / ".state" / "canonical-project.json", project
                ),
            )
            atomic_write_json(project_dir / ".state" / "audit.json", project["audit"])
            render_to_path(project, markdown_path)
        validation = validate_project(project_dir, use_cached_file_hash=True)
    if validation.valid:
        # Persist the renderer/output checks in the same audit displayed by the
        # Markdown. audit_project is intentionally canonical-data-only, so the
        # filesystem/navigation results are attached after the production
        # validator has parsed the rendered artifact and all embedded images.
        project["audit"]["anchor_navigation_checks"] = [
            f"{validation.checks.get('anchors', 0)} explicit anchors validated with no unresolved internal links"
        ]
        project["audit"]["output_contract_checks"] = [
            f"{validation.checks.get('markdown_count', 0)}/1 Markdown artifacts",
            f"{validation.checks.get('html_count', 0)}/0 forbidden HTML artifacts",
            f"{validation.checks.get('image_links', 0)}/{validation.checks.get('evidence_images', 0)} final evidence images linked",
            "canonical schema, timeline, trace links, and embedded image metadata validated",
        ]
        atomic_update_json_fields(
            project_dir / ".state" / "canonical-project.json",
            {"audit": project["audit"]},
            fallback_payload=project,
        )
        atomic_write_json(project_dir / ".state" / "audit.json", project["audit"])
        render_to_path(project, markdown_path)
        # The final post-compaction validation below verifies this rewritten
        # canonical/audit/render set for normal runs. A blocked project skips
        # compaction, so validate the blocked rewrite immediately instead of
        # leaving the returned result tied to the pre-audit pass.
        if project.get("project_status") == "blocked":
            validation = validate_project(project_dir, use_cached_file_hash=True)
    semantic_usage: dict[str, Any] | None = None
    semantic_error: str | None = None
    if frames and vision_mode == "host-agent":
        # Create a bounded, content-addressed request bundle instead of one
        # request file per frame.  The host can let a Codex subagent inspect
        # only this frontier and later apply schema-valid responses atomically.
        try:
            from .providers.host_agent import HostAgentVisionProvider
            from .semantic_pipeline import apply_vision_provider, pending_packet_count
            from .subagent_review import create_review_bundle

            raw_budget = str(
                semantic_max_packets
                if semantic_max_packets is not None
                else os.environ.get(
                    "VSR_HOST_REVIEW_MAX_PACKETS",
                    os.environ.get(
                        "VSR_SEMANTIC_MAX_PACKETS", str(_DEFAULT_HOST_REVIEW_MAX_PACKETS)
                    ),
                )
            ).strip()
            try:
                review_budget = max(1, int(raw_budget))
            except ValueError:
                review_budget = 8
            deterministic_summary = apply_vision_provider(
                project_dir,
                HostAgentVisionProvider(),
                semantic_workers=1,
                deterministic_only=True,
            )
            pending_count = pending_packet_count(project_dir)
            bundle = (
                create_review_bundle(project_dir, max_packets=review_budget)
                if pending_count
                else None
            )
            request_count = int(bundle.get("request_count", 0)) if bundle else 0
            semantic_usage = {
                "route": "host_agent",
                "provider": "codex-subagent",
                "model": "codex-subagent",
                "model_version": None,
                "purpose": "visual",
                "network_required": False,
                # A no-work resume should not accumulate empty content-addressed
                # bundles.  Keep the explicit create_review_bundle API intact,
                # but do not publish its empty directory through the pipeline.
                "review_bundle_dir": bundle["bundle_dir"] if bundle and request_count else None,
                "review_max_packets": review_budget,
                "review_request_count": request_count,
                "review_candidate_ids": [
                    str(item.get("candidate_id"))
                    for item in (bundle or {}).get("requests", [])
                    if request_count and isinstance(item, dict) and item.get("candidate_id")
                ],
                "deterministic_no_change_count": int(
                    deterministic_summary.get("semantic_deterministic_no_change_count", 0)
                ),
            }
        except Exception as exc:
            semantic_error = str(exc)
    # ``auto`` is normalized to ``host-agent`` at the public entrypoint above;
    # keep the legacy local/external branch explicit so a future refactor
    # cannot silently route the default path through the local Qwen provider.
    elif frames and vision_mode in {"local", "external"}:
        from .semantic_pipeline import (
            apply_vision_provider,
            pending_packet_count,
            refresh_semantic_state,
        )

        active_provider = vision_provider
        managed_provider = None
        try:
            pending_semantic = pending_packet_count(project_dir)
        except Exception as exc:
            pending_semantic = 0
            semantic_error = str(exc)
        # A resumed project may already have persisted observations and claims
        # but still contain stale block-level placeholders from an earlier
        # incremental ingest. Reconcile that state without constructing or
        # invoking an expensive vision provider when there is no work left.
        if semantic_error is None and not pending_semantic:
            try:
                refresh_semantic_state(project_dir)
            except Exception as exc:
                semantic_error = str(exc)
        if semantic_error is None and pending_semantic:
            if active_provider is None and vision_mode == "local":
                try:
                    from .providers.llama_cpp import LlamaCppVisionProvider

                    managed_provider = LlamaCppVisionProvider()
                    active_provider = managed_provider
                except Exception as exc:
                    if vision_mode == "local":
                        semantic_error = str(exc)
            elif active_provider is None and vision_mode == "external":
                semantic_error = "External vision mode requires an explicitly configured provider."
            if active_provider is not None:
                try:
                    summary = apply_vision_provider(project_dir, active_provider)
                    semantic_usage = {
                        "route": active_provider.descriptor.route,
                        "provider": active_provider.descriptor.provider_id,
                        "model": active_provider.descriptor.model,
                        "model_version": active_provider.descriptor.model_version,
                        "purpose": "visual",
                        "applied_observations": len(summary["applied"]),
                        "skipped_events": len(summary["skipped_event_ids"]),
                        "semantic_cache_enabled": bool(
                            summary.get("semantic_cache_enabled", False)
                        ),
                        "semantic_cache_hit_count": int(
                            summary.get("semantic_cache_hit_count", 0)
                        ),
                        "semantic_cache_miss_count": int(
                            summary.get("semantic_cache_miss_count", 0)
                        ),
                        "semantic_cache_write_count": int(
                            summary.get("semantic_cache_write_count", 0)
                        ),
                        "semantic_content_cache_hit_count": int(
                            summary.get("semantic_content_cache_hit_count", 0)
                        ),
                        "semantic_content_cache_write_count": int(
                            summary.get("semantic_content_cache_write_count", 0)
                        ),
                        "semantic_visual_reuse_hit_count": int(
                            summary.get("semantic_visual_reuse_hit_count", 0)
                        ),
                        "semantic_visual_content_reuse_hit_count": int(
                            summary.get("semantic_visual_content_reuse_hit_count", 0)
                        ),
                        "semantic_provider_attempt_failure_count": int(
                            summary.get("semantic_provider_attempt_failure_count", 0)
                        ),
                        "semantic_fallback_annotation_count": int(
                            summary.get("semantic_fallback_annotation_count", 0)
                        ),
                        "semantic_circuit_breaker_triggered": bool(
                            summary.get("semantic_circuit_breaker_triggered", False)
                        ),
                        "semantic_provider_failures": list(
                            summary.get("semantic_provider_failures", [])
                        ),
                        "semantic_deferred_event_count": len(
                            summary.get("semantic_deferred_event_ids", [])
                        ),
                        "semantic_deferred_event_ids": list(
                            summary.get("semantic_deferred_event_ids", [])
                        ),
                    }
                except Exception as exc:
                    semantic_error = str(exc)
                finally:
                    if managed_provider is not None:
                        managed_provider.close()

    if semantic_usage is not None or semantic_error is not None:
        project = json.loads(
            (project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
        )
        stored_manifest = project.setdefault("manifest", {})
        usage = stored_manifest.setdefault("provider_usage", [])
        if semantic_usage is not None:
            usage.append(semantic_usage)
        if semantic_error is not None:
            stored_manifest.setdefault("degradations", []).append(
                f"Semantic vision provider failed: {semantic_error}"
            )
            if vision_mode in {"local", "external"}:
                project.setdefault("review_items", []).append(
                    {
                        "review_id": _next_review_id(project.get("review_items", [])),
                        "severity": "critical",
                        "category": "blocked_prerequisite",
                        "start_ms": 0,
                        "end_ms": media.get("duration_ms"),
                        "block_ids": [],
                        "segment_ids": [],
                        "event_ids": [],
                        "frame_ids": [],
                        "ocr_observation_ids": [],
                        "image_claim_ids": [],
                        "metadata_revision_ids": [],
                        "sufficiency_decision_ids": [],
                        "problem": f"Requested semantic vision backend failed: {semantic_error}",
                        "alternatives": [],
                        "required_action": "Repair the configured provider and resume the project.",
                        "blocking": True,
                        "decision": None,
                        "reviewer": None,
                        "decision_timestamp_utc": None,
                        "rationale": None,
                    }
                )
        if semantic_usage is not None and int(
            semantic_usage.get("semantic_fallback_annotation_count", 0)
        ):
            fallback_count = int(semantic_usage["semantic_fallback_annotation_count"])
            attempted_failures = int(
                semantic_usage.get("semantic_provider_attempt_failure_count", 0)
            )
            degradation = (
                "Semantic provider returned invalid responses for "
                f"{fallback_count} packet(s); conservative review-only fallbacks were persisted"
                + (
                    f" after {attempted_failures} provider attempt failure(s)."
                    if attempted_failures
                    else "."
                )
            )
            degradations = stored_manifest.setdefault("degradations", [])
            if degradation not in degradations:
                degradations.append(degradation)
        if semantic_usage is not None and int(
            semantic_usage.get("semantic_deferred_event_count", 0)
        ):
            deferred_count = int(semantic_usage["semantic_deferred_event_count"])
            degradation = (
                f"Semantic packet budget deferred {deferred_count} event(s); "
                "deterministic evidence and targeted review items were retained."
            )
            degradations = stored_manifest.setdefault("degradations", [])
            if degradation not in degradations:
                degradations.append(degradation)
        project["audit"] = audit_project(project)
        if semantic_error is not None and vision_mode in {"local", "external"}:
            project["audit"]["final_project_status"] = "blocked"
            if "blocked_prerequisite" not in project["audit"]["blocking_failures"]:
                project["audit"]["blocking_failures"].append("blocked_prerequisite")
        project["project_status"] = project["audit"]["final_project_status"]
        project["status_reason"] = (
            f"Requested semantic vision backend failed: {semantic_error}"
            if semantic_error is not None and vision_mode in {"local", "external"}
            else project.get("status_reason", "Semantic evidence processing completed.")
        )
        canonical_path = project_dir / ".state" / "canonical-project.json"
        atomic_write_json(
            canonical_path,
            project,
            compact=canonical_compact_for_payload(canonical_path, project),
        )
        atomic_write_json(project_dir / ".state" / "audit.json", project["audit"])
        atomic_write_json(project_dir / ".state" / "review-queue.json", project["review_items"])
        atomic_write_json(project_dir / ".state" / "run-manifest.json", stored_manifest)
        render_to_path(project, markdown_path)
        # Bundle creation changes only hidden request/manifest state.  Defer its
        # filesystem proof to the single final post-compaction validation; this
        # keeps the default host-agent route from decoding the entire evidence
        # tree twice while preserving the same returned validation contract.
        if vision_mode != "host-agent" or semantic_error is not None:
            validation = validate_project(project_dir, use_cached_file_hash=True)

    if validation.valid and project.get("project_status") != "blocked":
        from .cache import compact_completed_checkpoints

        try:
            compaction = compact_completed_checkpoints(
                project_dir,
                keep=_keep_completed_checkpoints(),
            )
        except Exception as exc:  # pragma: no cover - defensive cleanup boundary
            LOGGER.warning("Unable to compact completed project checkpoints: %s", exc)
            compaction = {
                "kept": True,
                "removed_files": 0,
                "reclaimed_bytes": 0,
                "targets": [],
                "error": str(exc),
            }
        project_manifest = project.setdefault("manifest", {})
        performance = project_manifest.setdefault("performance", {})
        performance["checkpoint_compaction"] = compaction
        _publish_resource_telemetry(project_dir, project, project_manifest)
        validation = validate_project(project_dir, use_cached_file_hash=True)
        if validation.valid:
            # The first telemetry fixed point deliberately precedes the full
            # metadata proof.  Write the receipt only after that proof, then
            # refresh telemetry once so the receipt itself is included in the
            # recorded output byte/file parity without another image decode.
            write_validation_receipt(
                project_dir,
                project,
                run_cache_key=run_key,
                validation=validation,
            )
            # The receipt now proves the evidence tree. Publish the completed
            # transaction marker before the final telemetry fixed-point loop so
            # the host-agent bundle (when present) hashes the same canonical
            # state that the caller receives.
            manifest.run_state = "completed"
            project_manifest["run_state"] = "completed"
            # Refreshing the receipt can change its encoded size when the
            # canonical-file signature crosses a decimal-width boundary. That
            # receipt is part of output telemetry, so settle the two mutable
            # files together instead of publishing a one-byte-stale snapshot.
            from .resource_usage import resource_snapshot

            # The receipt's own stat-bound inventory is a small self-reference:
            # refreshing its canonical signature can cross a decimal-width
            # boundary in the generated byte total.  Keep iterating until the
            # manifest's output snapshot matches the post-refresh tree rather
            # than allowing a rare one-byte stale receipt/manifest pair.
            for _ in range(12):
                _publish_resource_telemetry(project_dir, project, project_manifest)
                refresh_validation_receipt_signature(project_dir)
                recorded_output = project_manifest.get("performance", {}).get(
                    "resource_usage", {}
                ).get("output")
                if recorded_output == resource_snapshot(project_dir).get("output"):
                    break
            # The host-agent bundle is created before the final manifest /
            # telemetry writes above. Refresh its content-addressed canonical
            # hash after those writes, then rewrite the receipt so its hidden
            # file inventory includes the final bundle state. No public
            # validation is needed: the bundle contains only packet references
            # and its own manifest, while the preceding proof is unchanged.
            if (
                vision_mode == "host-agent"
                and semantic_error is None
                and isinstance(semantic_usage, dict)
                and isinstance(semantic_usage.get("review_bundle_dir"), str)
            ):
                try:
                    from .subagent_review import create_review_bundle

                    bundle_path = Path(str(semantic_usage["review_bundle_dir"]))
                    max_packets = int(semantic_usage.get("review_max_packets", 32))
                    create_review_bundle(
                        project_dir,
                        output_dir=bundle_path,
                        max_packets=max(1, max_packets),
                    )
                    write_validation_receipt(
                        project_dir,
                        project,
                        run_cache_key=run_key,
                        validation=validation,
                    )
                    refresh_validation_receipt_signature(project_dir)
                except Exception as exc:  # pragma: no cover - defensive handoff refresh
                    LOGGER.warning("Unable to refresh host-agent review bundle hash: %s", exc)
            if (
                read_trusted_validation_receipt(
                    project_dir,
                    project,
                    run_cache_key=run_key,
                )
                is None
            ):
                (project_dir / ".state" / "validation-receipt.json").unlink(missing_ok=True)
    else:
        # Runs that cannot compact (blocked or failed validation) still publish
        # the same resource telemetry before the final full validation.
        project_manifest = project.setdefault("manifest", {})
        _publish_resource_telemetry(project_dir, project, project_manifest)
        validation = validate_project(project_dir, use_cached_file_hash=True)
    status = project["project_status"]
    exit_code = (
        0
        if status in {"automatically_checked", "human_reviewed", "fully_verified"}
        else (3 if status == "review_required" else 4)
    )
    return RunResult(project_dir, markdown_path, status, exit_code, validation)
