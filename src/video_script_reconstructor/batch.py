"""Storage-aware sequential execution for a directory of source videos.

The batch runner deliberately keeps one active reconstruction at a time.  It
records a resumable manifest before and after every source, forecasts output
bytes from completed projects, and stops before violating a caller-selected
free-space reserve.  It never touches source media or model stores.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from .errors import InputError
from .media_probe import probe_media
from .pipeline import RunResult, colocated_output_dir, run_pipeline
from .security import atomic_write_json, safe_slug

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"})
DEFAULT_MIN_FREE_BYTES = 10 * 1024**3
DEFAULT_MAX_PROJECT_BYTES = 8 * 1024**3
SIDECAR_EXTENSIONS = (".srt", ".vtt", ".ass", ".ssa")


def _iter_tree_files(root: Path, *, sort_entries: bool = True) -> Iterator[Path]:
    """Yield non-symlink regular-file descendants without redundant stats.

    Batch discovery used to allocate a ``Path`` for every entry via
    ``Path.rglob`` and then perform another metadata lookup in ``is_file``.
    ``DirEntry`` type information lets this walk avoid a stat for unrelated
    files, while still rejecting symlinked directories/files and keeping
    traversal deterministic. Callers that need bytes use ``_tree_bytes``'s
    dedicated size-aware walk below.
    """

    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                ordered_entries = (
                    sorted(entries, key=lambda entry: entry.name.casefold())
                    if sort_entries
                    else entries
                )
                for entry in ordered_entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                            continue
                        if entry.is_file(follow_symlinks=False):
                            yield Path(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue


def discover_videos(source_root: str | Path) -> tuple[Path, ...]:
    """Return supported videos in deterministic relative-path order."""

    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise InputError(f"Batch source root is not a directory: {root}")
    videos = tuple(
        sorted(
            (
                path.resolve()
                for path in _iter_tree_files(root, sort_entries=False)
                if path.suffix.casefold() in VIDEO_EXTENSIONS
            ),
            key=lambda path: path.relative_to(root).as_posix().casefold(),
        )
    )
    if not videos:
        raise InputError(f"No supported videos found under batch source root: {root}")
    return videos


def discover_sidecars(source: str | Path) -> tuple[Path, ...]:
    """Return non-empty subtitle sidecars adjacent to one source video."""

    source_path = Path(source).expanduser().resolve(strict=True)
    sidecars: list[Path] = []
    for suffix in SIDECAR_EXTENSIONS:
        candidate = source_path.with_suffix(suffix)
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                sidecars.append(candidate)
        except OSError:
            continue
    return tuple(sidecars)


def _tree_bytes(root: Path) -> int:
    """Return regular output-tree bytes without following symlinks.

    ``os.walk`` materializes path strings and then performs a second metadata
    lookup for every file.  Batch planning calls this inventory before and
    after each source, so use one non-following ``DirEntry.stat`` per entry
    instead.  The mode check deliberately counts non-directory, non-symlink
    entries just as the previous filename walk did (for example, a FIFO),
    while preserving the corpus boundary by never traversing symlinked dirs.
    """

    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                ordered_entries = sorted(entries, key=lambda entry: entry.name)
                for entry in ordered_entries:
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                        mode = entry_stat.st_mode
                        if stat.S_ISLNK(mode):
                            continue
                        if stat.S_ISDIR(mode):
                            pending.append(Path(entry.path))
                            continue
                        total += int(entry_stat.st_size)
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _duration_class(duration_seconds: float) -> str:
    if duration_seconds <= 180:
        return "short"
    if duration_seconds <= 1800:
        return "medium"
    return "long"


def _historical_rates(history_roots: tuple[Path, ...]) -> dict[str, list[float]]:
    rates: dict[str, list[float]] = {"short": [], "medium": [], "long": []}
    seen: set[Path] = set()
    for root in history_roots:
        if not root.exists():
            continue
        markers = (root / ".state" / "canonical-project.json",) if root.is_dir() else ()
        canonical_paths = (
            markers
            if markers and markers[0].is_file()
            else tuple(
                path
                for path in _iter_tree_files(root, sort_entries=False)
                if path.name == "canonical-project.json"
            )
        )
        for canonical_path in canonical_paths:
            project_dir = canonical_path.parent.parent.resolve()
            if project_dir in seen:
                continue
            seen.add(project_dir)
            try:
                canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
                duration_ms = int(canonical["media"]["duration_ms"])
                manifest = json.loads(
                    (project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
                )
                output_bytes = int(
                    manifest["performance"]["resource_usage"]["output"]["bytes"]
                )
                duration_seconds = duration_ms / 1000
                if duration_seconds > 0 and output_bytes > 0:
                    rates[_duration_class(duration_seconds)].append(
                        output_bytes / duration_seconds
                    )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return rates


def estimate_project_bytes(
    duration_ms: int | None,
    *,
    history_roots: tuple[Path, ...] = (),
    historical_rates: Mapping[str, Sequence[float]] | None = None,
    safety_factor: float = 1.5,
    max_project_bytes: int = DEFAULT_MAX_PROJECT_BYTES,
) -> int:
    """Forecast one output tree conservatively without decoding the media.

    ``historical_rates`` is an optional caller-owned snapshot.  Batch runs use
    one snapshot for the whole pass and update it after each completed source,
    avoiding a recursive history scan for every video while preserving the
    public, standalone behavior when the argument is omitted.
    """

    seconds = max(1.0, (duration_ms or 0) / 1000)
    duration_class = _duration_class(seconds)
    historical = (
        historical_rates
        if historical_rates is not None
        else _historical_rates(history_roots)
    )
    fallback_rates = {"short": 1_500_000.0, "medium": 400_000.0, "long": 250_000.0}
    rate = float(median(historical[duration_class])) if historical[duration_class] else fallback_rates[duration_class]
    forecast = int(max(64 * 1024**2, seconds * rate * max(1.0, safety_factor)))
    return min(max_project_bytes, forecast)


def _record_historical_rate(
    rates: dict[str, list[float]], *, duration_ms: int | None, output_bytes: int
) -> None:
    """Add one completed output measurement to a mutable batch snapshot."""

    if duration_ms is not None and duration_ms > 0 and output_bytes > 0:
        seconds = duration_ms / 1000
        rates[_duration_class(seconds)].append(output_bytes / seconds)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"schema_version": "1.0", "sources": []}
    return value if isinstance(value, dict) else {"schema_version": "1.0", "sources": []}


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(path, manifest, compact=True)


def _result_record(source: Path, result: RunResult, *, output_bytes: int) -> dict[str, Any]:
    validation = result.validation
    return {
        "source": str(source),
        "project_dir": str(result.project_dir),
        "markdown": str(result.markdown_path),
        "status": result.status,
        "exit_code": result.exit_code,
        "output_bytes": output_bytes,
        "validation_valid": bool(validation.valid) if validation is not None else None,
        "validation_errors": list(validation.errors) if validation is not None else [],
    }


def run_batch(
    source_root: str | Path,
    *,
    output_root: str | Path | None = None,
    preset: str = "strict",
    vision_mode: str = "host-agent",
    language: str | None = None,
    compare_sidecars: bool = False,
    offline: bool = True,
    resume: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    max_project_bytes: int = DEFAULT_MAX_PROJECT_BYTES,
    semantic_max_packets: int | None = None,
    history_roots: tuple[Path, ...] = (),
    stop_on_blocked: bool = True,
    dry_run: bool = False,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run supported videos one at a time with a resumable storage guard.

    ``language`` is an optional shared Whisper hint for homogeneous corpora;
    leaving it unset preserves independent per-chunk language detection.
    ``compare_sidecars`` opt-in runs Whisper alongside adjacent subtitle
    sidecars and preserves candidate disagreements without trusting either
    source blindly.
    """

    source_root_path = Path(source_root).expanduser().resolve(strict=True)
    configured_root = os.environ.get("VSR_OUTPUT_ROOT", "").strip()
    colocated_batch = output_root is None and not configured_root
    if output_root is not None:
        output_root_path = Path(output_root).expanduser().resolve()
    elif configured_root:
        output_root_path = Path(configured_root).expanduser().resolve()
    else:
        # Keep the resumable batch manifest out of the source repository while
        # leaving each project's report/evidence beside its own video.
        output_root_path = source_root_path / "(Analyzer Batch Outputs)"
    output_root_path.mkdir(parents=True, exist_ok=True)
    videos = discover_videos(source_root_path)
    if len({safe_slug(path.stem) for path in videos}) != len(videos):
        raise InputError("Batch videos have colliding output slugs; rename duplicates before running")
    manifest_path = output_root_path / ".challenge-batch.json"
    manifest = _load_manifest(manifest_path)
    manifest.setdefault("schema_version", "1.0")
    manifest["source_root"] = str(source_root_path)
    manifest["output_root"] = str(output_root_path)
    manifest["policy"] = {
        "min_free_bytes": min_free_bytes,
        "max_project_bytes": max_project_bytes,
        "semantic_max_packets": semantic_max_packets,
        "language": language,
        "compare_sidecars": compare_sidecars,
        "stop_on_blocked": stop_on_blocked,
    }
    records: list[dict[str, Any]] = [
        item for item in manifest.get("sources", []) if isinstance(item, dict)
    ]
    by_source = {str(item.get("source")): item for item in records}
    planned: list[dict[str, Any]] = []
    executed: list[dict[str, Any]] = []
    blocked = False
    # History is immutable for the duration of a forecast pass except for
    # measurements produced by this pass.  Keep one in-memory snapshot rather
    # than recursively parsing every prior project once per source video.
    historical_rates = _historical_rates((*history_roots, output_root_path))
    previous_budget = os.environ.get("VSR_SEMANTIC_MAX_PACKETS")
    try:
        if semantic_max_packets is not None:
            os.environ["VSR_SEMANTIC_MAX_PACKETS"] = str(max(1, semantic_max_packets))
        for source in videos:
            probe = probe_media(source)
            estimate = estimate_project_bytes(
                probe.duration_ms,
                historical_rates=historical_rates,
                max_project_bytes=max_project_bytes,
            )
            project_dir = (
                colocated_output_dir(source)
                if colocated_batch
                else output_root_path / safe_slug(source.stem)
            )
            sidecars = discover_sidecars(source) if compare_sidecars else ()
            existing_bytes = _tree_bytes(project_dir) if project_dir.is_dir() else 0
            free_bytes = int(shutil.disk_usage(output_root_path).free)
            required_bytes = max(0, estimate - existing_bytes) + min_free_bytes
            plan = {
                "source": str(source),
                "duration_ms": probe.duration_ms,
                "source_bytes": probe.size_bytes,
                "project_dir": str(project_dir),
                "estimated_output_bytes": estimate,
                "existing_output_bytes": existing_bytes,
                "free_bytes": free_bytes,
                "required_bytes": required_bytes,
                "fits_storage_policy": free_bytes >= required_bytes,
                "sidecars": [str(path) for path in sidecars],
            }
            planned.append(plan)
            if free_bytes < required_bytes:
                blocked = True
                record = {**plan, "status": "storage_blocked", "exit_code": 4}
                by_source[str(source)] = record
                if stop_on_blocked:
                    break
                continue
            if dry_run:
                continue
            try:
                result = run_pipeline(
                    source,
                    output_root=None if colocated_batch else output_root_path,
                    preset=preset,
                    vision_mode=vision_mode,
                    language=language,
                    subtitles=sidecars,
                    subtitle_mode="compare-all" if compare_sidecars else "auto",
                    resume=resume,
                    offline=offline,
                    progress_callback=progress_callback,
                )
                record = {
                    **plan,
                    **_result_record(
                        source,
                        result,
                        output_bytes=_tree_bytes(result.project_dir),
                    ),
                }
            except Exception as exc:
                record = {
                    **plan,
                    "status": "error",
                    "exit_code": 4,
                    "error": str(exc),
                    "output_bytes": _tree_bytes(project_dir),
                }
                blocked = True
            by_source[str(source)] = record
            executed.append(record)
            measured_output = record.get("output_bytes")
            if record.get("validation_valid") is True and isinstance(measured_output, int):
                _record_historical_rate(
                    historical_rates,
                    duration_ms=probe.duration_ms,
                    output_bytes=measured_output,
                )
            _write_manifest(manifest_path, {**manifest, "sources": list(by_source.values())})
            record_exit_code = record.get("exit_code", 0)
            if stop_on_blocked and isinstance(record_exit_code, int) and record_exit_code == 4:
                blocked = True
                break
    finally:
        if previous_budget is None:
            os.environ.pop("VSR_SEMANTIC_MAX_PACKETS", None)
        else:
            os.environ["VSR_SEMANTIC_MAX_PACKETS"] = previous_budget
    final_manifest = {
        **manifest,
        "sources": list(by_source.values()),
        "planned": planned,
        "executed": executed,
        "completed_count": sum(1 for item in by_source.values() if item.get("status") in {"automatically_checked", "review_required", "human_reviewed", "fully_verified"}),
        "blocked": blocked,
    }
    _write_manifest(manifest_path, final_manifest)
    return final_manifest


__all__ = ["discover_sidecars", "discover_videos", "estimate_project_bytes", "run_batch"]
