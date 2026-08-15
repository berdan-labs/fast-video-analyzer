"""Corpus-level orchestration for file-based Codex/subagent review bundles.

The single-project bundle implementation owns packet selection, content hashes,
and annotation validation.  This module deliberately stays outside that
semantic core: it discovers canonical projects, skips projects whose review
frontier is already empty, and schedules one small JSON handoff at a time.

Review bundles contain references and hashes, never copied evidence images or
source media.  The storage guard therefore budgets only the bounded request
metadata that can be written; it does not count source media as output and it
never purges an existing project or bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import InputError
from .security import safe_slug
from .semantic_pipeline import pending_packet_count
from .subagent_review import create_review_bundle

DEFAULT_REVIEW_MIN_FREE_BYTES = 10 * 1024**3
DEFAULT_REVIEW_MAX_PACKETS = 8
# A request is validated as a bounded file by the bundle applier.  Reserve the
# same upper bound per request plus a small amount for bundle metadata.  This is
# intentionally conservative: real bundles are usually several orders of
# magnitude smaller and contain no image bytes.
DEFAULT_REVIEW_MAX_BUNDLE_BYTES = 64 * 1024**2
_REQUEST_SIZE_LIMIT = 4 * 1024**2
_BUNDLE_METADATA_RESERVE = 1 * 1024**2
# Dynamic staging is intentionally disabled for tiny caller budgets.  Those
# budgets are usually tests or explicit safety probes, and should retain the
# strict analytical preflight without writing even a temporary handoff.
_DYNAMIC_PREFLIGHT_MIN_BYTES = 1 * 1024**2


class _MeasuredBundleBudgetExceeded(InputError):
    """A staged reference-only bundle exceeded the caller's measured cap."""

    def __init__(self, actual_bytes: int, max_bytes: int) -> None:
        super().__init__(
            f"Measured review bundle exceeds the configured metadata budget: "
            f"{actual_bytes} > {max_bytes} bytes"
        )
        self.actual_bytes = actual_bytes


def _canonical_marker(project: Path) -> Path:
    return project / ".state" / "canonical-project.json"


def discover_review_projects(root: str | Path) -> tuple[Path, ...]:
    """Find canonical reconstruction projects in deterministic path order.

    Once a project marker is found, its descendants are pruned from the walk;
    generated evidence trees can be very large and nested canonical projects
    are not a valid reconstruction layout.  Symlinked directories are never
    traversed.  This keeps discovery cheap and prevents a review command from
    escaping the requested corpus boundary.
    """

    target = Path(root).expanduser().resolve(strict=True)
    if not target.is_dir():
        raise InputError(f"Review project root is not a directory: {target}")
    state_dir = target / ".state"
    direct_marker = _canonical_marker(target)
    if (
        state_dir.is_dir()
        and not state_dir.is_symlink()
        and direct_marker.is_file()
        and not direct_marker.is_symlink()
    ):
        return (target,)

    found: set[Path] = set()
    try:
        for directory, directory_names, _file_names in os.walk(
            target, topdown=True, followlinks=False
        ):
            current = Path(directory)
            # Do not recurse through reparse/symlink directories.  They are
            # outside the corpus contract and may point at source/model data.
            safe_names: list[str] = []
            for name in directory_names:
                child = current / name
                try:
                    if child.is_symlink():
                        continue
                except OSError as exc:
                    raise InputError(f"Unable to inspect review root: {child}") from exc
                safe_names.append(name)
            directory_names[:] = safe_names

            state_dir = current / ".state"
            marker = state_dir / "canonical-project.json"
            if (
                ".state" in directory_names
                and state_dir.is_dir()
                and not state_dir.is_symlink()
                and marker.is_file()
                and not marker.is_symlink()
            ):
                found.add(current.resolve())
                # No need to enumerate evidence/checkpoint descendants after
                # the marker proves this is a complete project root.
                directory_names[:] = []
    except OSError as exc:
        raise InputError(f"Unable to discover review projects below: {target}") from exc

    if not found:
        raise InputError(f"No canonical reconstruction projects found below: {target}")
    return tuple(
        sorted(
            found,
            key=lambda path: (
                path.relative_to(target).as_posix().casefold(),
                path.relative_to(target).as_posix(),
            ),
        )
    )


def estimate_review_bundle_bytes(
    max_packets: int,
    *,
    max_request_bytes: int = _REQUEST_SIZE_LIMIT,
    metadata_reserve: int = _BUNDLE_METADATA_RESERVE,
) -> int:
    """Return a conservative upper bound for one JSON-only review bundle."""

    if max_packets <= 0:
        raise ValueError("max_packets must be positive")
    if max_request_bytes <= 0 or metadata_reserve < 0:
        raise ValueError("review bundle size limits must be non-negative")
    return max_packets * max_request_bytes + metadata_reserve


def _tree_bytes(root: Path) -> int:
    """Measure regular-file bytes below ``root`` without following links.

    Review handoffs are intentionally metadata-only, but a corpus can retain
    many prior bundles.  ``os.walk`` allocates a pair of lists for every
    directory and performs a second path-based stat for every file.  The
    scandir iterator already exposes directory entries and their lstat-style
    metadata, so an explicit stack keeps storage preflight bounded without
    traversing symlink/reparse trees.
    """

    total = 0
    if not root.is_dir() or root.is_symlink():
        return 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    mode = entry_stat.st_mode
                    if stat.S_ISLNK(mode):
                        continue
                    if stat.S_ISDIR(mode):
                        pending.append(Path(entry.path))
                        continue
                    total += int(entry_stat.st_size)
        except OSError:
            continue
    return total


def _project_output_dir(project: Path, corpus_root: Path, output_root: Path | None) -> Path | None:
    if output_root is None:
        return None
    relative = project.relative_to(corpus_root).as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
    return output_root / f"{safe_slug(project.name)}-{digest}"


def _path_is_within(path: Path, parent: Path) -> bool:
    """Return whether a resolved path is equal to or below ``parent``."""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _normalize_provider_filter(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    normalized: dict[str, str] = {}
    for raw in values:
        provider = str(raw).strip()
        if not provider or "/" in provider or "\\" in provider:
            raise ValueError("annotation provider IDs must be non-empty and path-free")
        normalized.setdefault(provider.casefold(), provider)
    return tuple(sorted(normalized.values(), key=lambda item: (item.casefold(), item)))


def _legacy_provider_event_count(project: Path, provider_ids: Sequence[str]) -> int:
    if not provider_ids:
        return 0
    allowed = {value.casefold() for value in provider_ids}
    try:
        payload = json.loads(
            (project / ".state" / "canonical-project.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 0
    events = payload.get("visual_events", []) if isinstance(payload, Mapping) else []
    return sum(
        1
        for event in events
        if isinstance(event, Mapping)
        and str(event.get("annotation_provider") or "").casefold() in allowed
    )


def _record_callback(
    callback: Callable[[Mapping[str, Any]], None] | None,
    record: Mapping[str, Any],
) -> None:
    if callback is None:
        return
    callback(record)


def create_review_bundles(
    projects_root: str | Path,
    *,
    output_root: str | Path | None = None,
    max_packets_per_project: int = DEFAULT_REVIEW_MAX_PACKETS,
    min_free_bytes: int = DEFAULT_REVIEW_MIN_FREE_BYTES,
    max_bundle_bytes: int = DEFAULT_REVIEW_MAX_BUNDLE_BYTES,
    continue_on_error: bool = True,
    dry_run: bool = False,
    include_annotation_providers: Sequence[str] = (),
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Create bounded review handoffs for pending or explicitly filtered projects.

    The operation is deterministic and resumable: projects are sorted by
    relative path, existing semantic observations are not re-opened unless an
    exact provider filter is explicitly supplied, and a repeated call produces
    the same content-addressed bundle directory for an unchanged revision.
    ``max_bundle_bytes`` is a conservative preflight budget sized to the
    smaller of the per-project cap and the project's current work frontier;
    the returned ``bundle_bytes`` records the actual JSON footprint after
    creation.
    """

    if max_packets_per_project <= 0:
        raise ValueError("max_packets_per_project must be positive")
    if min_free_bytes < 0:
        raise ValueError("min_free_bytes must be non-negative")
    if max_bundle_bytes <= 0:
        raise ValueError("max_bundle_bytes must be positive")
    provider_filter = _normalize_provider_filter(include_annotation_providers)
    corpus_root = Path(projects_root).expanduser().resolve(strict=True)
    projects = discover_review_projects(corpus_root)
    destination_root: Path | None = None
    if output_root is not None:
        destination_root = Path(output_root).expanduser().resolve()
        for project in projects:
            if _path_is_within(destination_root, project):
                raise InputError(
                    "Review bundle output root must not be inside a canonical project: "
                    + str(project)
                )
        destination_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    blocked = False
    for project in projects:
        record: dict[str, Any]
        destination = _project_output_dir(project, corpus_root, destination_root)
        usage_root = destination_root or project
        pending_before: int | None = None
        try:
            pending_before = pending_packet_count(project)
        except Exception as exc:
            record = {
                "project_dir": str(project),
                "status": "error",
                "error": str(exc),
                "pending_before": None,
            }
            blocked = True
            records.append(record)
            _record_callback(progress_callback, record)
            if not continue_on_error:
                blocked = True
                break
            continue

        legacy_before = _legacy_provider_event_count(project, provider_filter)
        no_work = legacy_before == 0 if provider_filter else pending_before == 0
        if no_work:
            record = {
                "project_dir": str(project),
                "status": (
                    "skipped_no_matching_provider_events"
                    if provider_filter
                    else "skipped_no_pending_packets"
                ),
                "pending_before": pending_before,
                "estimated_bundle_bytes": 0,
                "bundle_bytes": 0,
                "copied_media_bytes": 0,
                "legacy_provider_event_count": legacy_before,
            }
            records.append(record)
            _record_callback(progress_callback, record)
            continue

        # Reserve only what this project can actually select.  A one-packet
        # frontier should not be blocked by the full per-project cap; explicit
        # provider re-review uses its matching event count instead of the
        # ordinary pending frontier count.
        available_work = legacy_before if provider_filter else int(pending_before or 0)
        estimated_packet_count = min(max_packets_per_project, max(1, available_work))
        estimate = estimate_review_bundle_bytes(estimated_packet_count)

        if destination is not None and destination.is_symlink():
            raise InputError(f"Review bundle output path must not be a symlink: {destination}")
        bundle_usage_root = (
            destination
            if destination is not None
            else project / ".state" / "vision" / "subagent-review"
        )
        if bundle_usage_root.is_symlink():
            raise InputError(f"Review bundle output path must not be a symlink: {bundle_usage_root}")
        existing_bytes = _tree_bytes(bundle_usage_root)
        if existing_bytes > max_bundle_bytes:
            blocked = True
            record = {
                "project_dir": str(project),
                "output_dir": str(destination) if destination is not None else None,
                "status": "bundle_budget_blocked",
                "pending_before": pending_before,
                "estimated_bundle_bytes": estimate,
                "estimated_packet_count": estimated_packet_count,
                "existing_bundle_bytes": existing_bytes,
                "max_bundle_bytes": max_bundle_bytes,
                "copied_media_bytes": 0,
                "legacy_provider_event_count": legacy_before,
            }
            records.append(record)
            _record_callback(progress_callback, record)
            if not continue_on_error:
                break
            continue
        dynamic_actual_preflight = (
            not dry_run
            and estimate > max_bundle_bytes
            and max_bundle_bytes >= _DYNAMIC_PREFLIGHT_MIN_BYTES
            and destination is not None
            and not destination.exists()
        )
        if estimate > max_bundle_bytes and not dynamic_actual_preflight:
            blocked = True
            record = {
                "project_dir": str(project),
                "output_dir": str(destination) if destination is not None else None,
                "status": "bundle_budget_blocked",
                "pending_before": pending_before,
                "estimated_bundle_bytes": estimate,
                "estimated_packet_count": estimated_packet_count,
                "existing_bundle_bytes": existing_bytes,
                "max_bundle_bytes": max_bundle_bytes,
                "copied_media_bytes": 0,
                "legacy_provider_event_count": legacy_before,
            }
            records.append(record)
            _record_callback(progress_callback, record)
            if not continue_on_error:
                break
            continue
        free_bytes = int(shutil.disk_usage(usage_root).free)
        required_bytes = max(0, estimate - existing_bytes) + min_free_bytes
        base_record: dict[str, Any] = {
            "project_dir": str(project),
            "output_dir": str(destination) if destination is not None else None,
            "pending_before": pending_before,
            "estimated_bundle_bytes": estimate,
            "estimated_packet_count": estimated_packet_count,
            "existing_bundle_bytes": existing_bytes,
            "free_bytes": free_bytes,
            "required_bytes": required_bytes,
            "fits_storage_policy": free_bytes >= required_bytes,
            "copied_media_bytes": 0,
            "legacy_provider_event_count": legacy_before,
            "include_annotation_providers": list(provider_filter),
        }
        if free_bytes < required_bytes:
            blocked = True
            record = {**base_record, "status": "storage_blocked"}
            records.append(record)
            _record_callback(progress_callback, record)
            if not continue_on_error:
                break
            continue
        if dry_run:
            record = {**base_record, "status": "planned", "bundle_bytes": 0}
            records.append(record)
            _record_callback(progress_callback, record)
            continue

        staging_dir: Path | None = None
        try:
            create_output = destination
            if dynamic_actual_preflight:
                assert destination is not None
                staging_dir = Path(
                    tempfile.mkdtemp(
                        prefix=f".{destination.name}.preflight-",
                        dir=str(destination.parent),
                    )
                )
                create_output = staging_dir
            bundle_kwargs: dict[str, Any] = {
                "output_dir": create_output,
                "max_packets": max_packets_per_project,
            }
            if provider_filter:
                bundle_kwargs["include_annotation_providers"] = provider_filter
            result = create_review_bundle(project, **bundle_kwargs)
            bundle_dir = Path(str(result["bundle_dir"])).expanduser().resolve()
            bundle_bytes = _tree_bytes(bundle_dir)
            if dynamic_actual_preflight and bundle_bytes > max_bundle_bytes:
                raise _MeasuredBundleBudgetExceeded(bundle_bytes, max_bundle_bytes)
            if staging_dir is not None:
                assert destination is not None
                staging_dir.replace(destination)
                bundle_dir = destination.resolve()
            record = {
                **base_record,
                "status": "created" if int(result.get("request_count", 0)) else "empty",
                "bundle_dir": str(bundle_dir),
                "bundle_id": result.get("bundle_id"),
                "request_count": int(result.get("request_count", 0)),
                "deferred_count": len(result.get("deferred_event_ids", [])),
                "bundle_bytes": bundle_bytes,
                "bundle_over_budget": bundle_bytes > max_bundle_bytes,
                "actual_preflight": dynamic_actual_preflight,
                "legacy_provider_event_count": legacy_before,
            }
        except _MeasuredBundleBudgetExceeded as exc:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            record = {
                **base_record,
                "status": "bundle_budget_blocked",
                "measured_bundle_bytes": exc.actual_bytes,
                "bundle_over_budget": True,
                "actual_preflight": True,
            }
        except Exception as exc:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            record = {**base_record, "status": "error", "error": str(exc)}
            blocked = True
        records.append(record)
        _record_callback(progress_callback, record)
        if record.get("status") == "error" and not continue_on_error:
            break

    return {
        "projects_root": str(corpus_root),
        "output_root": str(destination_root) if destination_root is not None else None,
        "project_count": len(projects),
        "processed_count": len(records),
        "created_count": sum(1 for item in records if item.get("status") == "created"),
        "skipped_count": sum(
            1 for item in records if item.get("status") == "skipped_no_pending_packets"
        ),
        "blocked": blocked,
        "policy": {
            "max_packets_per_project": max_packets_per_project,
            "min_free_bytes": min_free_bytes,
            "max_bundle_bytes": max_bundle_bytes,
            "media_copy_policy": "reference_only",
            "include_annotation_providers": list(provider_filter),
        },
        "projects": records,
    }


__all__ = [
    "DEFAULT_REVIEW_MAX_BUNDLE_BYTES",
    "DEFAULT_REVIEW_MAX_PACKETS",
    "DEFAULT_REVIEW_MIN_FREE_BYTES",
    "create_review_bundles",
    "discover_review_projects",
    "estimate_review_bundle_bytes",
]
