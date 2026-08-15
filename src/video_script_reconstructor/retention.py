"""Safe reporting and retention for generated reconstruction runs.

The retention boundary is intentionally the output-root/project directory, not
the user's workspace. Only directories carrying a canonical project marker are
eligible, and pruning is dry-run by default. Source media, models, tests, and
arbitrary neighboring folders are never candidates.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import Any

from .errors import InputError
from .security import ensure_contained


@dataclass(frozen=True)
class RetentionProject:
    path: str
    run_id: str | None
    written_at_utc: str | None
    source_reference: str | None
    file_count: int
    bytes: int
    reclaimable_file_count: int
    reclaimable_bytes: int


@dataclass(frozen=True)
class RetentionOrphan:
    """A generated-looking tree that lacks a canonical project marker.

    Orphans are report-only.  They are intentionally never eligible for
    ``prune_runs`` because a missing canonical marker means the retention
    contract cannot prove ownership of the files.
    """

    path: str
    reason: str
    markers: tuple[str, ...]
    file_count: int
    bytes: int


_ORPHAN_MARKERS: tuple[tuple[str, ...], ...] = (
    (".state", "cache"),
    (".state", "checkpoints"),
    (".state", "timeline"),
    (".state", "vision"),
    ("evidence", "full"),
    ("evidence", "crops"),
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"Unable to read generated project state: {path}") from exc
    if not isinstance(value, dict):
        raise InputError(f"Generated project state is not an object: {path}")
    return value


def _directory_usage(project: Path) -> tuple[int, int, int, int]:
    """Return total and reclaimable usage for one marked project.

    The retention report may size dozens of projects, so avoid the multiple
    metadata syscalls that ``Path.rglob`` + ``is_symlink`` + ``is_file``
    otherwise performs for each entry.  ``os.scandir`` gives us one directory
    listing and one non-following ``DirEntry.stat`` per entry while retaining
    the safety boundary: symlinks are still rejected rather than followed or
    silently skipped.
    """

    total_bytes = 0
    file_count = 0
    reclaimable_bytes = 0
    reclaimable_file_count = 0

    def visit(directory: str) -> None:
        nonlocal file_count, total_bytes, reclaimable_file_count, reclaimable_bytes
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            raise InputError(f"Unable to inspect generated project: {project}") from exc
        with entries:
            for entry in sorted(entries, key=lambda item: item.name.casefold()):
                path = Path(entry.path)
                try:
                    stat = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise InputError(f"Unable to stat generated artifact: {path}") from exc
                mode = stat.st_mode
                if S_ISLNK(mode):
                    raise InputError(
                        f"Refusing to inspect symlink inside generated project: {path}"
                    )
                if S_ISDIR(mode):
                    visit(entry.path)
                    continue
                if not S_ISREG(mode):
                    continue
                size = int(stat.st_size)
                file_count += 1
                total_bytes += size
                relative_parts = path.relative_to(project).parts
                reclaimable = relative_parts[:2] == (".state", "cache") or relative_parts[:3] in {
                    (".state", "checkpoints", "visual-frames"),
                    (".state", "checkpoints", "ocr"),
                }
                if reclaimable:
                    reclaimable_file_count += 1
                    reclaimable_bytes += size

    visit(os.fspath(project))
    return file_count, total_bytes, reclaimable_file_count, reclaimable_bytes


def _project_record(path: Path, *, root: Path) -> RetentionProject:
    canonical = path / ".state" / "canonical-project.json"
    if not canonical.is_file():
        raise InputError(f"Not a reconstruction project (canonical state missing): {path}")
    ensure_contained(root, path, allow_missing=False)
    state = _read_json(canonical)
    manifest_path = path / ".state" / "run-manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    written_at = manifest.get("written_at_utc") or state.get("generated_at_utc")
    if written_at is not None and not isinstance(written_at, str):
        written_at = str(written_at)
    input_reference = state.get("input_reference")
    source_reference = str(input_reference) if input_reference is not None else None
    file_count, total_bytes, reclaimable_file_count, reclaimable_bytes = _directory_usage(path)
    return RetentionProject(
        path=str(path.resolve()),
        run_id=str(manifest["run_id"]) if manifest.get("run_id") is not None else None,
        written_at_utc=written_at,
        source_reference=source_reference,
        file_count=file_count,
        bytes=total_bytes,
        reclaimable_file_count=reclaimable_file_count,
        reclaimable_bytes=reclaimable_bytes,
    )


def _directory_tree(root: Path) -> tuple[Path, ...]:
    """Return real directories below ``root`` without following symlinks.

    Retention discovery only needs directory markers, but ``Path.rglob``
    still allocates and filters every file in a large generated tree.  A
    deterministic ``scandir`` walk lets us inspect each directory entry once
    and skip linked/special entries before descending.
    """

    directories: list[Path] = []

    def visit(directory: str) -> None:
        try:
            entries = os.scandir(directory)
        except OSError:
            return
        children: list[Path] = []
        with entries:
            for entry in entries:
                try:
                    is_directory = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if not is_directory:
                    continue
                children.append(Path(entry.path))
        # Callers sort their externally visible records.  Avoid sorting every
        # directory here: on large artifact roots that turns marker discovery
        # into an unnecessary per-directory allocation/sort pass.
        for child in children:
            directories.append(child)
            visit(os.fspath(child))

    visit(os.fspath(root))
    return tuple(directories)


def _sort_key(item: RetentionProject) -> tuple[datetime, str]:
    value = item.written_at_utc
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = datetime.min
    else:
        parsed = datetime.min
    return parsed, item.path


def _discover_project_paths(target: Path) -> tuple[Path, ...]:
    """Find canonical project roots without reading or sizing their files."""

    canonical = target / ".state" / "canonical-project.json"
    if canonical.is_file():
        return (target,)
    if not target.is_dir():
        raise InputError(f"Retention root is not a directory: {target}")
    projects: list[Path] = []
    seen: set[Path] = set()
    for state_dir in _directory_tree(target):
        if state_dir.name != ".state":
            continue
        marker = state_dir / "canonical-project.json"
        if marker.is_symlink() or not marker.is_file():
            continue
        project = marker.parent.parent
        if project in seen or project.is_symlink() or not project.is_dir():
            continue
        seen.add(project)
        projects.append(project)
    return tuple(projects)


def discover_projects(root: str | Path) -> tuple[RetentionProject, ...]:
    """Discover generated projects marked by a canonical state file.

    Output collections may contain named run groups (for example
    ``outputs/public/<project>``), so direct-child discovery can silently report
    zero bytes while leaving large generated trees unmanaged.  Recursing only
    through the canonical ``.state/canonical-project.json`` marker keeps the
    boundary explicit: arbitrary folders, source media, and model directories
    remain ineligible.
    """

    target = Path(root).expanduser().resolve(strict=True)
    single_project = (target / ".state" / "canonical-project.json").is_file()
    projects = tuple(
        _project_record(project, root=target.parent if single_project else target)
        for project in _discover_project_paths(target)
    )
    return tuple(sorted(projects, key=_sort_key, reverse=True))


def _orphan_marker_names(project: Path) -> tuple[str, ...]:
    markers: list[str] = []
    for relative in _ORPHAN_MARKERS:
        marker = project.joinpath(*relative)
        if marker.is_symlink():
            raise InputError(f"Refusing to inspect symlink inside orphan candidate: {marker}")
        if marker.exists():
            markers.append("/".join(relative))
    return tuple(markers)


def _discover_orphans(
    target: Path,
    recognized_projects: tuple[Path, ...],
) -> tuple[RetentionOrphan, ...]:
    """Discover report-only generated footprints using a known project set.

    ``retention_report`` already has to discover and size every canonical
    project.  Accepting those paths here avoids a second recursive canonical
    search and a second full usage walk for the same projects.  The public
    ``discover_orphans`` wrapper below only needs paths, so orphan-only
    commands do not pay for canonical project sizing at all.
    """

    recognized = tuple(path.resolve() for path in recognized_projects)
    candidates: list[RetentionOrphan] = []
    seen: set[Path] = set()
    for state_dir in _directory_tree(target):
        if state_dir.name != ".state":
            continue
        candidate = state_dir.parent.resolve()
        if any(candidate == existing or candidate.is_relative_to(existing) for existing in seen):
            continue
        if any(
            candidate == project or project.is_relative_to(candidate) for project in recognized
        ):
            continue
        canonical = state_dir / "canonical-project.json"
        if canonical.is_file():
            continue
        markers = _orphan_marker_names(candidate)
        if not markers:
            continue
        ensure_contained(target, candidate, allow_missing=False)
        file_count, total_bytes, _reclaimable_file_count, _reclaimable_bytes = _directory_usage(
            candidate
        )
        seen.add(candidate)
        candidates.append(
            RetentionOrphan(
                path=str(candidate),
                reason="generated footprint without canonical project marker",
                markers=markers,
                file_count=file_count,
                bytes=total_bytes,
            )
        )
    return tuple(sorted(candidates, key=lambda item: item.path.casefold()))


def discover_orphans(root: str | Path) -> tuple[RetentionOrphan, ...]:
    """Find generated-looking incomplete trees without making them deletable.

    A candidate must contain one of the known reconstruction footprints and
    must not contain ``.state/canonical-project.json``.  The conservative
    marker rule keeps arbitrary source, model, and notes directories outside
    the report.  Candidates are informational only; callers must inspect and
    remove them explicitly if ownership is certain.
    """

    target = Path(root).expanduser().resolve(strict=True)
    if not target.is_dir():
        raise InputError(f"Retention root is not a directory: {target}")
    return _discover_orphans(target, _discover_project_paths(target))


def _unclassified_usage(root: Path, covered_roots: tuple[Path, ...]) -> tuple[int, int]:
    """Count files outside marked projects/orphans without making them targets.

    This is deliberately a reporting-only inventory. A caller may point the
    command at a collection containing profiles, source media, models, or
    arbitrary notes; those bytes are surfaced as unclassified and remain out
    of every prune candidate set.
    """

    file_count = 0
    total_bytes = 0
    covered = tuple(path.resolve() for path in covered_roots)
    covered_set = frozenset(covered)
    # A single-project retention report can be rooted at the project itself.
    # In that case every file is already accounted for by the project walk.
    if root.resolve() in covered_set:
        return 0, 0

    # Prune canonical projects and recognized orphan trees before walking their
    # files. They were already sized by their owning discovery pass;
    # traversing them again only to skip every file made large retention
    # reports needlessly expensive.
    def visit(directory: str) -> None:
        nonlocal file_count, total_bytes
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            raise InputError(f"Unable to inspect retention root: {root}") from exc
        with entries:
            for entry in sorted(entries, key=lambda item: item.name.casefold()):
                path = Path(entry.path)
                try:
                    stat = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise InputError(
                        f"Unable to inspect unclassified output artifact: {path}"
                    ) from exc
                mode = stat.st_mode
                # Covered roots and symlinks are intentionally skipped. The
                # unclassified report must not follow links or count files in
                # trees already accounted for by another report.
                if S_ISLNK(mode) or path in covered_set:
                    continue
                if S_ISDIR(mode):
                    visit(entry.path)
                    continue
                if not S_ISREG(mode):
                    continue
                # Covered roots are pruned above, so a regular file reached
                # here cannot be inside a classified project/orphan tree.
                total_bytes += int(stat.st_size)
                file_count += 1

    visit(os.fspath(root))
    return file_count, total_bytes


def retention_report(root: str | Path) -> dict[str, Any]:
    """Return deterministic usage for marked runs and unmarked footprints.

    Canonical projects remain the only automatic-prune candidates.  The
    additional orphan totals make the read-only report honest about generated
    benchmark/profile trees that were interrupted before a canonical commit;
    callers must still use ``prune-orphans --apply`` explicitly for those.
    """

    target = Path(root).expanduser().resolve(strict=True)
    projects = discover_projects(target)
    # Reuse the canonical project inventory; rediscovering it inside
    # ``discover_orphans`` would rescan every project tree a second time.
    orphans = _discover_orphans(
        target, tuple(Path(item.path).resolve() for item in projects)
    )
    orphan_files = sum(item.file_count for item in orphans)
    orphan_bytes = sum(item.bytes for item in orphans)
    project_files = sum(item.file_count for item in projects)
    project_bytes = sum(item.bytes for item in projects)
    covered_roots = tuple(Path(item.path).resolve() for item in projects) + tuple(
        Path(item.path).resolve() for item in orphans
    )
    unclassified_files, unclassified_bytes = _unclassified_usage(target, covered_roots)
    return {
        "schema_version": "1.0",
        "root": str(target),
        "project_count": len(projects),
        # Backward-compatible fields: these cover only canonical projects and
        # therefore remain the exact set eligible for ``retention prune``.
        "total_files": project_files,
        "total_bytes": project_bytes,
        "reclaimable_files": sum(item.reclaimable_file_count for item in projects),
        "reclaimable_bytes": sum(item.reclaimable_bytes for item in projects),
        # Generated-looking but unmarked trees are visible, never silently
        # counted as canonical projects, and never implicit prune candidates.
        "orphan_count": len(orphans),
        "orphan_files": orphan_files,
        "orphan_bytes": orphan_bytes,
        "observed_generated_files": project_files + orphan_files,
        "observed_generated_bytes": project_bytes + orphan_bytes,
        "unclassified_files": unclassified_files,
        "unclassified_bytes": unclassified_bytes,
        "observed_root_files": project_files + orphan_files + unclassified_files,
        "observed_root_bytes": project_bytes + orphan_bytes + unclassified_bytes,
        "projects": [asdict(item) for item in projects],
    }


def orphan_report(root: str | Path) -> dict[str, Any]:
    """Return a read-only report of unmarked generated-looking trees."""

    target = Path(root).expanduser().resolve(strict=True)
    orphans = discover_orphans(target)
    orphan_payloads = []
    for item in orphans:
        payload = asdict(item)
        payload["markers"] = list(item.markers)
        orphan_payloads.append(payload)
    return {
        "schema_version": "1.0",
        "root": str(target),
        "orphan_count": len(orphans),
        "orphan_files": sum(item.file_count for item in orphans),
        "orphan_bytes": sum(item.bytes for item in orphans),
        "orphans": orphan_payloads,
    }


def prune_runs(root: str | Path, *, keep: int = 1, apply: bool = False) -> dict[str, Any]:
    """Plan or apply deletion of the oldest recognized generated projects.

    ``apply=False`` is deliberate: a plain CLI invocation is an auditable
    dry-run. When applying, only direct children of an output collection are
    removed; passing a single project directory never permits deleting that
    project root itself.
    """

    if keep < 0:
        raise InputError("retention --keep must be zero or greater")
    target = Path(root).expanduser().resolve(strict=True)
    single_project = (target / ".state" / "canonical-project.json").is_file()
    projects = list(discover_projects(target))
    if single_project and keep == 0:
        raise InputError("Refusing to delete a project root; pass its parent output directory")
    removable = projects[keep:] if not single_project else []
    entries = [asdict(item) for item in removable]
    reclaimed = sum(item.bytes for item in removable)
    removed: list[str] = []
    if apply:
        for item in removable:
            project = Path(item.path)
            ensure_contained(target, project, allow_missing=False)
            if not project.is_dir() or project.is_symlink():
                raise InputError(f"Refusing to prune unexpected retention target: {project}")
            shutil.rmtree(project)
            removed.append(str(project))
    return {
        "schema_version": "1.0",
        "root": str(target),
        "keep": keep,
        "dry_run": not apply,
        "planned_projects": entries,
        "removed_projects": removed,
        "planned_bytes": reclaimed,
        "reclaimed_bytes": reclaimed if apply else 0,
        "remaining_projects": [asdict(item) for item in projects[:keep]],
    }


def prune_orphans(root: str | Path, *, apply: bool = False) -> dict[str, Any]:
    """Plan or apply deletion of explicitly recognized incomplete footprints.

    Orphans are never included in :func:`prune_runs` because their missing
    canonical marker prevents automatic ownership proof. This separate command
    requires an explicit root and remains dry-run by default. Applying it only
    removes candidate directories returned by :func:`discover_orphans`, after
    rechecking containment, symlink status, and the absence of a canonical
    marker immediately before each deletion.
    """

    target = Path(root).expanduser().resolve(strict=True)
    if not target.is_dir():
        raise InputError(f"Retention root is not a directory: {target}")
    orphans = list(discover_orphans(target))
    planned = [asdict(item) for item in orphans]
    planned_bytes = sum(item.bytes for item in orphans)
    removed: list[str] = []
    if apply:
        for item in orphans:
            candidate = Path(item.path).resolve()
            ensure_contained(target, candidate, allow_missing=False)
            if candidate == target or candidate.is_symlink() or not candidate.is_dir():
                raise InputError(f"Refusing to prune unexpected orphan target: {candidate}")
            if (candidate / ".state" / "canonical-project.json").is_file():
                raise InputError(
                    f"Orphan changed into a canonical project during pruning: {candidate}"
                )
            markers = _orphan_marker_names(candidate)
            if not markers:
                raise InputError(f"Orphan markers disappeared before pruning: {candidate}")
            shutil.rmtree(candidate)
            removed.append(str(candidate))
    return {
        "schema_version": "1.0",
        "root": str(target),
        "dry_run": not apply,
        "planned_orphans": planned,
        "removed_orphans": removed,
        "planned_bytes": planned_bytes,
        "reclaimed_bytes": planned_bytes if apply else 0,
    }


__all__ = [
    "RetentionOrphan",
    "RetentionProject",
    "discover_orphans",
    "discover_projects",
    "orphan_report",
    "prune_orphans",
    "prune_runs",
    "retention_report",
]
