from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import Any

from .errors import InputError
from .security import atomic_write_json, ensure_contained


def _validated_cache_files(root: Path) -> list[tuple[Path, int]]:
    """Inventory regular cache files with one non-following stat per entry.

    Compaction runs after a successful commit and may inspect hundreds of raw
    PNGs.  ``Path.rglob`` followed by ``is_symlink``/``is_file``/``stat``
    performs several metadata calls per entry.  A single ``scandir`` recursion
    preserves the safety contract while reducing metadata overhead.  Any
    symlink below a compaction target is rejected rather than followed.
    """

    if root.is_symlink():
        raise InputError(f"Refusing to inspect a symlinked cache target: {root}")
    files: list[tuple[Path, int]] = []

    def visit(directory: str) -> None:
        try:
            entries = os.scandir(directory)
        except OSError:
            # ``os.walk``'s default error handling skipped an inaccessible
            # subtree.  Keep that behavior for this best-effort compaction
            # inventory while still failing on entries that can be inspected
            # but cannot be stat'ed safely.
            return
        with entries:
            for entry in sorted(entries, key=lambda item: item.name.casefold()):
                path = Path(entry.path)
                try:
                    stat_result = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise InputError(
                        f"Unable to inspect generated cache artifact: {path}"
                    ) from exc
                mode = stat_result.st_mode
                if S_ISLNK(mode):
                    raise InputError(f"Refusing to compact a cache containing a symlink: {path}")
                if S_ISREG(mode):
                    files.append((path, int(stat_result.st_size)))
                elif S_ISDIR(mode):
                    # Recurse only into real directories. Other special
                    # entries remain outside the regular-file inventory.
                    visit(entry.path)

    visit(os.fspath(root))
    return files


def cache_key(*parts: Any) -> str:
    encoded = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class StageCache:
    def __init__(self, project_root: Path) -> None:
        self.root = project_root / ".state" / "cache"
        self.root.mkdir(parents=True, exist_ok=True)

    def marker(self, stage: str) -> Path:
        return self.root / f"{stage}.json"

    def matches(self, stage: str, key: str) -> bool:
        path = self.marker(stage)
        if not path.exists():
            return False
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return isinstance(loaded, dict) and loaded.get("cache_key") == key
        except (OSError, json.JSONDecodeError):
            return False

    def commit(self, stage: str, key: str, outputs: list[str]) -> None:
        atomic_write_json(self.marker(stage), {"cache_key": key, "outputs": sorted(outputs)})

    def invalidate_from(self, ordered_stages: list[str], stage: str) -> list[str]:
        if stage not in ordered_stages:
            raise ValueError(f"Unknown stage: {stage}")
        removed: list[str] = []
        for name in ordered_stages[ordered_stages.index(stage) :]:
            marker = self.marker(name)
            if marker.exists():
                marker.unlink()
                removed.append(name)
        return removed


def purge_project_cache(project_root: Path) -> int:
    """Remove only the cache belonging to a recognizable project package.

    The canonical-state marker and containment checks prevent this helper from
    becoming a general recursive-delete primitive.  Symlinked cache paths are
    rejected by :func:`ensure_contained`.
    """

    root = project_root.expanduser().resolve(strict=True)
    canonical = root / ".state" / "canonical-project.json"
    if not canonical.is_file():
        raise InputError(f"Not a reconstruction project (canonical state missing): {root}")
    cache_dir = root / ".state" / "cache"
    visual_frame_dir = root / ".state" / "checkpoints" / "visual-frames"
    ocr_dir = root / ".state" / "checkpoints" / "ocr"
    visual_survey_file = root / ".state" / "checkpoints" / "visual-survey.json"
    visual_structural_survey_file = (
        root / ".state" / "checkpoints" / "visual-survey-structural.json"
    )
    removed = 0
    for target in (
        cache_dir,
        visual_frame_dir,
        ocr_dir,
        visual_survey_file,
        visual_structural_survey_file,
    ):
        if target.is_symlink():
            raise InputError(f"Refusing to purge a symlinked cache target: {target}")
        if not target.exists():
            continue
        validated = ensure_contained(root, target, allow_missing=False)
        if validated != target.resolve(strict=True):
            raise InputError(f"Refusing to purge an unexpected cache target: {target}")
        if validated.is_dir():
            removed += sum(1 for path in validated.rglob("*") if path.is_file())
            shutil.rmtree(validated)
            if validated == cache_dir:
                validated.mkdir(parents=True)
        else:
            removed += 1
            validated.unlink()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return removed


def compact_completed_checkpoints(project_root: Path, *, keep: bool = False) -> dict[str, Any]:
    """Remove only resumable visual caches after a successful project commit.

    A completed project already has canonical evidence images, metadata, and
    transcript checkpoints.  The raw visual-frame/OCR caches are useful while
    a stage is interrupted, but retaining them indefinitely duplicates a large
    fraction of the output.  The small measured visual-survey markers are kept:
    it avoids repeating the expensive hard-cut/2-fps detector on a downstream
    rebuild while containing no pixels.  This helper is deliberately narrower
    than :func:`purge_project_cache`: ASR, repair checkpoints, and the survey
    marker are never touched, and callers can opt out with ``keep=True`` when
    iterating on a downstream visual configuration.

    The returned accounting is suitable for the run manifest and makes the
    cleanup observable rather than silently changing artifact usage.
    """

    root = project_root.expanduser().resolve(strict=True)
    canonical = root / ".state" / "canonical-project.json"
    if not canonical.is_file():
        raise InputError(f"Not a reconstruction project (canonical state missing): {root}")
    targets = (
        root / ".state" / "cache",
        root / ".state" / "checkpoints" / "visual-frames",
        root / ".state" / "checkpoints" / "ocr",
    )
    if keep:
        return {
            "kept": True,
            "removed_files": 0,
            "reclaimed_bytes": 0,
            "targets": [],
        }

    removed_files = 0
    reclaimed_bytes = 0
    removed_targets: list[str] = []
    for target in targets:
        if not target.exists() and not target.is_symlink():
            continue
        if target.is_symlink():
            raise InputError(f"Refusing to compact a symlinked cache target: {target}")
        validated = ensure_contained(root, target, allow_missing=False)
        if validated != target.resolve(strict=True):
            raise InputError(f"Refusing to compact an unexpected cache target: {target}")
        if validated.is_dir():
            cache_files = _validated_cache_files(validated)
            removed_files += len(cache_files)
            reclaimed_bytes += sum(size for _path, size in cache_files)
            shutil.rmtree(validated)
            if validated == root / ".state" / "cache":
                validated.mkdir(parents=True, exist_ok=True)
        else:
            try:
                reclaimed_bytes += validated.stat().st_size
            except OSError as exc:
                raise InputError(f"Unable to stat generated cache artifact: {validated}") from exc
            removed_files += 1
            validated.unlink()
        removed_targets.append(str(target.relative_to(root).as_posix()))
    return {
        "kept": False,
        "removed_files": removed_files,
        "reclaimed_bytes": reclaimed_bytes,
        "targets": removed_targets,
    }


def completed_checkpoint_compaction_plan(project_root: Path) -> dict[str, Any]:
    """Return a read-only plan for compacting duplicate visual/OCR checkpoints.

    The plan uses the same containment and symlink checks as the mutating
    compactor, but never removes anything.  Keeping this as a separate public
    operation lets the CLI show an exact reclaim estimate before an explicit
    ``--apply`` while leaving transcript/ASR checkpoints outside the target
    set.
    """

    root = project_root.expanduser().resolve(strict=True)
    canonical = root / ".state" / "canonical-project.json"
    if not canonical.is_file():
        raise InputError(f"Not a reconstruction project (canonical state missing): {root}")
    targets = (
        root / ".state" / "cache",
        root / ".state" / "checkpoints" / "visual-frames",
        root / ".state" / "checkpoints" / "ocr",
    )
    removed_files = 0
    reclaimed_bytes = 0
    planned_targets: list[str] = []
    for target in targets:
        if not target.exists() and not target.is_symlink():
            continue
        if target.is_symlink():
            raise InputError(f"Refusing to inspect a symlinked cache target: {target}")
        validated = ensure_contained(root, target, allow_missing=False)
        if validated != target.resolve(strict=True):
            raise InputError(f"Refusing to inspect an unexpected cache target: {target}")
        if validated.is_dir():
            cache_files = _validated_cache_files(validated)
            removed_files += len(cache_files)
            reclaimed_bytes += sum(size for _path, size in cache_files)
        else:
            try:
                reclaimed_bytes += validated.stat().st_size
            except OSError as exc:
                raise InputError(f"Unable to stat generated cache artifact: {validated}") from exc
            removed_files += 1
        planned_targets.append(str(target.relative_to(root).as_posix()))
    return {
        "dry_run": True,
        "kept": False,
        "removed_files": removed_files,
        "reclaimed_bytes": reclaimed_bytes,
        "targets": planned_targets,
    }


__all__ = [
    "StageCache",
    "cache_key",
    "compact_completed_checkpoints",
    "completed_checkpoint_compaction_plan",
    "purge_project_cache",
]
