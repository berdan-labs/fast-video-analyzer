from __future__ import annotations

import json
from pathlib import Path

import pytest

import video_script_reconstructor.retention as retention_module
from video_script_reconstructor.errors import InputError
from video_script_reconstructor.retention import (
    orphan_report,
    prune_orphans,
    prune_runs,
    retention_report,
)


def _project(root: Path, name: str, written_at: str, size: int) -> Path:
    project = root / name
    state = project / ".state"
    state.mkdir(parents=True)
    (state / "canonical-project.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generated_at_utc": written_at,
                "input_reference": f"C:/media/{name}.mp4",
            }
        ),
        encoding="utf-8",
    )
    (state / "run-manifest.json").write_text(
        json.dumps({"run_id": f"RUN-{name}", "written_at_utc": written_at}),
        encoding="utf-8",
    )
    (project / "artifact.bin").write_bytes(b"x" * size)
    return project


def test_retention_report_is_scoped_and_newest_first(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    _project(root, "old", "2026-08-01T00:00:00Z", 3)
    _project(root, "new", "2026-08-02T00:00:00Z", 7)
    (root / "models").mkdir()
    (root / "models" / "do-not-touch.bin").write_bytes(b"model")
    cache = root / "new" / ".state" / "checkpoints" / "ocr"
    cache.mkdir(parents=True)
    (cache / "checkpoint.json").write_bytes(b"cache")

    report = retention_report(root)

    assert report["project_count"] == 2
    assert report["total_bytes"] == sum(item["bytes"] for item in report["projects"])
    assert report["total_bytes"] > 10
    assert report["reclaimable_bytes"] == 5
    assert report["reclaimable_files"] == 1
    assert Path(report["projects"][0]["path"]).name == "new"
    assert (root / "models" / "do-not-touch.bin").is_file()
    assert report["orphan_count"] == 0
    assert report["observed_generated_bytes"] == report["total_bytes"]
    assert report["unclassified_files"] == 1
    assert report["unclassified_bytes"] == 5
    assert report["observed_root_bytes"] == report["total_bytes"] + 5


def test_retention_report_surfaces_unmarked_generated_footprints(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    _project(root, "complete", "2026-08-01T00:00:00Z", 4)
    orphan = root / "interrupted-profile"
    (orphan / ".state" / "checkpoints").mkdir(parents=True)
    (orphan / "evidence" / "full").mkdir(parents=True)
    (orphan / ".state" / "checkpoints" / "receipt.json").write_bytes(b"cache")
    (orphan / "evidence" / "full" / "frame.png").write_bytes(b"pixels")
    (root / "models").mkdir()
    (root / "models" / "weights.bin").write_bytes(b"model")

    report = retention_report(root)

    assert report["project_count"] == 1
    assert report["orphan_count"] == 1
    assert report["orphan_files"] == 2
    assert report["orphan_bytes"] == 11
    assert report["observed_generated_bytes"] == report["total_bytes"] + 11
    assert (root / "models" / "weights.bin").is_file()
    assert report["unclassified_files"] == 1
    assert report["unclassified_bytes"] == 5
    assert report["observed_root_bytes"] == report["total_bytes"] + 16


def test_retention_report_reuses_canonical_inventory_for_orphan_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    _project(root, "complete", "2026-08-01T00:00:00Z", 4)
    orphan = root / "interrupted"
    (orphan / ".state" / "checkpoints").mkdir(parents=True)
    (orphan / ".state" / "checkpoints" / "receipt.json").write_bytes(b"cache")

    original = retention_module.discover_projects
    calls = 0

    def counted_discovery(path: str | Path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(retention_module, "discover_projects", counted_discovery)
    report = retention_module.retention_report(root)

    assert calls == 1
    assert report["project_count"] == 1
    assert report["orphan_count"] == 1


def test_orphan_report_does_not_size_complete_projects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    _project(root, "complete", "2026-08-01T00:00:00Z", 4)
    orphan = root / "interrupted"
    (orphan / ".state" / "checkpoints").mkdir(parents=True)
    (orphan / ".state" / "checkpoints" / "receipt.json").write_bytes(b"cache")

    original = retention_module._directory_usage
    sized: list[Path] = []

    def counted(path: Path) -> tuple[int, int, int, int]:
        sized.append(path.resolve())
        return original(path)

    monkeypatch.setattr(retention_module, "_directory_usage", counted)
    report = retention_module.orphan_report(root)

    assert report["orphan_count"] == 1
    assert sized == [orphan.resolve()]


def test_unclassified_usage_short_circuits_when_root_is_classified(tmp_path: Path) -> None:
    project = _project(tmp_path, "project", "2026-08-01T00:00:00Z", 4)

    assert retention_module._unclassified_usage(project, (project,)) == (0, 0)


def test_retention_inventory_uses_scandir_not_os_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    _project(root, "project", "2026-08-01T00:00:00Z", 4)
    orphan = root / "interrupted"
    (orphan / ".state" / "checkpoints").mkdir(parents=True)
    (orphan / ".state" / "checkpoints" / "receipt.json").write_bytes(b"cache")
    (root / "notes").mkdir()
    (root / "notes" / "keep.txt").write_bytes(b"notes")

    def fail_walk(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retention inventory must recurse with os.scandir")

    def fail_rglob(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retention discovery must not scan with Path.rglob")

    monkeypatch.setattr(retention_module.os, "walk", fail_walk)
    monkeypatch.setattr(retention_module.Path, "rglob", fail_rglob)

    report = retention_module.retention_report(root)

    assert report["project_count"] == 1
    assert report["orphan_count"] == 1
    assert report["unclassified_files"] == 1


def test_retention_inventory_rejects_symlinked_entries(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    project = _project(root, "project", "2026-08-01T00:00:00Z", 4)
    target = project / "target.bin"
    target.write_bytes(b"target")
    link = project / "link.bin"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(InputError, match="symlink"):
        retention_report(root)


def test_prune_is_dry_run_by_default_and_removes_only_old_projects(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    old = _project(root, "old", "2026-08-01T00:00:00Z", 4)
    new = _project(root, "new", "2026-08-02T00:00:00Z", 4)
    unrelated = root / "notes"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")

    planned = prune_runs(root, keep=1)
    assert planned["dry_run"] is True
    assert Path(planned["planned_projects"][0]["path"]) == old.resolve()
    assert old.is_dir() and new.is_dir()

    applied = prune_runs(root, keep=1, apply=True)
    assert applied["dry_run"] is False
    assert applied["removed_projects"] == [str(old.resolve())]
    assert not old.exists()
    assert new.is_dir()
    assert (unrelated / "keep.txt").is_file()


def test_single_project_cannot_be_deleted_by_retention(tmp_path: Path) -> None:
    project = _project(tmp_path, "project", "2026-08-01T00:00:00Z", 1)
    with pytest.raises(InputError, match="project root"):
        prune_runs(project, keep=0, apply=True)


def test_retention_discovers_projects_nested_in_named_run_groups(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    project = root / "public" / "lecture"
    project.parent.mkdir(parents=True)
    _project(root / "public", "lecture", "2026-08-03T00:00:00Z", 9)
    (root / "public" / "models").mkdir()
    (root / "public" / "models" / "keep.bin").write_bytes(b"model")

    report = retention_report(root)

    assert report["project_count"] == 1
    assert Path(report["projects"][0]["path"]) == project.resolve()
    assert report["total_bytes"] >= 9


def test_nested_retention_prune_removes_only_marked_project(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    old = root / "runs" / "old"
    new = root / "runs" / "new"
    _project(root / "runs", "old", "2026-08-01T00:00:00Z", 4)
    _project(root / "runs", "new", "2026-08-02T00:00:00Z", 4)
    unrelated = root / "runs" / "notes"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")

    planned = prune_runs(root, keep=1)
    assert planned["dry_run"] is True
    assert Path(planned["planned_projects"][0]["path"]) == old.resolve()

    applied = prune_runs(root, keep=1, apply=True)
    assert applied["removed_projects"] == [str(old.resolve())]
    assert not old.exists()
    assert new.is_dir()
    assert (unrelated / "keep.txt").is_file()


def test_orphan_report_finds_incomplete_generated_tree_without_touching_neighbors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    _project(root, "complete", "2026-08-04T00:00:00Z", 2)
    orphan = root / "incomplete-profile"
    (orphan / ".state" / "vision").mkdir(parents=True)
    (orphan / "evidence" / "full").mkdir(parents=True)
    (orphan / ".state" / "vision" / "receipt.json").write_text("{}", encoding="utf-8")
    (orphan / "evidence" / "full" / "frame.png").write_bytes(b"pixels")
    models = root / "models"
    models.mkdir()
    (models / "weights.bin").write_bytes(b"model")
    notes = root / "notes"
    notes.mkdir()
    (notes / "keep.txt").write_text("keep", encoding="utf-8")

    report = orphan_report(root)

    assert report["orphan_count"] == 1
    assert report["orphan_files"] == 2
    assert report["orphan_bytes"] == 8
    item = report["orphans"][0]
    assert Path(item["path"]) == orphan.resolve()
    assert item["markers"] == [".state/vision", "evidence/full"]
    assert Path(report["root"]) == root.resolve()
    assert (root / "models" / "weights.bin").is_file()
    assert (root / "notes" / "keep.txt").is_file()


def test_orphan_report_does_not_duplicate_nested_candidates(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    parent = root / "profile"
    (parent / ".state" / "checkpoints" / "visual-frames").mkdir(parents=True)
    child = parent / "nested"
    (child / ".state" / "vision").mkdir(parents=True)
    (parent / "artifact.bin").write_bytes(b"parent")
    (child / "artifact.bin").write_bytes(b"child")

    report = orphan_report(root)

    assert report["orphan_count"] == 1
    assert Path(report["orphans"][0]["path"]) == parent.resolve()


def test_prune_orphans_is_dry_run_by_default_and_strictly_scoped(tmp_path: Path) -> None:
    root = tmp_path / "benchmark-root"
    root.mkdir()
    orphan = root / "baseline"
    (orphan / ".state" / "checkpoints").mkdir(parents=True)
    (orphan / "evidence" / "full").mkdir(parents=True)
    (orphan / ".state" / "checkpoints" / "receipt.json").write_text("{}", encoding="utf-8")
    (orphan / "evidence" / "full" / "frame.png").write_bytes(b"pixels")
    neighbor = root / "source-media"
    neighbor.mkdir()
    (neighbor / "source.mp4").write_bytes(b"source")

    planned = prune_orphans(root)
    assert planned["dry_run"] is True
    assert Path(planned["planned_orphans"][0]["path"]) == orphan.resolve()
    assert planned["planned_bytes"] == 8
    assert orphan.is_dir() and (neighbor / "source.mp4").is_file()

    applied = prune_orphans(root, apply=True)
    assert applied["dry_run"] is False
    assert applied["removed_orphans"] == [str(orphan.resolve())]
    assert not orphan.exists()
    assert (neighbor / "source.mp4").is_file()


def test_prune_orphans_never_removes_a_marked_project(tmp_path: Path) -> None:
    root = tmp_path / "benchmark-root"
    root.mkdir()
    project = _project(root, "complete", "2026-08-05T00:00:00Z", 2)

    result = prune_orphans(root, apply=True)

    assert result["removed_orphans"] == []
    assert project.is_dir()
