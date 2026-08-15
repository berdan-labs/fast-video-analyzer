from __future__ import annotations

from pathlib import Path

import pytest

import video_script_reconstructor.cache as cache_module
from video_script_reconstructor.cache import (
    StageCache,
    compact_completed_checkpoints,
    completed_checkpoint_compaction_plan,
    purge_project_cache,
)
from video_script_reconstructor.errors import InputError


def test_project_cache_purge_is_scoped_and_recreates_cache(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / ".state").mkdir(parents=True)
    (project / ".state" / "canonical-project.json").write_text("{}", encoding="utf-8")
    cache = StageCache(project)
    cache.commit("timeline", "key", [".state/timeline/timeline.json"])
    visual_frames = project / ".state" / "checkpoints" / "visual-frames" / "key"
    visual_frames.mkdir(parents=True)
    (visual_frames / "frame.png").write_bytes(b"frame")
    ocr_dir = project / ".state" / "checkpoints" / "ocr"
    ocr_dir.mkdir(parents=True)
    (ocr_dir / "observations.json").write_text("{}", encoding="utf-8")
    (project / ".state" / "checkpoints" / "visual-survey.json").write_text("{}", encoding="utf-8")
    (project / ".state" / "checkpoints" / "visual-survey-structural.json").write_text(
        "{}", encoding="utf-8"
    )
    unrelated = project / "evidence.txt"
    unrelated.write_text("preserve me", encoding="utf-8")

    assert purge_project_cache(project) == 5
    assert list((project / ".state" / "cache").iterdir()) == []
    assert not (project / ".state" / "checkpoints" / "visual-frames").exists()
    assert not (project / ".state" / "checkpoints" / "ocr").exists()
    assert not (project / ".state" / "checkpoints" / "visual-survey.json").exists()
    assert not (project / ".state" / "checkpoints" / "visual-survey-structural.json").exists()
    assert unrelated.read_text(encoding="utf-8") == "preserve me"


def test_cache_purge_refuses_unrecognized_project(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="canonical state missing"):
        purge_project_cache(tmp_path)


def test_completed_checkpoint_compaction_preserves_evidence_and_asr_state(tmp_path: Path) -> None:
    project = tmp_path / "completed"
    (project / ".state").mkdir(parents=True)
    (project / ".state" / "canonical-project.json").write_text("{}", encoding="utf-8")
    (project / "artifact.bin").write_bytes(b"evidence")
    cache = project / ".state" / "cache"
    cache.mkdir(parents=True)
    (cache / "visual.json").write_bytes(b"visual-cache")
    visual = project / ".state" / "checkpoints" / "visual-frames" / "key"
    visual.mkdir(parents=True)
    (visual / "frame.png").write_bytes(b"raw-frame")
    ocr = project / ".state" / "checkpoints" / "ocr"
    ocr.mkdir(parents=True)
    (ocr / "observations.json").write_bytes(b"ocr-cache")
    survey = project / ".state" / "checkpoints" / "visual-survey.json"
    survey.write_text("{}", encoding="utf-8")
    structural_survey = project / ".state" / "checkpoints" / "visual-survey-structural.json"
    structural_survey.write_text("{}", encoding="utf-8")
    asr = project / ".state" / "checkpoints" / "asr" / "large-v3"
    asr.mkdir(parents=True)
    (asr / "chunk-000.json").write_text("{}", encoding="utf-8")

    result = compact_completed_checkpoints(project)

    assert result["kept"] is False
    assert result["removed_files"] == 3
    assert result["reclaimed_bytes"] > 0
    assert result["targets"]
    assert list(cache.iterdir()) == []
    assert not visual.exists()
    assert not ocr.exists()
    assert survey.is_file()
    assert structural_survey.is_file()
    assert (asr / "chunk-000.json").is_file()
    assert (project / "artifact.bin").is_file()


def test_completed_checkpoint_compaction_can_be_opted_out(tmp_path: Path) -> None:
    project = tmp_path / "keep"
    (project / ".state").mkdir(parents=True)
    (project / ".state" / "canonical-project.json").write_text("{}", encoding="utf-8")
    visual = project / ".state" / "checkpoints" / "visual-frames" / "key"
    visual.mkdir(parents=True)
    frame = visual / "frame.png"
    frame.write_bytes(b"raw-frame")

    result = compact_completed_checkpoints(project, keep=True)

    assert result == {
        "kept": True,
        "removed_files": 0,
        "reclaimed_bytes": 0,
        "targets": [],
    }
    assert frame.is_file()


def test_completed_checkpoint_compaction_plan_is_read_only(tmp_path: Path) -> None:
    project = tmp_path / "planned"
    (project / ".state").mkdir(parents=True)
    (project / ".state" / "canonical-project.json").write_text("{}", encoding="utf-8")
    visual = project / ".state" / "checkpoints" / "visual-frames" / "key"
    visual.mkdir(parents=True)
    frame = visual / "frame.png"
    frame.write_bytes(b"raw-frame")
    asr = project / ".state" / "checkpoints" / "asr"
    asr.mkdir(parents=True)
    (asr / "chunk.json").write_text("keep", encoding="utf-8")

    plan = completed_checkpoint_compaction_plan(project)

    assert plan["dry_run"] is True
    assert plan["removed_files"] == 1
    assert plan["reclaimed_bytes"] == len(b"raw-frame")
    assert frame.is_file()
    assert (asr / "chunk.json").is_file()


def test_completed_checkpoint_inventory_rejects_symlinked_entries(tmp_path: Path) -> None:
    project = tmp_path / "symlinked"
    (project / ".state").mkdir(parents=True)
    (project / ".state" / "canonical-project.json").write_text("{}", encoding="utf-8")
    visual = project / ".state" / "checkpoints" / "visual-frames"
    visual.mkdir(parents=True)
    target = visual / "target.png"
    target.write_bytes(b"target")
    link = visual / "link.png"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(InputError, match="symlink"):
        completed_checkpoint_compaction_plan(project)


def test_completed_checkpoint_inventory_does_not_use_recursive_rglob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "single-pass"
    (project / ".state").mkdir(parents=True)
    (project / ".state" / "canonical-project.json").write_text("{}", encoding="utf-8")
    visual = project / ".state" / "checkpoints" / "visual-frames" / "nested"
    visual.mkdir(parents=True)
    (visual / "frame-1.png").write_bytes(b"one")
    (visual / "frame-2.png").write_bytes(b"two-two")

    def fail_rglob(_self: Path, _pattern: str) -> object:
        raise AssertionError("completed checkpoint inventory must not call Path.rglob")

    def fail_walk(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("completed checkpoint inventory must recurse with os.scandir")

    monkeypatch.setattr(cache_module.Path, "rglob", fail_rglob)
    with monkeypatch.context() as inventory_patch:
        inventory_patch.setattr(cache_module.os, "walk", fail_walk)
        plan = cache_module.completed_checkpoint_compaction_plan(project)
    result = cache_module.compact_completed_checkpoints(project)

    assert plan["removed_files"] == 2
    assert plan["reclaimed_bytes"] == len(b"one") + len(b"two-two")
    assert result["removed_files"] == plan["removed_files"]
    assert result["reclaimed_bytes"] == plan["reclaimed_bytes"]
