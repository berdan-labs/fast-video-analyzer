"""Focused durability checks for staged semantic batch materialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_pipeline_contract import FixtureVisionProvider, _generate

from video_script_reconstructor.image_metadata import normalized_pixel_hash
from video_script_reconstructor.semantic_pipeline import apply_vision_provider
from video_script_reconstructor.validate_output import validate_project
from video_script_reconstructor.vision_packets import VisionPacket


def _project_payload(project_dir: Path) -> dict[str, object]:
    return json.loads(
        (project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )


def test_semantic_batch_materializes_canonical_and_ledger_once_with_pixel_invariance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    from video_script_reconstructor.pipeline import run_pipeline

    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    before = _project_payload(result.project_dir)
    before_pixels = {
        str(frame["frame_id"]): normalized_pixel_hash(result.project_dir / str(frame["path"]))
        for frame in before["frames"]  # type: ignore[index]
    }

    import video_script_reconstructor.evidence as evidence_module
    import video_script_reconstructor.semantic_pipeline as semantic_module

    ledger_writes = 0
    evidence_write = evidence_module.atomic_write_json
    semantic_write = semantic_module.atomic_write_json

    def count_evidence_write(path: Path, payload: object, **kwargs: object) -> None:
        nonlocal ledger_writes
        if path.name == "image-observations.json":
            ledger_writes += 1
        evidence_write(path, payload, **kwargs)

    def count_semantic_write(path: Path, payload: object, **kwargs: object) -> None:
        nonlocal ledger_writes
        if path.name == "image-observations.json":
            ledger_writes += 1
        semantic_write(path, payload, **kwargs)

    monkeypatch.setattr(evidence_module, "atomic_write_json", count_evidence_write)
    monkeypatch.setattr(semantic_module, "atomic_write_json", count_semantic_write)
    summary = apply_vision_provider(result.project_dir, FixtureVisionProvider())
    assert summary["applied"]
    assert not (
        result.project_dir / ".state" / "checkpoints" / "semantic-batch-journal.jsonl"
    ).exists()
    after = _project_payload(result.project_dir)
    ledger = json.loads(
        (result.project_dir / ".state" / "vision" / "image-observations.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(after["visual_observations"]) == len(ledger["observations"])  # type: ignore[index]
    assert ledger_writes == 1
    assert len({item["observation_id"] for item in after["visual_observations"]}) == len(  # type: ignore[index]
        after["visual_observations"]  # type: ignore[index]
    )
    assert validate_project(result.project_dir, use_cached_file_hash=True).valid
    assert {
        str(frame["frame_id"]): normalized_pixel_hash(result.project_dir / str(frame["path"]))
        for frame in after["frames"]  # type: ignore[index]
    } == before_pixels


def test_semantic_batch_journal_recovers_after_provider_failure_without_duplicates(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    from video_script_reconstructor.pipeline import run_pipeline

    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    failing = FixtureVisionProvider()
    original_annotate = failing.annotate

    def fail_on_second(packet: VisionPacket, *, project_root: Path):
        if len(failing.calls) >= 1:
            raise RuntimeError("injected semantic interruption")
        return original_annotate(packet, project_root=project_root)

    failing.annotate = fail_on_second  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="injected semantic interruption"):
        apply_vision_provider(result.project_dir, failing)
    journal = result.project_dir / ".state" / "checkpoints" / "semantic-batch-journal.jsonl"
    assert journal.is_file()

    resumed = apply_vision_provider(result.project_dir, FixtureVisionProvider())
    assert resumed["semantic_batch_journal_recovered_candidate_ids"]
    assert not journal.exists()
    canonical = _project_payload(result.project_dir)
    ledger = json.loads(
        (result.project_dir / ".state" / "vision" / "image-observations.json").read_text(
            encoding="utf-8"
        )
    )
    canonical_ids = [item["observation_id"] for item in canonical["visual_observations"]]  # type: ignore[index]
    ledger_ids = [item["observation_id"] for item in ledger["observations"]]
    assert len(canonical_ids) == len(set(canonical_ids))
    assert canonical_ids == ledger_ids
    assert validate_project(result.project_dir, use_cached_file_hash=True).valid


def test_semantic_batch_journal_recovers_after_ledger_write_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    from video_script_reconstructor.pipeline import run_pipeline

    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    import video_script_reconstructor.semantic_pipeline as semantic_module

    original_atomic_write_json = semantic_module.atomic_write_json
    injected = False

    def fail_ledger_materialization(path: Path, payload: object, **kwargs: object) -> None:
        nonlocal injected
        if path.name == "image-observations.json" and not injected:
            injected = True
            raise OSError("simulated semantic ledger write failure")
        original_atomic_write_json(path, payload, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(semantic_module, "atomic_write_json", fail_ledger_materialization)
        with pytest.raises(OSError, match="simulated semantic ledger write failure"):
            apply_vision_provider(result.project_dir, FixtureVisionProvider())

    assert injected
    journal = result.project_dir / ".state" / "checkpoints" / "semantic-batch-journal.jsonl"
    assert journal.is_file()
    interrupted = _project_payload(result.project_dir)
    assert interrupted["visual_observations"]

    resumed = apply_vision_provider(result.project_dir, FixtureVisionProvider())
    assert resumed["semantic_batch_journal_recovered_candidate_ids"]
    assert not journal.exists()
    canonical = _project_payload(result.project_dir)
    ledger = json.loads(
        (result.project_dir / ".state" / "vision" / "image-observations.json").read_text(
            encoding="utf-8"
        )
    )
    canonical_ids = [item["observation_id"] for item in canonical["visual_observations"]]
    ledger_ids = [item["observation_id"] for item in ledger["observations"]]
    assert len(canonical_ids) == len(set(canonical_ids))
    assert canonical_ids == ledger_ids
    assert validate_project(result.project_dir, use_cached_file_hash=True).valid
