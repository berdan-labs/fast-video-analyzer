from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import video_script_reconstructor.api as api
from video_script_reconstructor.errors import InputError


def _plan_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "input_classification": "transcript",
        "probe": None,
        "likely_transcript_sources": ["captions.srt"],
        "planned_stages": ["identity", "reconstruction"],
        "strict_prerequisites": ["FFmpeg/FFprobe for media"],
        "asr_expected": False,
        "ocr_expected": False,
        "visual_review_expected": False,
        "image_metadata_plan": "deterministic",
        "semantic_pending_possible": False,
        "network_actions_requiring_permission": [],
        "estimated_evidence_images": 0,
        "estimated_disk_bytes": 42,
        "asr_plan": {"required": False},
        "output_path": "C:/outputs/captions",
        "output_contract": "one Markdown file",
        "no_full_processing_statement": "No model was downloaded.",
        "offline": True,
    }


def test_plan_returns_a_frozen_typed_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_plan(input_value: str | Path, **kwargs: object) -> dict[str, object]:
        captured["input"] = input_value
        captured.update(kwargs)
        return _plan_payload()

    monkeypatch.setattr(api, "_plan_input", fake_plan)

    result = api.plan(
        "captions.srt",
        output_root="outputs",
        subtitles=("captions.srt",),
        transcript="captions.txt",
    )

    assert isinstance(result, api.Plan)
    assert result.input_classification == "transcript"
    assert result.output_path == Path("C:/outputs/captions")
    assert captured["output_root"] == Path("outputs")
    assert captured["subtitles"] == (Path("captions.srt"),)
    assert captured["transcript"] == Path("captions.txt")
    with pytest.raises(TypeError):
        result.asr_plan["required"] = True  # type: ignore[index]


def test_run_and_validate_normalize_internal_results(monkeypatch: pytest.MonkeyPatch) -> None:
    internal_validation = SimpleNamespace(
        valid=True,
        errors=[],
        warnings=["cached"],
        checks={"markdown_count": 1},
        project_status="automatically_checked",
    )
    internal_run = SimpleNamespace(
        project_dir=Path("outputs/project"),
        markdown_path=Path("outputs/project/project.md"),
        status="automatically_checked",
        exit_code=0,
        validation=internal_validation,
    )
    monkeypatch.setattr(api, "_run_pipeline", lambda *_args, **_kwargs: internal_run)
    monkeypatch.setattr(api, "_validate_project", lambda *_args, **_kwargs: internal_validation)

    result = api.run("captions.txt", output_root="outputs", vision_mode="none")
    validation = api.validate("outputs/project")

    assert result.status == "automatically_checked"
    assert result.exit_code == 0
    assert result.validation is not None
    assert result.validation.project_status == "automatically_checked"
    assert validation.checks["markdown_count"] == 1
    with pytest.raises(TypeError):
        validation.checks["markdown_count"] = 2  # type: ignore[index]


def test_review_snapshots_preserve_core_fields_and_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    item = {
        "review_id": "R000001",
        "severity": "high",
        "category": "visual",
        "problem": "The frame is ambiguous.",
        "required_action": "Inspect the evidence.",
        "blocking": True,
        "start_ms": 100,
        "end_ms": 200,
        "frame_ids": ["F000001"],
        "image_claim_ids": ["C000001"],
        "alternatives": ["A", "B"],
        "decision": None,
    }
    detail = {
        **item,
        "time_range_ms": {"start": 100, "end": 200},
        "image_paths": ["evidence/full/F000001.png"],
        "source_ids": {"frames": ["F000001"]},
        "competing_evidence": [
            {
                "claim_id": "C000001",
                "statement": "A claim",
                "status": "uncertain",
                "alternatives": ["An alternative"],
            }
        ],
    }
    monkeypatch.setattr(api, "_list_review_items", lambda *_args: [item])
    monkeypatch.setattr(api, "_show_review_item", lambda *_args: detail)

    listed = api.list_review_items("project")
    shown = api.show_review_item("project", "R000001")

    assert isinstance(listed, tuple)
    assert listed[0].review_id == "R000001"
    assert shown.image_paths == (Path("evidence/full/F000001.png"),)
    assert shown.source_ids["frames"] == ("F000001",)
    assert shown.competing_evidence[0].claim_id == "C000001"


def test_missing_paths_are_normalized_to_input_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("missing project")

    monkeypatch.setattr(api, "_validate_project", missing)
    with pytest.raises(InputError, match="missing project"):
        api.validate("missing")
