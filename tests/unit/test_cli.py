from __future__ import annotations

import json
from pathlib import Path

import video_script_reconstructor.cli as cli
import video_script_reconstructor.pipeline as pipeline
import video_script_reconstructor.subagent_review as subagent_review
import video_script_reconstructor.validate_output as validate_output
from video_script_reconstructor.cli import (
    _compact_doctor_report,
    _compact_plan,
    _compact_semantic_batch_result,
)
from video_script_reconstructor.validate_output import ValidationResult


def test_compact_semantic_batch_result_bounds_large_packet_lists() -> None:
    result = _compact_semantic_batch_result(
        {
            "status": "review_required",
            "projects": [
                {
                    "project_dir": "project",
                    "summary": {
                        "applied": [{"observation_id": "VO000001"}],
                        "skipped_event_ids": ["VE000001"],
                        "semantic_deferred_event_ids": [f"VE{i:06d}" for i in range(20)],
                        "semantic_provider_failures": [
                            {"candidate_id": "VE000002", "error": "x" * 500}
                        ],
                        "semantic_cache_hit_count": 2,
                    },
                }
            ],
        }
    )

    summary = result["projects"][0]["summary"]
    assert result["output_mode"] == "compact"
    assert summary["applied_count"] == 1
    assert summary["applied_observation_id_sample"] == ["VO000001"]
    assert summary["skipped_event_count"] == 1
    assert summary["semantic_deferred_event_count"] == 20
    assert len(summary["semantic_deferred_event_id_sample"]) == 7
    assert summary["semantic_provider_failure_count"] == 1
    assert len(summary["semantic_provider_failure_sample"][0]["error"]) == 240
    assert "applied" not in summary
    assert "semantic_deferred_event_ids" not in summary


def test_compact_semantic_batch_result_preserves_non_summary_projects() -> None:
    result = _compact_semantic_batch_result(
        {
            "projects": [
                {"project_dir": "empty", "status": "skipped", "pending_before": 0},
                "not-a-project-record",
            ]
        }
    )

    assert result["projects"][0]["status"] == "skipped"
    assert result["projects"][1] == "not-a-project-record"
    assert result["output_mode"] == "compact"


def test_compact_doctor_report_is_path_free_and_actionable() -> None:
    checks = {
        name: {"status": "available"}
        for name in (
            "python",
            "ffmpeg",
            "ffprobe",
            "package_import",
            "package_data",
            "image_metadata",
            "disk",
            "output_write",
            "network_policy",
            "ocr",
            "vision_provider",
        )
    }
    checks["speech_recognition"] = {"status": "optional"}
    summary = _compact_doctor_report({"schema_version": "1.0", "checks": checks})

    assert summary["output_mode"] == "summary"
    assert summary["status"] == "ready"
    assert summary["blocking_checks"] == []
    assert summary["optional_capabilities"] == ["speech_recognition"]
    assert any("plan <input>" in item for item in summary["next_actions"])
    assert "path" not in json.dumps(summary)

    checks["ffmpeg"] = {"status": "blocking-for-strict", "fix": "Install FFmpeg."}
    blocked = _compact_doctor_report({"schema_version": "1.0", "checks": checks})
    assert blocked["status"] == "blocked"
    assert blocked["blocking_checks"] == ["ffmpeg"]
    assert blocked["next_actions"][0] == "Fix ffmpeg: Install FFmpeg."


def test_compact_plan_emits_copyable_run_and_validation_actions() -> None:
    summary = _compact_plan(
        {
            "schema_version": "1.0",
            "input_classification": "video",
            "asr_expected": False,
            "ocr_expected": True,
            "visual_review_expected": True,
            "estimated_evidence_images": 2,
            "estimated_disk_bytes": 1234,
            "output_path": "C:/outputs/demo",
            "output_contract": "one Markdown report",
            "network_actions_requiring_permission": [],
            "no_full_processing_statement": "No full processing.",
        },
        input_value="demo.mp4",
        subtitles=[Path("demo.srt")],
        transcript=None,
        output_root=Path("outputs"),
        preset="strict",
        vision_mode="host-agent",
    )

    assert summary["output_mode"] == "summary"
    assert summary["workflow"] == "subtitle-led"
    assert summary["next_actions"][0].startswith("Subtitle-led workflow")
    assert 'fast-video-analyzer run "demo.mp4" --subtitle "demo.srt"' in summary["next_actions"][1]
    assert 'fast-video-analyzer validate "C:/outputs/demo"' in summary["next_actions"][2]
    assert any("review_required" in item for item in summary["next_actions"])


def test_doctor_and_plan_parsers_expose_opt_in_summary() -> None:
    assert cli._parser().parse_args(["doctor", "--summary"]).summary is True
    assert cli._parser().parse_args(["plan", "demo.mp4", "--summary"]).summary is True


def test_review_bundle_create_all_parser_exposes_storage_and_budget_controls() -> None:
    args = cli._parser().parse_args(
        [
            "review",
            "bundle",
            "create-all",
            "projects",
            "--output-root",
            "handoffs",
            "--max-packets-per-project",
            "12",
            "--min-free-bytes",
            "123",
            "--max-bundle-bytes",
            "456",
            "--continue-on-error",
            "--dry-run",
            "--include-provider",
            "llama.cpp-local",
        ]
    )
    assert args.projects_root == Path("projects")
    assert args.output_root == Path("handoffs")
    assert args.max_packets_per_project == 12
    assert args.min_free_bytes == 123
    assert args.max_bundle_bytes == 456
    assert args.continue_on_error is True
    assert args.dry_run is True
    assert args.include_annotation_providers == ["llama.cpp-local"]
    for alias in ("batch-create", "create-batch"):
        alias_args = cli._parser().parse_args(["review", "bundle", alias, "projects"])
        assert alias_args.review_bundle_command == alias
        assert alias_args.projects_root == Path("projects")
        assert alias_args.include_annotation_providers == []


def test_batch_parser_accepts_language_hint() -> None:
    args = cli._parser().parse_args(
        [
            "batch",
            "sources",
            "--output",
            "out",
            "--language",
            "fil",
            "--compare-sidecars",
        ]
    )
    assert args.language == "fil"
    assert args.compare_sidecars is True


def test_run_and_batch_output_is_optional_for_user_documents_root() -> None:
    run_args = cli._parser().parse_args(["run", "lesson.mp4"])
    batch_args = cli._parser().parse_args(["batch", "lessons"])
    assert run_args.output is None
    assert batch_args.output is None


def test_run_parser_exposes_bounded_semantic_review_budget() -> None:
    args = cli._parser().parse_args(["run", "lesson.mp4", "--semantic-max-packets", "240"])
    assert args.semantic_max_packets == 240


def test_review_bundle_apply_parser_exposes_opt_in_partial_success() -> None:
    args = cli._parser().parse_args(
        [
            "review",
            "bundle",
            "apply",
            "project",
            "--bundle",
            "bundle",
            "--accept-partial",
        ]
    )
    assert args.accept_partial is True


def test_review_bundle_apply_accept_partial_requires_clean_commit(
    monkeypatch,
) -> None:
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "_json", emitted.append)
    monkeypatch.setattr(
        subagent_review,
        "apply_review_bundle",
        lambda *args, **kwargs: {
            "status": "review_required",
            "summary": {"applied": [{"observation_id": "VA000001"}]},
            "validation_errors": [],
            "missing_candidate_ids": [],
        },
    )
    args = cli._parser().parse_args(
        [
            "review",
            "bundle",
            "apply",
            "project",
            "--bundle",
            "bundle",
            "--accept-partial",
        ]
    )
    assert cli._run(args) == 0
    assert emitted[-1]["status"] == "review_required"

    monkeypatch.setattr(
        subagent_review,
        "apply_review_bundle",
        lambda *args, **kwargs: {
            "status": "review_required",
            "summary": {"applied": [{"observation_id": "VA000001"}]},
            "validation_errors": ["bad proof"],
            "missing_candidate_ids": [],
        },
    )
    assert cli._run(args) == 3


def test_default_output_root_honors_explicit_environment_override(
    tmp_path: Path, monkeypatch
) -> None:
    configured = tmp_path / "reconstruction-output"
    monkeypatch.setenv("VSR_OUTPUT_ROOT", str(configured))
    assert pipeline.default_output_root() == configured.resolve()


def test_default_output_root_uses_documents_script_reconstructor_outputs(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VSR_OUTPUT_ROOT", raising=False)
    assert pipeline.default_output_root() == (
        Path.home() / "Documents" / "Script Reconstructor Outputs"
    ).resolve()


def test_colocated_output_dir_uses_source_stem_and_analyzer_suffix(tmp_path: Path) -> None:
    source = tmp_path / "training-session.mp4"
    source.write_bytes(b"fixture")

    assert pipeline.colocated_output_dir(source) == (
        tmp_path / "training-session (Analyzer Outputs)"
    )


def test_plan_input_uses_colocated_path_unless_shared_root_is_explicit(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "Demo Session.txt"
    source.write_text("A complete transcript.", encoding="utf-8")
    monkeypatch.delenv("VSR_OUTPUT_ROOT", raising=False)

    colocated = pipeline.plan_input(source, offline=True)
    assert Path(colocated["output_path"]) == tmp_path / "Demo Session (Analyzer Outputs)"

    shared = tmp_path / "shared"
    monkeypatch.setenv("VSR_OUTPUT_ROOT", str(shared))
    configured = pipeline.plan_input(source, offline=True)
    assert Path(configured["output_path"]) == shared / "Demo-Session"


def test_run_pipeline_without_output_writes_source_adjacent_report(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "Demo Session.txt"
    source.write_text("A complete transcript.", encoding="utf-8")
    monkeypatch.delenv("VSR_OUTPUT_ROOT", raising=False)

    result = pipeline.run_pipeline(
        source,
        subtitle_mode="provided-only",
        vision_mode="none",
        offline=True,
        resume=False,
    )

    expected = tmp_path / "Demo Session (Analyzer Outputs)"
    assert result.project_dir == expected
    assert result.markdown_path == expected / "Demo Session.md"
    assert result.markdown_path.is_file()
    assert (expected / "evidence" / "full").is_dir()


def test_run_pipeline_honors_vsr_output_root_as_explicit_override(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "Demo Session.txt"
    source.write_text("A complete transcript.", encoding="utf-8")
    shared = tmp_path / "shared-output"
    monkeypatch.setenv("VSR_OUTPUT_ROOT", str(shared))

    result = pipeline.run_pipeline(
        source,
        subtitle_mode="provided-only",
        vision_mode="none",
        offline=True,
        resume=False,
    )

    expected = shared / "Demo-Session"
    assert result.project_dir == expected
    assert result.markdown_path == expected / "Demo-Session.reconstruction.md"
    assert result.markdown_path.is_file()


def test_cache_compact_apply_publishes_telemetry_and_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    state = project / ".state"
    state.mkdir(parents=True)
    canonical = state / "canonical-project.json"
    canonical.write_text(
        json.dumps(
            {
                "manifest": {"run_cache_key": "RUN-KEY"},
                "project_status": "automatically_checked",
            }
        ),
        encoding="utf-8",
    )
    visual = state / "checkpoints" / "visual-frames"
    visual.mkdir(parents=True)
    (visual / "raw.png").write_bytes(b"raw")
    ocr = state / "checkpoints" / "ocr"
    ocr.mkdir(parents=True)
    (ocr / "raw.json").write_text("{}", encoding="utf-8")
    asr = state / "checkpoints" / "asr"
    asr.mkdir(parents=True)
    asr_file = asr / "chunk.json"
    asr_file.write_text("keep", encoding="utf-8")
    evidence = project / "evidence.bin"
    evidence.write_bytes(b"keep")

    validation = ValidationResult(valid=True, checks={"markdown_count": 1})
    monkeypatch.setattr(validate_output, "validate_project", lambda *args, **kwargs: validation)
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "_json", emitted.append)

    args = cli._parser().parse_args(["cache", "compact", str(project), "--apply"])
    assert cli._run(args) == 0

    stored = json.loads(canonical.read_text(encoding="utf-8"))
    compaction = stored["manifest"]["performance"]["checkpoint_compaction"]
    assert compaction["removed_files"] == 2
    assert compaction["reclaimed_bytes"] == len(b"raw") + 2
    assert stored["manifest"]["performance"]["resource_usage"]["output"][
        "reclaimable_bytes"
    ] == 0
    assert not visual.exists()
    assert not ocr.exists()
    assert asr_file.read_text(encoding="utf-8") == "keep"
    assert evidence.read_bytes() == b"keep"
    assert (state / "validation-receipt.json").is_file()
    assert emitted[-1]["validation_errors"] == []
