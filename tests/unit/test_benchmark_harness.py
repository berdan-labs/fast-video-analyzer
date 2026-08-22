from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _benchmark_module() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "benchmark_pipeline.py"
    spec = importlib.util.spec_from_file_location("vsr_benchmark_pipeline", path)
    if spec is None or spec.loader is None:
        raise AssertionError("benchmark script could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timing_summary_is_conservative_for_small_samples() -> None:
    module = _benchmark_module()
    summary = module.summarize_elapsed([1.0, 3.0, 2.0])
    assert summary == {
        "count": 3,
        "min_seconds": 1.0,
        "median_seconds": 2.0,
        "p95_seconds": 3.0,
        "max_seconds": 3.0,
    }


def test_runtime_summary_sanitizes_gpu_and_revision_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _benchmark_module()

    def command_output(name: str, *_args: str) -> str | None:
        return {
            "git": "0123456789abcdef0123456789abcdef01234567",
            "ffmpeg": "ffmpeg version 7.1\nCopyright owner-local path must not appear",
            "nvidia-smi": "NVIDIA GeForce RTX 3060, 12288, 560.01",
        }.get(name)

    monkeypatch.setattr(module, "_command_output", command_output)
    monkeypatch.setattr(module, "_package_versions", lambda: {"paddleocr": "3.7.0"})
    summary = module._runtime_summary()

    assert summary["git_revision"] == "0123456789ab"
    assert summary["ffmpeg_version"] == "ffmpeg version 7.1"
    assert summary["gpus"] == [
        {
            "name": "NVIDIA GeForce RTX 3060",
            "memory_total_mib": 12288,
            "driver_version": "560.01",
        }
    ]
    assert summary["package_versions"] == {"paddleocr": "3.7.0"}
    assert "uuid" not in json.dumps(summary).casefold()


def test_runtime_summary_degrades_cleanly_without_a_gpu() -> None:
    module = _benchmark_module()
    assert module._nvidia_gpus(None) == []
    assert module._nvidia_gpus("malformed") == []


def test_quality_summary_hashes_only_declared_stable_lanes(tmp_path: Path) -> None:
    module = _benchmark_module()
    state = tmp_path / ".state"
    state.mkdir()
    canonical = {
        "generated_at_utc": "first",
        "media": {"duration_ms": 18_000_000},
        "transcript_segments": [
            {
                "substantive": True,
                "normalized_text": "Exact text",
                "words": [{"text": "Exact"}, {"text": "text"}],
            }
        ],
        "frames": [{"frame_id": "F000001", "sha256": "a" * 64}],
        "ocr_observations": [{"frame_id": "F000001", "text": "42"}],
        "script_blocks": [{"block_id": "B000001", "text": "Exact text"}],
        "timeline": [{"start_ms": 0, "end_ms": 1000}],
        "visual_events": [{"event_id": "V000001"}],
        "evidence_image_metadata": [{"frame_id": "F000001"}],
        "tools_models_summary": {"ocr": "PP-OCRv5-server"},
        "audit": {"blocking_failures": []},
    }
    canonical_path = state / "canonical-project.json"
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    first = module._quality_summary(tmp_path)

    canonical["generated_at_utc"] = "second"
    canonical["unrelated_runtime_telemetry"] = {"elapsed_ms": 999}
    canonical_path.write_text(json.dumps(canonical, sort_keys=True), encoding="utf-8")
    second = module._quality_summary(tmp_path)

    assert first["media_duration_ms"] == 18_000_000
    assert first["lane_item_counts"]["ocr_observations"] == 1
    assert first["quality_contract_sha256"] == second["quality_contract_sha256"]
    assert first["lane_sha256"] == second["lane_sha256"]


def test_benchmark_repeat_reuses_warm_output_and_reports_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _benchmark_module()
    calls: list[tuple[Path, bool]] = []

    def fake_run(input_path: Path, *, output_root: Path, resume: bool, **_kwargs: object):
        calls.append((output_root, resume))
        project = output_root / "fixture"
        (project / ".state").mkdir(parents=True, exist_ok=True)
        (project / ".state" / "run-manifest.json").write_text(
            json.dumps({"run_cache_key": "fixture", "performance": {}}), encoding="utf-8"
        )
        return SimpleNamespace(
            project_dir=project,
            status="review_required",
            exit_code=3,
        )

    monkeypatch.setattr(module, "run_pipeline", fake_run)
    monkeypatch.setattr(
        module,
        "validate_project",
        lambda _project: SimpleNamespace(valid=True, errors=[]),
    )
    report = module.benchmark(
        tmp_path / "input.mp4",
        output_root=tmp_path / "warm",
        resume=True,
        repeat=3,
    )
    assert calls == [(tmp_path / "warm", True)] * 3
    assert report["schema_version"] == "1.1"
    assert report["report_kind"] == "pipeline-benchmark"
    assert report["timing_summary"]["count"] == 3
    assert len(report["iterations"]) == 3
    assert report["validation_valid"] is True


def test_benchmark_reuses_pipeline_final_validation_without_a_duplicate_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _benchmark_module()

    def fake_run(input_path: Path, *, output_root: Path, **_kwargs: object):
        project = output_root / "fixture"
        (project / ".state").mkdir(parents=True, exist_ok=True)
        (project / ".state" / "run-manifest.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            project_dir=project,
            status="review_required",
            exit_code=3,
            validation=SimpleNamespace(valid=True, errors=[]),
        )

    def unexpected_independent_validation(_project: Path) -> object:
        pytest.fail("default benchmark validation should reuse run_pipeline's final proof")

    monkeypatch.setattr(module, "run_pipeline", fake_run)
    monkeypatch.setattr(module, "validate_project", unexpected_independent_validation)
    report = module._benchmark_once(tmp_path / "input.mp4", output_root=tmp_path / "out")

    assert report["validation_valid"] is True
    assert report["validation_source"] == "pipeline-final"


def test_benchmark_independent_validation_is_explicit_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _benchmark_module()
    calls: list[Path] = []

    def fake_run(input_path: Path, *, output_root: Path, **_kwargs: object):
        project = output_root / "fixture"
        (project / ".state").mkdir(parents=True, exist_ok=True)
        (project / ".state" / "run-manifest.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            project_dir=project,
            status="review_required",
            exit_code=3,
            validation=SimpleNamespace(valid=True, errors=[]),
        )

    def independent_validation(project: Path) -> object:
        calls.append(project)
        return SimpleNamespace(valid=True, errors=[])

    monkeypatch.setattr(module, "run_pipeline", fake_run)
    monkeypatch.setattr(module, "validate_project", independent_validation)
    report = module._benchmark_once(
        tmp_path / "input.mp4",
        output_root=tmp_path / "out",
        independent_validation=True,
    )

    assert len(calls) == 1
    assert report["validation_valid"] is True
    assert report["validation_source"] == "independent-public"


def test_benchmark_repeat_isolates_cold_stage_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _benchmark_module()
    calls: list[Path] = []

    def fake_run(_input_path: Path, *, output_root: Path, **_kwargs: object):
        calls.append(output_root)
        project = output_root / "fixture"
        (project / ".state").mkdir(parents=True, exist_ok=True)
        (project / ".state" / "run-manifest.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(project_dir=project, status="review_required", exit_code=3)

    monkeypatch.setattr(module, "run_pipeline", fake_run)
    monkeypatch.setattr(
        module,
        "validate_project",
        lambda _project: SimpleNamespace(valid=True, errors=[]),
    )
    module.benchmark(
        tmp_path / "input.mp4",
        output_root=tmp_path / "cold",
        resume=False,
        repeat=2,
    )
    assert calls == [tmp_path / "cold" / "cold-001", tmp_path / "cold" / "cold-002"]


def test_benchmark_rejects_non_positive_repeat(tmp_path: Path) -> None:
    module = _benchmark_module()
    with pytest.raises(ValueError, match="repeat must be positive"):
        module.benchmark(tmp_path / "input.mp4", output_root=tmp_path, repeat=0)


def test_benchmark_forwards_asr_window_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _benchmark_module()
    captured: dict[str, object] = {}

    def fake_run(input_path: Path, *, output_root: Path, **kwargs: object):
        captured.update(kwargs)
        project = output_root / "fixture"
        (project / ".state").mkdir(parents=True, exist_ok=True)
        (project / ".state" / "run-manifest.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(project_dir=project, status="review_required", exit_code=3)

    monkeypatch.setattr(module, "run_pipeline", fake_run)
    monkeypatch.setattr(
        module,
        "validate_project",
        lambda _project: SimpleNamespace(valid=True, errors=[]),
    )
    report = module.benchmark(
        tmp_path / "input.mp4",
        output_root=tmp_path / "out",
        asr_chunk_seconds=150,
        asr_overlap_seconds=15,
    )
    assert captured["asr_chunk_seconds"] == 150
    assert captured["asr_overlap_seconds"] == 15
    assert report["asr_chunk_seconds"] == 150
    assert report["asr_overlap_seconds"] == 15


def test_benchmark_reports_resource_telemetry_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _benchmark_module()
    output = {"file_count": 2, "bytes": 3}

    def fake_run(input_path: Path, *, output_root: Path, **_kwargs: object):
        project = output_root / "fixture"
        (project / ".state").mkdir(parents=True, exist_ok=True)
        (project / ".state" / "run-manifest.json").write_text(
            json.dumps({"performance": {"resource_usage": {"output": output}}}),
            encoding="utf-8",
        )
        return SimpleNamespace(project_dir=project, status="review_required", exit_code=3)

    monkeypatch.setattr(module, "run_pipeline", fake_run)
    monkeypatch.setattr(module, "validate_project", lambda _project: SimpleNamespace(valid=True, errors=[]))
    monkeypatch.setattr(module, "resource_snapshot", lambda _project: {"output": output})

    report = module._benchmark_once(tmp_path / "input.mp4", output_root=tmp_path / "out")

    assert report["resource_output"] == output
    assert report["resource_output_matches_disk"] is True


def test_performance_summary_exposes_critical_path_and_resource_facts() -> None:
    module = _benchmark_module()
    summary = module._performance_summary(
        {
            "stage_records": [
                {"name": "transcript", "elapsed_ms": 8000},
                {"name": "visual_evidence", "elapsed_ms": 3000},
            ],
            "performance": {
                "resource_usage": {
                    "memory": {"peak_rss_bytes": 1234},
                    "output": {"bytes": 5678, "file_count": 9},
                },
                "scheduling": {"parallel_visual_survey": True},
                "visual_events": [
                    {"event": "survey_parallel_completed", "elapsed_seconds": 7.5}
                ],
            },
        },
        elapsed_seconds=9.0,
    )
    assert summary["stage_elapsed_seconds"] == {
        "transcript": 8.0,
        "visual_evidence": 3.0,
    }
    assert summary["stage_sum_seconds"] == 11.0
    assert summary["stage_sum_minus_wall_seconds"] == 2.0
    assert summary["peak_rss_bytes"] == 1234
    assert summary["output_bytes"] == 5678
    assert summary["parallel_visual_survey"] is True
    assert summary["survey_parallel_elapsed_seconds"] == 7.5
    assert summary["measurement_mode"] == "pipeline-execution"
    assert summary["stage_telemetry_source"] == "current-run-manifest"
    assert summary["stage_telemetry_current"] is True

    warm = module._performance_summary(
        {"stage_records": [{"name": "transcript", "elapsed_ms": 8000}]},
        elapsed_seconds=0.02,
        cache_reused=True,
    )
    assert warm["measurement_mode"] == "warm-cache-hit"
    assert warm["stage_telemetry_source"] == "previous-run-manifest"
    assert warm["stage_telemetry_current"] is False


def test_parse_chunk_sweep_deduplicates_and_rejects_invalid_values() -> None:
    module = _benchmark_module()
    assert module._parse_chunk_sweep("150, 300,150") == (150, 300)
    with pytest.raises(ValueError, match="positive"):
        module._parse_chunk_sweep("150,0")
    with pytest.raises(ValueError, match="invalid"):
        module._parse_chunk_sweep("fast")


def test_asr_chunk_sweep_recommends_only_maximum_coverage_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _benchmark_module()
    calls: list[tuple[int | None, int | None, bool]] = []

    def fake_benchmark(
        _input: Path,
        *,
        output_root: Path,
        asr_chunk_seconds: int | None,
        asr_overlap_seconds: int | None,
        resume: bool,
        **_kwargs: object,
    ) -> dict[str, object]:
        assert asr_chunk_seconds is not None
        calls.append((asr_chunk_seconds, asr_overlap_seconds, resume))
        quality = {
            "available": True,
            "substantive_segment_count": 10 if asr_chunk_seconds == 150 else 9,
            "word_count": 100 if asr_chunk_seconds == 150 else 99,
        }
        return {
            "elapsed_seconds": 20.0 if asr_chunk_seconds == 150 else 10.0,
            "timing_summary": {"count": 1},
            "status": "review_required",
            "validation_valid": True,
            "project_dir": str(output_root / "fixture"),
            "quality": quality,
        }

    monkeypatch.setattr(module, "benchmark", fake_benchmark)
    report = module.benchmark_asr_chunk_sweep(
        tmp_path / "input.mp4",
        output_root=tmp_path / "out",
        chunk_seconds=(150, 300),
        overlap_seconds=15,
    )
    assert calls == [(150, 15, False), (300, 15, False)]
    assert report["recommendation"]["chunk_seconds"] == 150


def test_asr_chunk_sweep_with_changed_coverage_still_picks_maximum_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _benchmark_module()

    def fake_benchmark(
        _input: Path,
        *,
        asr_chunk_seconds: int | None,
        **_kwargs: object,
    ) -> dict[str, object]:
        assert asr_chunk_seconds is not None
        return {
            "elapsed_seconds": float(asr_chunk_seconds),
            "timing_summary": {"count": 1},
            "status": "review_required",
            "validation_valid": True,
            "project_dir": str(tmp_path / str(asr_chunk_seconds)),
            "quality": {
                "available": True,
                "substantive_segment_count": asr_chunk_seconds // 100,
                "word_count": asr_chunk_seconds // 10,
            },
        }

    monkeypatch.setattr(module, "benchmark", fake_benchmark)
    report = module.benchmark_asr_chunk_sweep(
        tmp_path / "input.mp4",
        output_root=tmp_path / "out",
        chunk_seconds=(150, 300),
        overlap_seconds=15,
    )
    assert report["recommendation"]["chunk_seconds"] == 300


def test_asr_chunk_sweep_reuses_and_closes_one_verified_whisper_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _benchmark_module()

    class Adapter:
        backend_name = "faster-whisper"

        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    adapter = Adapter()
    seen: list[object | None] = []
    monkeypatch.setattr(module, "_resolve_reusable_asr_adapter", lambda: adapter)

    def fake_benchmark(_input: Path, *, asr_adapter: object | None, **_kwargs: object) -> dict[str, object]:
        seen.append(asr_adapter)
        return {
            "elapsed_seconds": 1.0,
            "timing_summary": {"count": 1},
            "status": "review_required",
            "validation_valid": True,
            "project_dir": str(tmp_path / "fixture"),
            "quality": {
                "available": True,
                "substantive_segment_count": 1,
                "word_count": 2,
            },
        }

    monkeypatch.setattr(module, "benchmark", fake_benchmark)
    report = module.benchmark_asr_chunk_sweep(
        tmp_path / "input.mp4",
        output_root=tmp_path / "out",
        chunk_seconds=(150, 300),
    )
    assert seen == [adapter, adapter]
    assert adapter.close_count == 1
    assert report["reused_faster_whisper_adapter"] is True
