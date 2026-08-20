from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "tests" / "corpus_manifest.json"
BASELINE_PATH = ROOT / "tests" / "corpus_baseline.json"


def _module() -> Any:
    path = ROOT / "scripts" / "evaluate_corpus.py"
    spec = importlib.util.spec_from_file_location("vsr_evaluate_corpus", path)
    if spec is None or spec.loader is None:
        raise AssertionError("corpus evaluator could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs() -> tuple[Any, Any, Any, dict[str, dict[str, Any]]]:
    module = _module()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    reports: dict[str, dict[str, Any]] = {}
    for case_id, target in baseline["cases"].items():
        quality = target["quality"]
        reports[case_id] = {
            "quality": {
                "production_valid": True,
                "expected_spoken_count": quality["expected_spoken_count"],
                "spoken_matches": quality["expected_spoken_count"],
                "expected_token_count": quality["expected_token_count"],
                "token_matches": quality["expected_token_count"],
                "minimum_visual_events": quality["minimum_visual_events"],
                "visual_event_count": quality["minimum_visual_events"],
                "errors": [],
            },
            "elapsed_seconds": 1.0,
        }
    return module, manifest, baseline, reports


def test_evaluator_accepts_compatible_quality_and_reports_uncompared_timing() -> None:
    module, manifest, baseline, reports = _inputs()

    result = module.evaluate_reports(manifest, baseline, reports)

    assert result["gate_pass"] is True
    assert result["quality"]["pass"] is True
    assert result["performance"]["pass"] is None
    assert all(item["status"] == "not_compared" for item in result["performance"]["cases"])


@pytest.mark.parametrize(
    ("metric", "value", "needle"),
    [
        ("spoken_matches", 1, "spoken_matches"),
        ("token_matches", 0, "token_matches"),
        ("visual_event_count", 0, "visual_event_count"),
    ],
)
def test_evaluator_catches_quality_regressions(metric: str, value: int, needle: str) -> None:
    module, manifest, baseline, reports = _inputs()
    case_id = "generated-slide-lecture"
    reports[case_id]["quality"][metric] = value

    result = module.evaluate_reports(manifest, baseline, reports)

    assert result["gate_pass"] is False
    case = next(item for item in result["quality"]["cases"] if item["case_id"] == case_id)
    assert any(needle in error for error in case["errors"])


def test_evaluator_catches_invalid_citations_through_production_validation() -> None:
    module, manifest, baseline, reports = _inputs()
    case_id = "generated-screen-tutorial"
    reports[case_id]["quality"]["production_valid"] = False

    result = module.evaluate_reports(manifest, baseline, reports)

    case = next(item for item in result["quality"]["cases"] if item["case_id"] == case_id)
    assert result["gate_pass"] is False
    assert "production output validation failed" in case["errors"]


def test_evaluator_rejects_incompatible_baseline_identity() -> None:
    module, manifest, baseline, reports = _inputs()
    incompatible = copy.deepcopy(baseline)
    incompatible["model_revision"] = "faster-whisper-large-v3@different"

    result = module.evaluate_reports(manifest, incompatible, reports)

    assert result["identity"]["compatible"] is False
    assert "baseline model_revision does not match this evaluation" in result["identity"]["errors"]
    assert result["gate_pass"] is False


def test_evaluator_fails_a_fifteen_percent_performance_regression() -> None:
    module, manifest, baseline, reports = _inputs()
    compared = copy.deepcopy(baseline)
    compared["performance"]["baseline_elapsed_seconds"] = {
        case_id: 10.0 for case_id in compared["cases"]
    }
    reports["generated-screen-tutorial"]["timing_summary"] = {"median_seconds": 11.6}
    reports["generated-screen-tutorial"]["elapsed_seconds"] = 11.6

    result = module.evaluate_reports(
        manifest,
        compared,
        reports,
        require_performance=True,
    )

    case = next(
        item
        for item in result["performance"]["cases"]
        if item["case_id"] == "generated-screen-tutorial"
    )
    assert result["gate_pass"] is False
    assert case["status"] == "compared"
    assert case["pass"] is False
    assert case["regression_fraction"] == pytest.approx(0.16)


def test_markdown_report_is_compact_and_actionable() -> None:
    module, manifest, baseline, reports = _inputs()
    report = module.evaluate_reports(manifest, baseline, reports)

    markdown = module.render_markdown(report)

    assert "# Corpus evaluation" in markdown
    assert "| `generated-screen-tutorial` | pass |" in markdown
    assert "Performance comparison: `None`" in markdown
