from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "qualify_benchmark.py"
LANES = [
    "transcript_segments",
    "frames",
    "ocr_observations",
    "script_blocks",
    "timeline",
    "visual_events",
    "evidence_image_metadata",
]


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("vsr_qualify_benchmark", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("qualification evaluator could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _policy() -> dict[str, Any]:
    return {
        "policy_version": 1,
        "workload_id": "dense-five-hour",
        "report_kind": "pipeline-benchmark",
        "schema_version": "1.1",
        "require_cache_reused_false": True,
        "require_shared_cache_disabled": True,
        "min_media_duration_s": 600,
        "quality_contract_sha256": "a" * 64,
        "required_lane_digests": LANES,
        "max_elapsed_seconds": 90,
        "max_p95_seconds": 90,
    }


def _report() -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "report_kind": "pipeline-benchmark",
        "workload_id": "dense-five-hour",
        "cache_reused": False,
        "runtime": {
            "shared_cache_disabled": {
                "VSR_DISABLE_ASR_SHARED_CACHE": True,
                "VSR_DISABLE_VISUAL_SHARED_CACHE": True,
                "VSR_DISABLE_SEMANTIC_SHARED_CACHE": True,
            }
        },
        "validation_valid": True,
        "elapsed_seconds": 80,
        "timing_summary": {"p95_seconds": 85},
        "quality": {
            "available": True,
            "media_duration_ms": 600_000,
            "quality_contract_sha256": "a" * 64,
            "lane_sha256": {lane: "b" * 64 for lane in LANES},
        },
    }


def test_valid_cold_report_qualifies() -> None:
    module = _module()
    assert module.validate_policy(_policy()) == []
    assert module.evaluate_report(_report(), _policy()) == []


def test_missing_p95_fails_closed() -> None:
    module = _module()
    report = _report()
    del report["timing_summary"]["p95_seconds"]
    reasons = module.evaluate_report(report, _policy())
    assert "missing timing_summary.p95_seconds" in reasons


def test_contract_mismatch_and_warm_cache_are_rejected() -> None:
    module = _module()
    report = _report()
    report["cache_reused"] = True
    report["quality"]["quality_contract_sha256"] = "c" * 64
    report["runtime"]["shared_cache_disabled"]["VSR_DISABLE_ASR_SHARED_CACHE"] = False
    reasons = module.evaluate_report(report, _policy())
    assert "cache_reused must be false" in reasons
    assert "quality_contract_sha256 mismatch" in reasons
    assert "VSR_DISABLE_ASR_SHARED_CACHE must be true" in reasons


def test_example_policy_cannot_qualify_a_report(tmp_path: Path) -> None:
    module = _module()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    code, result = module.evaluate_files(
        ROOT / "tests" / "qualification_policy.example.json",
        (report_path,),
    )
    assert code == 5
    assert result["verdict"] == "rejected"
    assert result["reports"][0]["reasons"]


@pytest.mark.parametrize("key", ["quality_contract_sha256", "required_lane_digests"])
def test_policy_rejects_missing_required_key(key: str) -> None:
    module = _module()
    policy = _policy()
    del policy[key]
    assert any(key in error for error in module.validate_policy(policy))
