from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _audit_module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "run_model_audit.py"
    spec = importlib.util.spec_from_file_location("vsr_model_audit", path)
    if spec is None or spec.loader is None:
        raise AssertionError("model audit script could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_audit_manifest_matches_all_required_lanes() -> None:
    module = _audit_module()
    manifest = module.load_audit_manifest(Path(__file__).parents[1] / "model_audit_manifest.json")
    assert {lane["id"] for lane in manifest["lanes"]} == {
        "qwen3-asr-1.7b",
        "qwen3-forced-aligner-0.6b",
        "moss-transcribe-diarize-0.9b",
        "pp-ocrv5-server",
        "tesseract",
        "faster-whisper-large-v3",
        "qwen3-vl-4b-q4",
    }


def test_audit_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    module = _audit_module()
    manifest = {
        "schema_version": "1.0",
        "pytest_target": "tests/model_dependent",
        "lanes": [
            {
                "id": "unsafe",
                "description": "unsafe",
                "models": [],
                "test_module": "../secrets.py",
            }
        ],
        "corpus": {
            "manifest": "tests/corpus_manifest.json",
            "baseline": "tests/corpus_baseline.json",
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="repository"):
        module.load_audit_manifest(path)


def test_parse_junit_and_classify_unavailable_lane(tmp_path: Path) -> None:
    module = _audit_module()
    junit = tmp_path / "results.xml"
    junit.write_text(
        """<testsuites><testsuite>
        <testcase classname="tests.model_dependent.test_tesseract_ocr" name="test_ocr" />
        <testcase classname="tests.model_dependent.test_semantic_vision" name="test_vision">
          <skipped message="backend unavailable: C:\\private\\worker.exe" />
        </testcase>
        </testsuite></testsuites>""",
        encoding="utf-8",
    )
    cases = module.parse_junit(junit)
    assert [case["status"] for case in cases] == ["pass", "unavailable"]
    assert "[REDACTED_PATH]" in cases[1]["reason"]
    lane = module._classify_lane(
        {
            "id": "qwen3-vl-4b-q4",
            "description": "vision",
            "models": ["qwen3-vl-4b-q4"],
            "test_module": "test_semantic_vision.py",
        },
        cases,
    )
    assert lane["status"] == "unavailable"
    assert lane["tests"] == 1


def test_build_report_never_passes_with_unavailable_lane() -> None:
    module = _audit_module()
    report = module.build_report(
        {"lanes": [{"id": "tesseract"}]},
        runtime={"git_revision": "a" * 40},
        model_tests={
            "status": "unavailable",
            "lanes": [{"id": "tesseract", "status": "unavailable"}],
        },
        corpus={"status": "pass", "gate_pass": True},
    )
    assert report["required_lanes_pass"] is False
    assert report["corpus_gate_pass"] is True
    assert report["gate_pass"] is False


def test_render_markdown_contains_statuses_but_not_local_paths() -> None:
    module = _audit_module()
    markdown = module.render_markdown(
        {
            "gate_pass": False,
            "repository_revision": "a" * 40,
            "runtime": {
                "application_version": "0.1.0",
                "python": "3.12.0",
                "os": "Windows",
                "machine": "AMD64",
                "nvidia_smi_present": False,
            },
            "model_tests": {
                "lanes": [
                    {
                        "id": "tesseract",
                        "status": "pass",
                        "models": [],
                        "tests": 1,
                        "reason": None,
                    }
                ]
            },
            "corpus": {
                "status": "fail",
                "corpus_hash": "b" * 64,
                "model_revision": "deterministic-fixtures",
                "quality_pass": False,
                "performance_pass": None,
                "performance_required": False,
                "case_count": 3,
                "failed_cases": [{"case_id": "slide-lecture", "errors": ["invalid output"]}],
            },
        }
    )
    assert "**PASS**" in markdown
    assert "**FAIL**" in markdown
    assert "C:\\private" not in markdown
    assert "model-audit.json" not in markdown
