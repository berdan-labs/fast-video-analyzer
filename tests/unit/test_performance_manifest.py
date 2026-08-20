from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "tests" / "performance_manifest.json"


def _module() -> Any:
    path = ROOT / "scripts" / "validate_performance_manifest.py"
    spec = importlib.util.spec_from_file_location("vsr_validate_performance_manifest", path)
    if spec is None or spec.loader is None:
        raise AssertionError("performance manifest validator could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_performance_manifest_is_valid() -> None:
    assert _module().load_and_validate_manifest(MANIFEST_PATH, repo_root=ROOT) == []


def test_performance_manifest_rejects_unknown_corpus_case() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["workloads"][0]["case_id"] = "not-a-corpus-case"

    errors = _module().validate_manifest(manifest, repo_root=ROOT)

    assert any("unknown corpus case" in error for error in errors)


def test_performance_manifest_rejects_warm_scenario_without_resume() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["workloads"][0]["scenarios"][1]["resume"] = False

    errors = _module().validate_manifest(manifest, repo_root=ROOT)

    assert any("warm must set resume=true" in error for error in errors)


def test_performance_manifest_rejects_batch_runner_for_pipeline_mode() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["workloads"][0]["scenarios"][0]["runner"] = "batch_cli"

    errors = _module().validate_manifest(manifest, repo_root=ROOT)

    assert any("cold must use runner='benchmark_pipeline'" in error for error in errors)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_performance_manifest_rejects_absolute_corpus_reference(suffix: str) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["corpus_manifest"] = str(ROOT / f"corpus{suffix}")

    errors = _module().validate_manifest(manifest, repo_root=ROOT)

    assert any("corpus_manifest must stay inside" in error for error in errors)
