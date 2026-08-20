from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "tests" / "corpus_manifest.json"


def _manifest_module() -> Any:
    path = ROOT / "scripts" / "validate_corpus_manifest.py"
    spec = importlib.util.spec_from_file_location("vsr_validate_corpus_manifest", path)
    if spec is None or spec.loader is None:
        raise AssertionError("corpus manifest validator could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_corpus_manifest_and_hashes_are_valid() -> None:
    assert _manifest_module().load_and_validate_manifest(MANIFEST_PATH) == []


def test_manifest_reports_hash_drift(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["cases"][0]["media"]["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = _manifest_module().load_and_validate_manifest(path, repo_root=ROOT)

    assert any("cases[0].media.sha256 does not match" in error for error in errors)
    assert (
        _manifest_module().load_and_validate_manifest(path, repo_root=ROOT, verify_files=False)
        == []
    )


def test_external_case_requires_reference_but_not_a_checked_out_file() -> None:
    manifest = {
        "schema_version": "1.0",
        "manifest_id": "external-case",
        "hash_algorithm": "sha256",
        "source_policy": {"allowed_kinds": ["generated", "licensed", "owner_controlled"]},
        "coverage_requirements": [
            {"id": "licensed-long-form", "status": "covered", "case_ids": ["licensed-case"]}
        ],
        "cases": [
            {
                "id": "licensed-case",
                "source_kind": "licensed",
                "availability": "external",
                "license": "CC-BY-4.0",
                "tags": ["long-duration"],
                "provenance": {
                    "source_reference": "corpus-store://licensed-long-form",
                    "license": "CC-BY-4.0",
                    "attribution": "Example rights holder",
                },
                "media": {
                    "sha256": "a" * 64,
                    "source_reference": "corpus-store://licensed-long-form",
                },
                "subtitles": [],
                "expected": {"spoken": [], "tokens": []},
            }
        ],
    }

    assert _manifest_module().validate_manifest(manifest, repo_root=ROOT) == []


def test_manifest_rejects_absolute_artifact_paths() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["cases"][0]["media"]["path"] = str(
        ROOT / "tests" / "fixtures" / "generated" / "talking-head.mp4"
    )

    errors = _manifest_module().validate_manifest(manifest, repo_root=ROOT)

    assert any("cases[0].media.path must stay inside" in error for error in errors)


@pytest.mark.parametrize("status", ["covered", "gap"])
def test_coverage_statuses_are_explicit(status: str) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    requirement = manifest["coverage_requirements"][0]
    requirement["status"] = status
    requirement["case_ids"] = ["generated-talking-head"] if status == "covered" else []

    assert _manifest_module().validate_manifest(manifest, repo_root=ROOT) == []
