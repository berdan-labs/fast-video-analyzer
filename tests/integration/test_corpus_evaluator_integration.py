from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    path = ROOT / "scripts" / "evaluate_corpus.py"
    spec = importlib.util.spec_from_file_location("vsr_evaluate_corpus_integration", path)
    if spec is None or spec.loader is None:
        raise AssertionError("corpus evaluator could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_seed_runs_through_manifest_evaluator(tmp_path: Path) -> None:
    module = _module()
    report = module.run_corpus(
        ROOT / "tests" / "corpus_manifest.json",
        ROOT / "tests" / "corpus_baseline.json",
        tmp_path / "outputs",
        vision_mode="none",
        model_revision="deterministic-fixtures",
    )

    assert report["gate_pass"] is True
    assert report["quality"]["pass"] is True
    assert set(report["cases"]) == {
        "generated-talking-head",
        "generated-slide-lecture",
        "generated-screen-tutorial",
        "generated-hostile-subtitle",
        "generated-caption-variants",
    }

    caption_project = Path(report["cases"]["generated-caption-variants"]["project_dir"])
    canonical = json.loads(
        (caption_project / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    candidates = canonical["transcript_candidates"]
    assert {item["origin"] for item in candidates} == {
        "caption-variants.vtt",
        "caption-variants.ass",
    }
    assert {item["segments"][0]["timing_provenance"] for item in candidates} == {
        "source_vtt",
        "source_ass",
    }
    selected = [item for item in candidates if item["decision_rationale"].startswith("Selected as")]
    assert len(selected) == 1
