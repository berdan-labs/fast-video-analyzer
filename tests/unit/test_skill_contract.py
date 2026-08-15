from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_skill_frontmatter_and_trigger_fixture_counts() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    _, frontmatter, body = skill.split("---", 2)
    metadata = yaml.safe_load(frontmatter)
    assert set(metadata) == {"name", "description"}
    assert metadata["name"] == "long-video-analyzer"
    assert len(body.splitlines()) < 500
    evaluation = json.loads(
        (ROOT / "tests" / "skill_trigger_eval.json").read_text(encoding="utf-8")
    )
    assert len(evaluation["positive"]) >= 20
    assert len(evaluation["negative"]) >= 15
    assert len(evaluation["ambiguous"]) >= 10


def test_mandatory_tests_have_no_unconditional_assertions() -> None:
    offenders: list[str] = []
    unconditional = "assert" + " True"
    for directory in ("unit", "integration", "e2e", "mutation", "packaging"):
        for path in (ROOT / "tests" / directory).glob("test_*.py"):
            if unconditional in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
