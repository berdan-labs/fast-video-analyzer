from __future__ import annotations

import argparse
from pathlib import Path

from video_script_reconstructor.cli import _parser

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "public-contracts.md"
README = ROOT / "README.md"

EXPECTED_TOP_LEVEL_COMMANDS = {
    "batch",
    "cache",
    "diagnostic-bundle",
    "diagnostics",
    "doctor",
    "evidence",
    "finalize",
    "models",
    "plan",
    "retention",
    "review",
    "run",
    "semantic",
    "semantic-batch",
    "support-bundle",
    "validate",
    "workers",
}


def _subparser_choices(parser: argparse.ArgumentParser) -> set[str]:
    action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return set(action.choices)


def test_top_level_cli_commands_are_documented() -> None:
    contract = CONTRACT.read_text(encoding="utf-8")
    actual = _subparser_choices(_parser())

    assert actual == EXPECTED_TOP_LEVEL_COMMANDS
    assert all(command in contract for command in actual)
    assert "diagnostics" in contract
    assert "support-bundle" in contract


def test_review_bundle_aliases_and_entrypoints_are_documented() -> None:
    contract = CONTRACT.read_text(encoding="utf-8")
    cli_reference = (ROOT / "docs" / "cli-reference.md").read_text(encoding="utf-8")

    for alias in ("batch-create", "create-batch"):
        assert alias in contract
        assert alias in cli_reference
    for entrypoint in (
        "fast-video-analyzer",
        "long-video-analyzer",
        "video-script-reconstructor",
    ):
        assert entrypoint in contract
        assert entrypoint in cli_reference


def test_public_docs_keep_python_imports_provisional() -> None:
    contract = CONTRACT.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert "## Python API (provisional)" in readme
    assert "docs/public-contracts.md" in readme
    assert "video_script_reconstructor.pipeline.run_pipeline" in contract
    assert "**provisional**" in contract


def test_result_and_schema_contracts_are_present() -> None:
    contract = CONTRACT.read_text(encoding="utf-8")

    for exit_code in ("`0`", "`1`", "`2`", "`3`", "`4`"):
        assert exit_code in contract
    for result_key in ("project_dir", "markdown", "status", "validation_errors"):
        assert result_key in contract
    for schema in ("configs/schema.json", "configs/strict.yaml", "configs/balanced.yaml"):
        assert schema in contract
