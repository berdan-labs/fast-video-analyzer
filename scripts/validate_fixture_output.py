"""Validate a generated fixture through the production output validator.

This wrapper intentionally delegates artifact, schema, metadata, link, and audit
checks to :func:`video_script_reconstructor.validate_output.validate_project`.
It adds only fixture expectations such as exact spoken sentences and high-impact
tokens; it does not maintain a second implementation of the output contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from video_script_reconstructor.validate_output import validate_project


def validate_fixture_output(
    project_dir: str | Path,
    *,
    expected_spoken: Sequence[str] = (),
    expected_tokens: Sequence[str] = (),
    expected_status: str | None = None,
) -> dict[str, Any]:
    """Run production validation and deterministic fixture-content checks."""

    root = Path(project_dir).resolve(strict=True)
    production = validate_project(root)
    errors = list(production.errors)
    markdown_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() == ".md"
    )
    text = markdown_files[0].read_text(encoding="utf-8") if len(markdown_files) == 1 else ""

    canonical_path = root / ".state" / "canonical-project.json"
    canonical: dict[str, Any] | None = None
    if canonical_path.is_file():
        try:
            loaded = json.loads(canonical_path.read_text(encoding="utf-8"))
            canonical = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError):
            canonical = None
    evidence_text_parts: list[str] = []
    if canonical is not None:
        for block in canonical.get("script_blocks", []):
            evidence_text_parts.append(str(block.get("spoken_text") or ""))
            evidence_text_parts.extend(str(value) for value in block.get("on_screen_text", []))
        if not evidence_text_parts:
            for segment in canonical.get("transcript_segments", []):
                for key in ("human_verified_text", "repaired_text", "normalized_text", "raw_text"):
                    value = segment.get(key)
                    if value:
                        evidence_text_parts.append(str(value))
                        break
    contract_text = "\n".join(evidence_text_parts) or text
    positions: list[int] = []
    for sentence in expected_spoken:
        count = contract_text.count(sentence)
        if count != 1:
            errors.append(
                f"fixture_content: expected spoken sentence {sentence!r} exactly once, found {count}"
            )
        positions.append(contract_text.find(sentence))
    if positions and (
        any(position < 0 for position in positions) or positions != sorted(positions)
    ):
        errors.append("fixture_content: expected spoken sentences are absent or out of order")

    for token in expected_tokens:
        if token not in contract_text:
            errors.append(f"fixture_content: expected exact token {token!r} is missing")

    if expected_status is not None:
        actual_status = canonical.get("project_status") if canonical else None
        if actual_status != expected_status:
            errors.append(
                f"fixture_content: expected status {expected_status!r}, found {actual_status!r}"
            )

    return {
        "valid": not errors,
        "project_dir": str(root),
        "production_valid": production.valid,
        "errors": errors,
        "warnings": production.warnings,
        "checks": {
            **production.checks,
            "expected_spoken_count": len(expected_spoken),
            "expected_token_count": len(expected_tokens),
            "expected_status": expected_status,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate generated fixture output through the production validator."
    )
    parser.add_argument("project_dir", type=Path)
    parser.add_argument(
        "--expect-spoken",
        action="append",
        default=[],
        help="Sentence that must occur exactly once; repeat to assert chronological order.",
    )
    parser.add_argument(
        "--expect-token",
        action="append",
        default=[],
        help="Exact high-impact token that must remain visible; repeat as needed.",
    )
    parser.add_argument("--expect-status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate_fixture_output(
            args.project_dir,
            expected_spoken=args.expect_spoken,
            expected_tokens=args.expect_token,
            expected_status=args.expect_status,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"valid": False, "errors": [str(exc)]}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
