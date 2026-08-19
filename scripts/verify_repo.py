"""Check the repository's maintainer-facing contract without mutating it.

This script intentionally performs structural checks only. It does not install
dependencies, contact GitHub, inspect secrets, or change the working tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    ".github/CODEOWNERS",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/feature.yml",
    ".github/ISSUE_TEMPLATE/question.yml",
    ".github/dependabot.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/codeql.yml",
    ".github/workflows/dependency-audit.yml",
    ".github/workflows/model-dependent.yml",
    ".github/workflows/release.yml",
    ".github/workflows/stale.yml",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "MAINTAINERS.md",
    "OPERATIONS.md",
    "ROADMAP.md",
    "SUPPORT.md",
    "docs/github-operations.md",
    "docs/maintenance-backlog.md",
    "docs/releasing.md",
    "docs/runbooks.md",
    ".python-version",
)


def check_required_files() -> list[str]:
    return [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]


def check_manifest() -> list[str]:
    errors: list[str] = []
    path = ROOT / "tests" / "acceptance_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unable to read acceptance manifest: {exc}"]
    mandatory = set(manifest.get("mandatory_suites", []))
    expected = {"unit", "integration", "e2e", "mutation", "packaging"}
    if mandatory != expected:
        errors.append(f"mandatory suites {sorted(mandatory)!r} do not match {sorted(expected)!r}")
    if manifest.get("output_contract", {}).get("markdown_files") != 1:
        errors.append("output contract must require exactly one Markdown file")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="only print failures")
    args = parser.parse_args()

    failures: list[str] = []
    missing = check_required_files()
    failures.extend(f"missing required file: {path}" for path in missing)
    failures.extend(check_manifest())

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    if not args.quiet:
        print(f"Repository maintenance contract verified ({len(REQUIRED_FILES)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
