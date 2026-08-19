"""Check the repository's maintainer-facing contract without mutating it.

This script intentionally performs structural checks only. It does not install
dependencies, contact GitHub, inspect secrets, or change the working tree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

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
    ".github/workflows/platform-smoke.yml",
    ".github/workflows/release.yml",
    ".github/workflows/stale.yml",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "MAINTAINERS.md",
    "OPERATIONS.md",
    "ROADMAP.md",
    "SUPPORT.md",
    "docs/action-allowlist.md",
    "docs/cli-reference.md",
    "docs/github-operations.md",
    "docs/maintenance-backlog.md",
    "docs/releasing.md",
    "docs/runbooks.md",
    ".python-version",
)

ALLOWED_ACTION_REPOSITORIES = frozenset(
    {
        "actions/attest-build-provenance",
        "actions/checkout",
        "actions/download-artifact",
        "actions/setup-python",
        "actions/stale",
        "actions/upload-artifact",
        "astral-sh/setup-uv",
        "github/codeql-action/analyze",
        "github/codeql-action/init",
        "pypa/gh-action-pypi-publish",
    }
)
_PINNED_ACTION = re.compile(r"^(?P<repository>[^@\s]+)@(?P<sha>[0-9a-f]{40})$")


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


def _workflow_uses(value: Any) -> list[str]:
    """Collect parsed workflow ``uses`` values without inspecting shell text."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "uses" and isinstance(item, str):
                found.append(item)
            found.extend(_workflow_uses(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_workflow_uses(item))
    return found


def check_workflow_actions() -> list[str]:
    """Require every workflow action to be in the documented SHA-pinned allowlist."""

    errors: list[str] = []
    workflow_paths = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
    seen: set[str] = set()
    for path in workflow_paths:
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"unable to parse workflow {path.relative_to(ROOT)}: {exc}")
            continue
        for reference in _workflow_uses(document):
            match = _PINNED_ACTION.fullmatch(reference)
            if match is None:
                errors.append(
                    f"workflow action is not pinned to a 40-hex commit: "
                    f"{path.relative_to(ROOT)} uses {reference!r}"
                )
                continue
            repository = match.group("repository")
            seen.add(repository)
            if repository not in ALLOWED_ACTION_REPOSITORIES:
                errors.append(
                    f"workflow action is outside the allowlist: "
                    f"{path.relative_to(ROOT)} uses {repository!r}"
                )
    try:
        allowlist = (ROOT / "docs" / "action-allowlist.md").read_text(encoding="utf-8")
    except OSError as exc:
        return [f"unable to read action allowlist: {exc}"]
    for repository in sorted(seen):
        if repository not in allowlist:
            errors.append(f"action allowlist does not document {repository!r}")
    return errors


def check_release_contract() -> list[str]:
    """Keep SBOM/release metadata separate from PyPI upload payloads."""

    path = ROOT / ".github" / "workflows" / "release.yml"
    try:
        workflow = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"unable to read release workflow: {exc}"]
    required_fragments = (
        "pip-audit",
        "cyclonedx-json",
        "fast-video-analyzer-sbom.cdx.json",
        "mkdir -p dist/packages",
        "packages-dir: dist/packages/",
        "subject-path: dist/packages/*",
    )
    return [
        f"release workflow is missing required fragment: {fragment!r}"
        for fragment in required_fragments
        if fragment not in workflow
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="only print failures")
    args = parser.parse_args()

    failures: list[str] = []
    missing = check_required_files()
    failures.extend(f"missing required file: {path}" for path in missing)
    failures.extend(check_manifest())
    failures.extend(check_workflow_actions())
    failures.extend(check_release_contract())

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    if not args.quiet:
        print(f"Repository maintenance contract verified ({len(REQUIRED_FILES)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
