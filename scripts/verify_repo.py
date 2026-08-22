"""Check the repository's maintainer-facing contract without mutating it.

This script intentionally performs structural checks only. It does not install
dependencies, contact GitHub, inspect secrets, or change the working tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
    "docs/backup-and-restore.md",
    "docs/cli-reference.md",
    "docs/corpus-evaluation.md",
    "docs/model-audit.md",
    "docs/public-contracts.md",
    "docs/github-operations.md",
    "docs/maintenance-backlog.md",
    "docs/performance-benchmarking.md",
    "docs/pypi-trusted-publishing.md",
    "docs/releasing.md",
    "docs/runbooks.md",
    "scripts/backup_repo.py",
    "scripts/benchmark_pipeline.py",
    "scripts/evaluate_corpus.py",
    "scripts/qualify_benchmark.py",
    "scripts/run_model_audit.py",
    "scripts/validate_corpus_manifest.py",
    "scripts/validate_performance_manifest.py",
    "tests/corpus_baseline.json",
    "tests/corpus_manifest.json",
    "tests/model_audit_manifest.json",
    "tests/performance_manifest.json",
    "tests/qualification_policy.example.json",
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


def check_corpus_baseline() -> list[str]:
    """Keep the checked-in quality baseline bound to the current corpus."""

    errors: list[str] = []
    manifest_path = ROOT / "tests" / "corpus_manifest.json"
    baseline_path = ROOT / "tests" / "corpus_baseline.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unable to read corpus baseline or manifest: {exc}"]
    if not isinstance(manifest, dict) or not isinstance(baseline, dict):
        return ["corpus baseline and manifest roots must be objects"]
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_hash = hashlib.sha256(canonical).hexdigest()
    if baseline.get("corpus_hash") != expected_hash:
        errors.append("corpus baseline hash does not match tests/corpus_manifest.json")
    if baseline.get("scoring_version") != "1.0":
        errors.append("corpus baseline scoring_version must be '1.0'")
    cases = baseline.get("cases")
    if not isinstance(cases, dict) or not cases:
        errors.append("corpus baseline cases must be a non-empty object")
    else:
        manifest_ids = {
            str(case.get("id"))
            for case in manifest.get("cases", [])
            if isinstance(case, dict) and case.get("id")
        }
        unknown = set(cases) - manifest_ids
        if unknown:
            errors.append(f"corpus baseline references unknown cases: {sorted(unknown)!r}")
    return errors


def check_model_audit_manifest() -> list[str]:
    """Keep release-audit lanes aligned with the model-dependent contract."""

    errors: list[str] = []
    try:
        acceptance = json.loads(
            (ROOT / "tests" / "acceptance_manifest.json").read_text(encoding="utf-8")
        )
        audit = json.loads(
            (ROOT / "tests" / "model_audit_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unable to read model audit manifest: {exc}"]
    if not isinstance(acceptance, dict) or not isinstance(audit, dict):
        return ["acceptance and model audit manifests must contain objects"]
    if audit.get("schema_version") != "1.0":
        errors.append("model audit manifest schema_version must be '1.0'")
    expected = {
        str(item) for item in acceptance.get("model_dependent", []) if isinstance(item, str)
    }
    lanes = audit.get("lanes", [])
    actual = {str(item.get("id")) for item in lanes if isinstance(item, dict) and item.get("id")}
    if actual != expected:
        errors.append(
            f"model audit lanes {sorted(actual)!r} do not match acceptance models {sorted(expected)!r}"
        )
    for item in lanes if isinstance(lanes, list) else []:
        if not isinstance(item, dict):
            continue
        module = item.get("test_module")
        if not isinstance(module, str) or Path(module).name != module or ".." in Path(module).parts:
            errors.append(f"model audit test_module is not a safe module name: {module!r}")
            continue
        if not (ROOT / "tests" / "model_dependent" / module).is_file():
            errors.append(f"model audit test module is missing: {module}")
    corpus = audit.get("corpus", {})
    if not isinstance(corpus, dict):
        errors.append("model audit corpus must be an object")
    else:
        for key in ("manifest", "baseline"):
            value = corpus.get(key)
            if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
                errors.append(f"model audit corpus {key} must be a safe repository path")
            elif not (ROOT / value).is_file():
                errors.append(f"model audit corpus file is missing: {value}")
    return errors


def check_performance_manifest() -> list[str]:
    """Keep the public performance workload matrix valid and corpus-bound."""

    path = ROOT / "tests" / "performance_manifest.json"
    validator_path = ROOT / "scripts" / "validate_performance_manifest.py"
    try:
        spec = importlib.util.spec_from_file_location(
            "vsr_performance_manifest_validator", validator_path
        )
        if spec is None or spec.loader is None:
            return ["unable to load performance manifest validator"]
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        errors = module.load_and_validate_manifest(path, repo_root=ROOT)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return [f"unable to validate performance manifest: {exc}"]
    return [f"performance manifest: {error}" for error in errors]


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
        "ref: ${{ env.RELEASE_TAG }}",
    )
    return [
        f"release workflow is missing required fragment: {fragment!r}"
        for fragment in required_fragments
        if fragment not in workflow
    ]


def check_pypi_setup_documentation() -> list[str]:
    """Keep the PyPI setup guide aligned with the release contract and security gate."""

    path = ROOT / "docs" / "pypi-trusted-publishing.md"
    try:
        document = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"unable to read PyPI setup documentation: {exc}"]
    required_fragments = (
        "fast-video-analyzer",
        "berdan-labs",
        "release.yml",
        "`pypi`",
        "two-factor authentication",
        "Do not create or store a `PYPI_TOKEN` secret.",
    )
    return [
        f"PyPI setup documentation is missing required fragment: {fragment!r}"
        for fragment in required_fragments
        if fragment not in document
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="only print failures")
    args = parser.parse_args()

    failures: list[str] = []
    missing = check_required_files()
    failures.extend(f"missing required file: {path}" for path in missing)
    failures.extend(check_manifest())
    failures.extend(check_corpus_baseline())
    failures.extend(check_model_audit_manifest())
    failures.extend(check_performance_manifest())
    failures.extend(check_workflow_actions())
    failures.extend(check_release_contract())
    failures.extend(check_pypi_setup_documentation())

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    if not args.quiet:
        print(f"Repository maintenance contract verified ({len(REQUIRED_FILES)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
