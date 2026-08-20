"""Validate the public evaluation-corpus manifest and tracked source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SOURCE_KINDS = {"generated", "licensed", "owner_controlled"}
AVAILABILITIES = {"repository", "external"}
COVERAGE_STATUSES = {"covered", "gap"}
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".srt",
    ".tsv",
    ".txt",
    ".vtt",
    ".yaml",
    ".yml",
}


def _is_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return None
    return value


def _string_list(value: Any, label: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list) or not all(_is_string(item) for item in value):
        errors.append(f"{label} must be a non-null list of strings")
        return None
    return [str(item) for item in value]


def _safe_path(root: Path, value: Any, label: str, errors: list[str]) -> Path | None:
    if not _is_string(value):
        errors.append(f"{label} must be a relative path")
        return None
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{label} must stay inside the repository: {value!r}")
        return None
    root = root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        errors.append(f"{label} must stay inside the repository: {value!r}")
        return None
    return resolved


def _sha256(path: Path) -> str:
    if path.suffix.casefold() in TEXT_SUFFIXES:
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return hashlib.sha256(content).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact(
    artifact: Any,
    *,
    root: Path,
    label: str,
    availability: str,
    expected_external_reference: bool,
    verify_files: bool,
    errors: list[str],
    seen_paths: set[Path],
) -> None:
    item = _mapping(artifact, label, errors)
    if item is None:
        return
    digest = item.get("sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        errors.append(f"{label}.sha256 must be a 64-character hexadecimal digest")

    raw_path = item.get("path")
    if raw_path is None:
        if availability == "repository":
            errors.append(f"{label}.path is required for repository media")
        elif expected_external_reference and not _is_string(item.get("source_reference")):
            errors.append(f"{label}.source_reference is required for external media")
        return
    path = _safe_path(root, raw_path, f"{label}.path", errors)
    if path is None:
        return
    if path in seen_paths:
        errors.append(f"{label}.path is duplicated: {raw_path!r}")
    seen_paths.add(path)
    if not path.is_file():
        if availability == "repository":
            errors.append(f"{label}.path does not exist: {raw_path!r}")
        return
    if verify_files and isinstance(digest, str) and SHA256_RE.fullmatch(digest):
        actual = _sha256(path)
        if actual.casefold() != digest.casefold():
            errors.append(f"{label}.sha256 does not match {raw_path!r}")


def validate_manifest(
    manifest: Mapping[str, Any], *, repo_root: Path, verify_files: bool = True
) -> list[str]:
    """Return human-readable validation errors for one parsed manifest."""

    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if not _is_string(manifest.get("manifest_id")):
        errors.append("manifest_id must be a non-empty string")
    if manifest.get("hash_algorithm") != "sha256":
        errors.append("hash_algorithm must be 'sha256'")

    policy = _mapping(manifest.get("source_policy"), "source_policy", errors)
    if policy is not None:
        allowed = _string_list(policy.get("allowed_kinds"), "source_policy.allowed_kinds", errors)
        if allowed is not None and set(allowed) != SOURCE_KINDS:
            errors.append(
                "source_policy.allowed_kinds must list generated, licensed, owner_controlled"
            )

    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        errors.append("cases must be a non-empty list")
        raw_cases = []
    case_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for index, raw_case in enumerate(raw_cases):
        label = f"cases[{index}]"
        case = _mapping(raw_case, label, errors)
        if case is None:
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
            errors.append(f"{label}.id must be a lowercase slug")
        elif case_id in case_ids:
            errors.append(f"duplicate case id: {case_id!r}")
        else:
            case_ids.add(case_id)
        source_kind = case.get("source_kind")
        if source_kind not in SOURCE_KINDS:
            errors.append(f"{label}.source_kind must be one of {sorted(SOURCE_KINDS)!r}")
        availability = case.get("availability")
        if availability not in AVAILABILITIES:
            errors.append(f"{label}.availability must be 'repository' or 'external'")
            availability = "repository"
        tags = _string_list(case.get("tags"), f"{label}.tags", errors)
        if tags is not None and len(tags) != len(set(tags)):
            errors.append(f"{label}.tags must not contain duplicates")
        provenance = _mapping(case.get("provenance"), f"{label}.provenance", errors)
        if provenance is not None:
            if source_kind == "generated":
                for key in ("generator", "command", "origin"):
                    if not _is_string(provenance.get(key)):
                        errors.append(f"{label}.provenance.{key} is required for generated media")
            elif source_kind == "licensed":
                for key in ("source_reference", "license", "attribution"):
                    if not _is_string(provenance.get(key)):
                        errors.append(f"{label}.provenance.{key} is required for licensed media")
            elif source_kind == "owner_controlled":
                for key in ("source_reference", "owner_confirmation_ref"):
                    if not _is_string(provenance.get(key)):
                        errors.append(
                            f"{label}.provenance.{key} is required for owner-controlled media"
                        )
        if source_kind == "generated" and availability != "repository":
            errors.append(f"{label} generated media must be available in the repository")
        media_label = f"{label}.media"
        _validate_artifact(
            case.get("media"),
            root=repo_root,
            label=media_label,
            availability=str(availability),
            expected_external_reference=availability == "external",
            verify_files=verify_files,
            errors=errors,
            seen_paths=seen_paths,
        )
        subtitles = case.get("subtitles", [])
        if not isinstance(subtitles, list):
            errors.append(f"{label}.subtitles must be a list")
        else:
            for subtitle_index, subtitle in enumerate(subtitles):
                _validate_artifact(
                    subtitle,
                    root=repo_root,
                    label=f"{label}.subtitles[{subtitle_index}]",
                    availability=str(availability),
                    expected_external_reference=availability == "external",
                    verify_files=verify_files,
                    errors=errors,
                    seen_paths=seen_paths,
                )
        expected = _mapping(case.get("expected"), f"{label}.expected", errors)
        if expected is not None:
            _string_list(expected.get("spoken"), f"{label}.expected.spoken", errors)
            _string_list(expected.get("tokens"), f"{label}.expected.tokens", errors)
            minimum_visual_events = expected.get("minimum_visual_events")
            if not isinstance(minimum_visual_events, int) or isinstance(
                minimum_visual_events, bool
            ):
                errors.append(
                    f"{label}.expected.minimum_visual_events must be a non-negative integer"
                )
            elif minimum_visual_events < 0:
                errors.append(
                    f"{label}.expected.minimum_visual_events must be a non-negative integer"
                )

    coverage = manifest.get("coverage_requirements")
    if not isinstance(coverage, list) or not coverage:
        errors.append("coverage_requirements must be a non-empty list")
    else:
        coverage_ids: set[str] = set()
        referenced_cases: set[str] = set()
        for index, raw_requirement in enumerate(coverage):
            label = f"coverage_requirements[{index}]"
            requirement = _mapping(raw_requirement, label, errors)
            if requirement is None:
                continue
            requirement_id = requirement.get("id")
            if not isinstance(requirement_id, str) or not CASE_ID_RE.fullmatch(requirement_id):
                errors.append(f"{label}.id must be a lowercase slug")
            elif requirement_id in coverage_ids:
                errors.append(f"duplicate coverage requirement id: {requirement_id!r}")
            else:
                coverage_ids.add(requirement_id)
            status = requirement.get("status")
            if status not in COVERAGE_STATUSES:
                errors.append(f"{label}.status must be 'covered' or 'gap'")
            case_list = _string_list(requirement.get("case_ids"), f"{label}.case_ids", errors)
            if case_list is not None:
                unknown = set(case_list) - case_ids
                if unknown:
                    errors.append(f"{label}.case_ids references unknown cases: {sorted(unknown)!r}")
                referenced_cases.update(case_list)
                if status == "covered" and not case_list:
                    errors.append(f"{label} marked covered but has no case_ids")
                if status == "gap" and case_list:
                    errors.append(f"{label} marked gap but lists case_ids")
        unreferenced = case_ids - referenced_cases
        if unreferenced:
            errors.append(
                f"cases are not referenced by coverage requirements: {sorted(unreferenced)!r}"
            )

    return errors


def load_and_validate_manifest(
    manifest_path: str | Path, *, repo_root: str | Path | None = None, verify_files: bool = True
) -> list[str]:
    path = Path(manifest_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else path.parent.parent.resolve()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"could not read manifest {path}: {exc}"]
    if not isinstance(loaded, Mapping):
        return ["manifest root must be an object"]
    return validate_manifest(loaded, repo_root=root, verify_files=verify_files)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Repository root used to resolve tracked artifact paths (default: manifest parent).",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Validate structure without comparing hashes of files that are present.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    errors = load_and_validate_manifest(
        args.manifest,
        repo_root=args.repo_root,
        verify_files=not args.skip_hash,
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Corpus manifest verified: {Path(args.manifest).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
