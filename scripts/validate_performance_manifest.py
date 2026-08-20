"""Validate the public performance workload matrix.

The matrix freezes comparable workload definitions and relative regression
budgets. Host-specific timings, hardware identifiers, model paths, and raw
benchmark output stay outside the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
HARDWARE_CLASSES = {"cpu", "cuda", "storage-constrained"}
RUNNERS = {"benchmark_pipeline", "batch_cli"}
MODES = {"cold", "warm", "validation", "batch"}
ROOT = Path(__file__).resolve().parents[1]


def _is_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return None
    return value


def _string_list(value: Any, label: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list) or not all(_is_string(item) for item in value):
        errors.append(f"{label} must be a non-empty list of strings")
        return None
    if not value:
        errors.append(f"{label} must not be empty")
        return None
    return [str(item) for item in value]


def _optional_string_list(value: Any, label: str, errors: list[str]) -> None:
    if value is not None and (
        not isinstance(value, list) or not all(_is_string(item) for item in value)
    ):
        errors.append(f"{label} must be a list of strings")


def _safe_path(root: Path, value: Any, label: str, errors: list[str]) -> Path | None:
    if not _is_string(value):
        errors.append(f"{label} must be a relative repository path")
        return None
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{label} must stay inside the repository: {value!r}")
        return None
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        errors.append(f"{label} must stay inside the repository: {value!r}")
        return None
    return resolved


def _load_json(path: Path, label: str, errors: list[str]) -> Mapping[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} could not be read: {exc}")
        return None
    return _mapping(loaded, label, errors)


def _fraction(value: Any, label: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        errors.append(f"{label} must be a non-negative number")


def validate_manifest(
    manifest: Mapping[str, Any], *, repo_root: Path, verify_references: bool = True
) -> list[str]:
    """Return human-readable errors for one performance manifest."""

    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if not _is_string(manifest.get("manifest_id")):
        errors.append("manifest_id must be a non-empty string")

    corpus_path = _safe_path(repo_root, manifest.get("corpus_manifest"), "corpus_manifest", errors)
    corpus: Mapping[str, Any] | None = None
    if corpus_path is not None and verify_references:
        corpus = _load_json(corpus_path, "corpus_manifest", errors)
    corpus_case_ids = {
        str(item.get("id"))
        for item in (corpus or {}).get("cases", [])
        if isinstance(item, Mapping) and item.get("id")
    }

    policy = _mapping(manifest.get("baseline_policy"), "baseline_policy", errors)
    if policy is not None:
        if not _is_string(policy.get("comparison_unit")):
            errors.append("baseline_policy.comparison_unit must be a non-empty string")
        for key in (
            "max_wall_time_regression_fraction",
            "max_peak_rss_regression_fraction",
            "max_output_bytes_regression_fraction",
        ):
            _fraction(policy.get(key), f"baseline_policy.{key}", errors)
        if not isinstance(policy.get("quality_gate_required"), bool):
            errors.append("baseline_policy.quality_gate_required must be a boolean")
        if not isinstance(policy.get("host_baselines_belong_in_owner_local_storage"), bool):
            errors.append(
                "baseline_policy.host_baselines_belong_in_owner_local_storage must be a boolean"
            )

    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        errors.append("profiles must be a non-empty list")
        profiles = []
    profile_ids: set[str] = set()
    for index, raw_profile in enumerate(profiles):
        label = f"profiles[{index}]"
        profile = _mapping(raw_profile, label, errors)
        if profile is None:
            continue
        profile_id = profile.get("id")
        if not _is_string(profile_id):
            errors.append(f"{label}.id must be a non-empty string")
        elif profile_id in profile_ids:
            errors.append(f"duplicate profile id: {profile_id!r}")
        else:
            profile_ids.add(str(profile_id))
        if profile.get("hardware_class") not in HARDWARE_CLASSES:
            errors.append(f"{label}.hardware_class must be one of {sorted(HARDWARE_CLASSES)!r}")
        _string_list(profile.get("required_capabilities"), f"{label}.required_capabilities", errors)
        if not _is_string(profile.get("notes")):
            errors.append(f"{label}.notes must be a non-empty string")

    workloads = manifest.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        errors.append("workloads must be a non-empty list")
        workloads = []
    workload_ids: set[str] = set()
    for index, raw_workload in enumerate(workloads):
        label = f"workloads[{index}]"
        workload = _mapping(raw_workload, label, errors)
        if workload is None:
            continue
        workload_id = workload.get("id")
        if not _is_string(workload_id):
            errors.append(f"{label}.id must be a non-empty string")
        elif workload_id in workload_ids:
            errors.append(f"duplicate workload id: {workload_id!r}")
        else:
            workload_ids.add(str(workload_id))
        case_id = workload.get("case_id")
        if not _is_string(case_id):
            errors.append(f"{label}.case_id must be a non-empty string")
        elif corpus_case_ids and str(case_id) not in corpus_case_ids:
            errors.append(f"{label}.case_id references an unknown corpus case: {case_id!r}")
        expected_case = workload.get("expected_quality_case", case_id)
        if expected_case != case_id:
            errors.append(f"{label}.expected_quality_case must match case_id")
        if not _is_string(workload.get("input_mode")):
            errors.append(f"{label}.input_mode must be a non-empty string")
        profile_list = _string_list(workload.get("profile_ids"), f"{label}.profile_ids", errors)
        if profile_list is not None:
            if len(profile_list) != len(set(profile_list)):
                errors.append(f"{label}.profile_ids must not contain duplicates")
            unknown_profiles = set(profile_list) - profile_ids
            if unknown_profiles:
                errors.append(
                    f"{label}.profile_ids references unknown profiles: {sorted(unknown_profiles)!r}"
                )
        if workload.get("preset") not in {"strict", "balanced"}:
            errors.append(f"{label}.preset must be 'strict' or 'balanced'")
        if workload.get("vision_mode") not in {"none", "host-agent", "auto", "local"}:
            errors.append(
                f"{label}.vision_mode must be one of ['auto', 'host-agent', 'local', 'none']"
            )

        scenarios = workload.get("scenarios")
        if not isinstance(scenarios, list) or not scenarios:
            errors.append(f"{label}.scenarios must be a non-empty list")
            continue
        scenario_ids: set[str] = set()
        for scenario_index, raw_scenario in enumerate(scenarios):
            scenario_label = f"{label}.scenarios[{scenario_index}]"
            scenario = _mapping(raw_scenario, scenario_label, errors)
            if scenario is None:
                continue
            scenario_id = scenario.get("id")
            if not _is_string(scenario_id):
                errors.append(f"{scenario_label}.id must be a non-empty string")
            elif scenario_id in scenario_ids:
                errors.append(f"{label} has duplicate scenario id: {scenario_id!r}")
            else:
                scenario_ids.add(str(scenario_id))
            runner = scenario.get("runner")
            if runner not in RUNNERS:
                errors.append(f"{scenario_label}.runner must be one of {sorted(RUNNERS)!r}")
            mode = scenario.get("mode")
            if mode not in MODES:
                errors.append(f"{scenario_label}.mode must be one of {sorted(MODES)!r}")
            repeat = scenario.get("repeat")
            if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat <= 0:
                errors.append(f"{scenario_label}.repeat must be a positive integer")
            resume = scenario.get("resume")
            if not isinstance(resume, bool):
                errors.append(f"{scenario_label}.resume must be a boolean")
            elif mode == "cold" and resume:
                errors.append(f"{scenario_label}.cold must set resume=false")
            elif mode == "warm" and not resume:
                errors.append(f"{scenario_label}.warm must set resume=true")
            independent = scenario.get("independent_validation")
            if not isinstance(independent, bool):
                errors.append(f"{scenario_label}.independent_validation must be a boolean")
            _optional_string_list(
                scenario.get("required_capabilities"),
                f"{scenario_label}.required_capabilities",
                errors,
            )
            if mode == "batch" and runner != "batch_cli":
                errors.append(f"{scenario_label}.batch must use runner='batch_cli'")
            if mode != "batch" and runner != "benchmark_pipeline":
                errors.append(f"{scenario_label}.{mode} must use runner='benchmark_pipeline'")
            if mode == "validation" and not independent:
                errors.append(f"{scenario_label}.validation must set independent_validation=true")

    return errors


def load_and_validate_manifest(
    manifest_path: str | Path, *, repo_root: str | Path | None = None
) -> list[str]:
    path = Path(manifest_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else path.parent.parent.resolve()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"could not read performance manifest {path}: {exc}"]
    if not isinstance(loaded, Mapping):
        return ["performance manifest root must be an object"]
    return validate_manifest(loaded, repo_root=root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Repository root used to resolve corpus_manifest (default: manifest parent).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    errors = load_and_validate_manifest(args.manifest, repo_root=args.repo_root)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Performance manifest verified: {Path(args.manifest).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
