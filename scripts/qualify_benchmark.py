"""Fail-closed evaluator for owner-local hardware qualification receipts.

The policy and report files are deliberately external to the repository. Git
contains only this evaluator and an impossible example policy; no host timing,
licensed media, or quality reference is treated as a public guarantee.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

POLICY_VERSION = 1
REPORT_SCHEMA_VERSION = "1.1"
REPORT_KIND = "pipeline-benchmark"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_POLICY_KEYS = {
    "policy_version",
    "workload_id",
    "report_kind",
    "schema_version",
    "require_cache_reused_false",
    "require_shared_cache_disabled",
    "min_media_duration_s",
    "quality_contract_sha256",
    "required_lane_digests",
    "max_elapsed_seconds",
    "max_p95_seconds",
}
OPTIONAL_POLICY_KEYS = {
    "minimum_report_count",
    "required_runtime_fingerprint_fields",
    "max_peak_rss_bytes",
    "max_output_bytes",
}
REQUIRED_RUNTIME_CACHE_FLAGS = (
    "VSR_DISABLE_ASR_SHARED_CACHE",
    "VSR_DISABLE_VISUAL_SHARED_CACHE",
    "VSR_DISABLE_SEMANTIC_SHARED_CACHE",
)


def _read_json(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"could not read JSON: {exc}"
    if not isinstance(loaded, Mapping):
        return None, "JSON root must be an object"
    return loaded, None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_hex_digest(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def validate_policy(policy: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_POLICY_KEYS - set(policy)
    unknown = set(policy) - REQUIRED_POLICY_KEYS - OPTIONAL_POLICY_KEYS
    errors.extend(f"missing policy key: {key}" for key in sorted(missing))
    errors.extend(f"unknown policy key: {key}" for key in sorted(unknown))
    if errors:
        return errors
    if policy["policy_version"] != POLICY_VERSION:
        errors.append(f"policy_version must be {POLICY_VERSION}")
    if not isinstance(policy["workload_id"], str) or not policy["workload_id"].strip():
        errors.append("workload_id must be a non-empty string")
    if policy["report_kind"] != REPORT_KIND:
        errors.append(f"report_kind must be {REPORT_KIND!r}")
    if policy["schema_version"] != REPORT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {REPORT_SCHEMA_VERSION!r}")
    for key in ("require_cache_reused_false", "require_shared_cache_disabled"):
        if not isinstance(policy[key], bool):
            errors.append(f"{key} must be a boolean")
    for key in (
        "min_media_duration_s",
        "max_elapsed_seconds",
        "max_p95_seconds",
    ):
        if not _is_number(policy[key]) or float(policy[key]) < 0:
            errors.append(f"{key} must be a non-negative number")
    if not _is_hex_digest(policy["quality_contract_sha256"]):
        errors.append("quality_contract_sha256 must be a lowercase 64-character SHA-256 digest")
    lanes = policy["required_lane_digests"]
    if (
        not isinstance(lanes, list)
        or not lanes
        or any(not isinstance(lane, str) or not lane.strip() for lane in lanes)
        or len(set(lanes)) != len(lanes)
    ):
        errors.append("required_lane_digests must be a non-empty list of unique strings")
    if "minimum_report_count" in policy:
        count = policy["minimum_report_count"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            errors.append("minimum_report_count must be a positive integer")
    if "required_runtime_fingerprint_fields" in policy:
        fields = policy["required_runtime_fingerprint_fields"]
        if (
            not isinstance(fields, list)
            or any(not isinstance(field, str) or not field.strip() for field in fields)
            or len(set(fields)) != len(fields)
        ):
            errors.append(
                "required_runtime_fingerprint_fields must be a list of unique non-empty strings"
            )
    for key in ("max_peak_rss_bytes", "max_output_bytes"):
        if key in policy and (not _is_number(policy[key]) or float(policy[key]) < 0):
            errors.append(f"{key} must be a non-negative number")
    return errors


def evaluate_report(report: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    """Return every substantive reason a report fails the owner policy."""

    reasons: list[str] = []

    def require(condition: bool, reason: str) -> None:
        if not condition:
            reasons.append(reason)

    require(report.get("schema_version") == REPORT_SCHEMA_VERSION, "schema_version mismatch")
    require(report.get("report_kind") == REPORT_KIND, "report_kind mismatch")
    require(report.get("workload_id") == policy["workload_id"], "workload_id mismatch")
    if policy["require_cache_reused_false"]:
        require(report.get("cache_reused") is False, "cache_reused must be false")

    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping):
        reasons.append("missing runtime evidence")
    elif policy["require_shared_cache_disabled"]:
        disabled = runtime.get("shared_cache_disabled")
        if not isinstance(disabled, Mapping):
            reasons.append("missing runtime.shared_cache_disabled evidence")
        else:
            for flag in REQUIRED_RUNTIME_CACHE_FLAGS:
                require(disabled.get(flag) is True, f"{flag} must be true")

    require(report.get("validation_valid") is True, "validation_valid must be true")
    quality = report.get("quality")
    if not isinstance(quality, Mapping):
        reasons.append("missing quality evidence")
        return reasons
    require(quality.get("available") is True, "quality.available must be true")

    duration_ms = quality.get("media_duration_ms")
    require(_is_number(duration_ms), "missing quality.media_duration_ms")
    if _is_number(duration_ms):
        minimum_ms = float(policy["min_media_duration_s"]) * 1000
        require(float(duration_ms) >= minimum_ms, "media duration is below policy minimum")

    expected_contract = policy["quality_contract_sha256"]
    require(
        quality.get("quality_contract_sha256") == expected_contract,
        "quality_contract_sha256 mismatch",
    )
    lane_digests = quality.get("lane_sha256")
    if not isinstance(lane_digests, Mapping):
        reasons.append("missing quality.lane_sha256 evidence")
    else:
        for lane in policy["required_lane_digests"]:
            value = lane_digests.get(lane)
            require(_is_hex_digest(value), f"missing or malformed lane digest: {lane}")

    elapsed = report.get("elapsed_seconds")
    require(_is_number(elapsed), "missing elapsed_seconds")
    if _is_number(elapsed):
        require(
            float(elapsed) <= float(policy["max_elapsed_seconds"]),
            "elapsed_seconds exceeds policy target",
        )
    timing = report.get("timing_summary")
    p95 = timing.get("p95_seconds") if isinstance(timing, Mapping) else None
    require(_is_number(p95), "missing timing_summary.p95_seconds")
    if _is_number(p95):
        require(
            float(p95) <= float(policy["max_p95_seconds"]),
            "timing_summary.p95_seconds exceeds policy target",
        )
    performance = report.get("performance_summary")
    if not isinstance(performance, Mapping):
        performance = {}
    if "max_peak_rss_bytes" in policy:
        peak_rss = performance.get("peak_rss_bytes")
        require(_is_number(peak_rss), "missing performance_summary.peak_rss_bytes")
        if _is_number(peak_rss):
            require(
                float(peak_rss) <= float(policy["max_peak_rss_bytes"]),
                "performance_summary.peak_rss_bytes exceeds policy budget",
            )
    if "max_output_bytes" in policy:
        output_bytes = performance.get("output_bytes")
        require(_is_number(output_bytes), "missing performance_summary.output_bytes")
        if _is_number(output_bytes):
            require(
                float(output_bytes) <= float(policy["max_output_bytes"]),
                "performance_summary.output_bytes exceeds policy budget",
            )
    return reasons


def _fingerprint_reasons(
    parsed_reports: Sequence[tuple[str, Mapping[str, Any]]],
    fields: Sequence[str],
) -> list[str]:
    """Return reasons for runtime fingerprint fields that are missing or differ."""

    if not fields or not parsed_reports:
        return []
    reasons: list[str] = []
    for field in fields:
        values: list[Any] = []
        missing: list[str] = []
        for name, report in parsed_reports:
            runtime = report.get("runtime")
            if not isinstance(runtime, Mapping) or field not in runtime:
                missing.append(name)
                continue
            values.append(runtime[field])
        if missing:
            reasons.append(f"runtime.{field} missing from reports: {', '.join(missing)}")
            continue
        first = values[0]
        if any(value != first for value in values[1:]):
            reasons.append(f"runtime.{field} is not identical across reports")
    return reasons


def evaluate_files(policy_path: Path, report_paths: Sequence[Path]) -> tuple[int, dict[str, Any]]:
    policy, policy_error = _read_json(policy_path)
    if policy_error is not None or policy is None:
        return 2, {"verdict": "invalid-input", "errors": [policy_error or "invalid policy"]}
    policy_errors = validate_policy(policy)
    if policy_errors:
        return 3, {"verdict": "invalid-policy", "errors": policy_errors}

    minimum_count = policy.get("minimum_report_count", 1)
    fingerprint_fields = policy.get("required_runtime_fingerprint_fields", [])

    parsed: list[tuple[str, Mapping[str, Any]]] = []
    reports: list[dict[str, Any]] = []
    malformed = False
    rejected = False
    for path in report_paths:
        report, report_error = _read_json(path)
        if report_error is not None or report is None:
            malformed = True
            reports.append({"report": str(path), "reasons": [report_error or "invalid report"]})
            continue
        reasons = evaluate_report(report, policy)
        rejected = rejected or bool(reasons)
        reports.append({"report": str(path), "reasons": reasons})
        parsed.append((str(path), report))

    combined_reasons: list[str] = []
    if len(report_paths) < minimum_count:
        combined_reasons.append(
            f"policy requires at least {minimum_count} report file(s), got {len(report_paths)}"
        )
    combined_reasons.extend(_fingerprint_reasons(parsed, fingerprint_fields))
    if combined_reasons:
        rejected = True

    if malformed:
        code = 4
        verdict = "invalid-input"
    elif rejected or not reports:
        code = 5
        verdict = "rejected"
    else:
        code = 0
        verdict = "qualified"
    result: dict[str, Any] = {"verdict": verdict, "reports": reports}
    if combined_reasons:
        result["reasons"] = combined_reasons
    return code, result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate owner-local hardware qualification receipts.")
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--report", action="append", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    code, result = evaluate_files(args.policy, tuple(args.report))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
