"""Run and score the manifest-driven evaluation corpus.

The evaluator keeps quality assertions separate from host-dependent timing. It
can run deterministic generated cases in CI and can use the same manifest with
external inputs and explicit model revisions on an owner-controlled machine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCORING_VERSION = "1.0"
DEFAULT_MODEL_REVISION = "deterministic-fixtures"
ROOT = Path(__file__).resolve().parents[1]


def _load_sibling(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> Mapping[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return loaded


def canonical_hash(value: Mapping[str, Any]) -> str:
    """Hash parsed JSON rather than platform-dependent source newlines."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: str | Path) -> Mapping[str, Any]:
    manifest_path = Path(path).resolve()
    validator = _load_sibling("vsr_corpus_manifest_validator", "validate_corpus_manifest.py")
    errors = validator.load_and_validate_manifest(manifest_path)
    if errors:
        raise ValueError("invalid corpus manifest: " + "; ".join(errors))
    return _load_json(manifest_path)


def _case_text_score(case: Mapping[str, Any], project_dir: Path) -> dict[str, Any]:
    fixture_validator = _load_sibling("vsr_fixture_output_validator", "validate_fixture_output.py")
    expected = case.get("expected", {})
    if not isinstance(expected, Mapping):
        expected = {}
    spoken = expected.get("spoken", [])
    tokens = expected.get("tokens", [])
    if not isinstance(spoken, list):
        spoken = []
    if not isinstance(tokens, list):
        tokens = []
    result = fixture_validator.validate_fixture_output(
        project_dir,
        expected_spoken=tuple(str(item) for item in spoken),
        expected_tokens=tuple(str(item) for item in tokens),
    )
    checks = result.get("checks", {})
    if not isinstance(checks, Mapping):
        checks = {}
    visual_event_count = int(checks.get("visual_event_count", 0) or 0)
    minimum_visual_events = int(expected.get("minimum_visual_events", 0) or 0)
    errors = [str(error) for error in result.get("errors", [])]
    if visual_event_count < minimum_visual_events:
        errors.append(
            "fixture_content: expected at least "
            f"{minimum_visual_events} visual events, found {visual_event_count}"
        )
    return {
        "pass": not errors,
        "production_valid": bool(result.get("production_valid")),
        "expected_spoken_count": len(spoken),
        "spoken_matches": int(checks.get("expected_spoken_matches", 0) or 0),
        "expected_token_count": len(tokens),
        "token_matches": int(checks.get("expected_token_matches", 0) or 0),
        "minimum_visual_events": minimum_visual_events,
        "visual_event_count": visual_event_count,
        "errors": errors,
    }


def score_project(
    case: Mapping[str, Any], project_dir: Path, benchmark_report: Mapping[str, Any]
) -> dict[str, Any]:
    quality = _case_text_score(case, project_dir)
    return {
        "case_id": case.get("id"),
        "status": benchmark_report.get("status"),
        "validation_valid": bool(benchmark_report.get("validation_valid")),
        "quality": quality,
        "elapsed_seconds": benchmark_report.get("elapsed_seconds"),
        "timing_summary": benchmark_report.get("timing_summary", {}),
        "performance_summary": benchmark_report.get("performance_summary", {}),
        "project_dir": str(project_dir.resolve()),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _current_elapsed(report: Mapping[str, Any]) -> float | None:
    timing = report.get("timing_summary")
    if isinstance(timing, Mapping):
        median = _number(timing.get("median_seconds"))
        if median is not None:
            return median
    return _number(report.get("elapsed_seconds"))


def _performance_comparison(
    case_id: str,
    report: Mapping[str, Any] | None,
    baseline_seconds: Any,
    threshold: float,
) -> dict[str, Any]:
    if report is None:
        return {
            "case_id": case_id,
            "status": "missing",
            "pass": False,
            "reason": "no evaluator result was produced",
        }
    baseline = _number(baseline_seconds)
    current = _current_elapsed(report)
    if baseline is None or baseline <= 0:
        return {
            "case_id": case_id,
            "status": "not_compared",
            "pass": None,
            "current_seconds": current,
            "baseline_seconds": baseline,
            "reason": "no host-compatible baseline timing is recorded",
        }
    if current is None:
        return {
            "case_id": case_id,
            "status": "unavailable",
            "pass": False,
            "baseline_seconds": baseline,
            "reason": "evaluator result has no elapsed timing",
        }
    regression = (current / baseline) - 1.0
    passed = regression <= threshold
    return {
        "case_id": case_id,
        "status": "compared",
        "pass": passed,
        "current_seconds": round(current, 6),
        "baseline_seconds": round(baseline, 6),
        "regression_fraction": round(regression, 6),
        "max_regression_fraction": threshold,
    }


def evaluate_reports(
    manifest: Mapping[str, Any],
    baseline: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    *,
    model_revision: str = DEFAULT_MODEL_REVISION,
    require_performance: bool = False,
) -> dict[str, Any]:
    """Compare scored case reports with a compatible quality/performance baseline."""

    manifest_case_ids = {
        str(case.get("id"))
        for case in manifest.get("cases", [])
        if isinstance(case, Mapping) and case.get("id")
    }
    baseline_cases = baseline.get("cases", {})
    if not isinstance(baseline_cases, Mapping) or not baseline_cases:
        raise ValueError("baseline.cases must be a non-empty object")
    identity_errors: list[str] = []
    if baseline.get("schema_version") != manifest.get("schema_version"):
        identity_errors.append("baseline schema_version does not match the corpus manifest")
    if baseline.get("scoring_version") != SCORING_VERSION:
        identity_errors.append(f"baseline scoring_version must be {SCORING_VERSION!r}")
    actual_corpus_hash = canonical_hash(manifest)
    if baseline.get("corpus_hash") != actual_corpus_hash:
        identity_errors.append("baseline corpus_hash does not match the manifest")
    if baseline.get("model_revision") != model_revision:
        identity_errors.append("baseline model_revision does not match this evaluation")
    unknown_baseline_cases = set(str(key) for key in baseline_cases) - manifest_case_ids
    if unknown_baseline_cases:
        identity_errors.append(
            f"baseline references unknown cases: {sorted(unknown_baseline_cases)!r}"
        )

    quality_cases: list[dict[str, Any]] = []
    for case_id, target_value in baseline_cases.items():
        case_key = str(case_id)
        target = target_value if isinstance(target_value, Mapping) else {}
        target_quality = target.get("quality", {})
        target_quality = target_quality if isinstance(target_quality, Mapping) else {}
        current = reports.get(case_key)
        current_quality = current.get("quality", {}) if isinstance(current, Mapping) else {}
        current_quality = current_quality if isinstance(current_quality, Mapping) else {}
        errors: list[str] = []
        if current is None:
            errors.append("no evaluator result was produced")
        else:
            if not bool(current_quality.get("production_valid")):
                errors.append("production output validation failed")
            for metric in (
                "expected_spoken_count",
                "expected_token_count",
                "minimum_visual_events",
            ):
                expected = _number(target_quality.get(metric))
                observed = _number(current_quality.get(metric))
                if expected is not None and observed != expected:
                    errors.append(f"{metric}: expected baseline {expected:g}, observed {observed}")
            comparisons = (
                ("spoken_matches", "expected_spoken_count"),
                ("token_matches", "expected_token_count"),
            )
            for observed_metric, expected_metric in comparisons:
                expected = _number(target_quality.get(expected_metric))
                observed = _number(current_quality.get(observed_metric))
                if expected is not None and (observed is None or observed < expected):
                    errors.append(
                        f"{observed_metric}: expected at least {expected:g}, observed {observed}"
                    )
            observed_visual = _number(current_quality.get("visual_event_count"))
            minimum_visual = _number(target_quality.get("minimum_visual_events"))
            if minimum_visual is not None and (
                observed_visual is None or observed_visual < minimum_visual
            ):
                errors.append(
                    f"visual_event_count: expected at least {minimum_visual:g}, observed {observed_visual}"
                )
            errors.extend(str(error) for error in current_quality.get("errors", []))
        quality_cases.append(
            {
                "case_id": case_key,
                "pass": not errors,
                "errors": sorted(set(errors)),
                "observed": dict(current_quality),
            }
        )

    performance_config = baseline.get("performance", {})
    performance_config = performance_config if isinstance(performance_config, Mapping) else {}
    threshold = _number(performance_config.get("max_regression_fraction"))
    if threshold is None or threshold < 0:
        raise ValueError("baseline.performance.max_regression_fraction must be non-negative")
    baseline_timings = performance_config.get("baseline_elapsed_seconds", {})
    baseline_timings = baseline_timings if isinstance(baseline_timings, Mapping) else {}
    performance_cases = [
        _performance_comparison(
            str(case_id),
            reports.get(str(case_id)),
            baseline_timings.get(str(case_id)),
            threshold,
        )
        for case_id in baseline_cases
    ]
    quality_pass = not identity_errors and all(bool(item["pass"]) for item in quality_cases)
    compared = [item for item in performance_cases if item.get("status") == "compared"]
    performance_failures = [item for item in performance_cases if item.get("pass") is False]
    performance_pass: bool | None
    if performance_failures:
        performance_pass = False
    elif compared:
        performance_pass = True
    else:
        performance_pass = None
    gate_pass = quality_pass and (
        performance_pass is True if require_performance else performance_pass is not False
    )
    return {
        "schema_version": "1.0",
        "scoring_version": SCORING_VERSION,
        "corpus_hash": actual_corpus_hash,
        "model_revision": model_revision,
        "identity": {"compatible": not identity_errors, "errors": identity_errors},
        "quality": {
            "pass": quality_pass,
            "cases": quality_cases,
        },
        "performance": {
            "pass": performance_pass,
            "required": require_performance,
            "max_regression_fraction": threshold,
            "cases": performance_cases,
        },
        "gate_pass": gate_pass,
    }


def run_corpus(
    manifest_path: str | Path,
    baseline_path: str | Path,
    output_root: str | Path,
    *,
    preset: str = "strict",
    vision_mode: str = "none",
    repeat: int = 1,
    model_revision: str = DEFAULT_MODEL_REVISION,
    require_performance: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    baseline = _load_json(Path(baseline_path).resolve())
    benchmark_module = _load_sibling("vsr_benchmark_pipeline", "benchmark_pipeline.py")
    repo_root = manifest_path.parent.parent
    cases_by_id = {
        str(case.get("id")): case
        for case in manifest.get("cases", [])
        if isinstance(case, Mapping) and case.get("id")
    }
    reports: dict[str, Mapping[str, Any]] = {}
    output_root = Path(output_root).resolve()
    for case_id in baseline.get("cases", {}):
        case = cases_by_id.get(str(case_id))
        if case is None:
            continue
        media = case.get("media", {})
        media_path = (
            (repo_root / str(media.get("path"))).resolve()
            if isinstance(media, Mapping) and media.get("path")
            else None
        )
        if media_path is None or not media_path.is_file():
            reports[str(case_id)] = {
                "quality": {
                    "pass": False,
                    "production_valid": False,
                    "errors": ["media is unavailable on this evaluator host"],
                }
            }
            continue
        subtitle_paths = tuple(
            (repo_root / str(item["path"])).resolve()
            for item in case.get("subtitles", [])
            if isinstance(item, Mapping) and item.get("path")
        )
        raw = benchmark_module.benchmark(
            media_path,
            output_root=output_root / str(case_id),
            subtitles=subtitle_paths,
            preset=preset,
            vision_mode=vision_mode,
            resume=repeat > 1,
            repeat=repeat,
        )
        reports[str(case_id)] = score_project(case, Path(raw["project_dir"]), raw)
    result = evaluate_reports(
        manifest,
        baseline,
        reports,
        model_revision=model_revision,
        require_performance=require_performance,
    )
    result["cases"] = reports
    result["manifest_path"] = str(manifest_path)
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    quality = report.get("quality", {})
    performance = report.get("performance", {})
    lines = [
        "# Corpus evaluation",
        "",
        f"- Gate: **{'PASS' if report.get('gate_pass') else 'FAIL'}**",
        f"- Scoring version: `{report.get('scoring_version')}`",
        f"- Model revision: `{report.get('model_revision')}`",
        f"- Corpus hash: `{report.get('corpus_hash')}`",
        f"- Quality gate: `{'pass' if quality.get('pass') else 'fail'}`",
        f"- Performance comparison: `{performance.get('pass')}` (required: `{performance.get('required')}`)",
        "",
        "| Case | Quality | Spoken | Tokens | Visual events | Performance |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    quality_cases = {
        str(item.get("case_id")): item
        for item in quality.get("cases", [])
        if isinstance(item, Mapping)
    }
    performance_cases = {
        str(item.get("case_id")): item
        for item in performance.get("cases", [])
        if isinstance(item, Mapping)
    }
    for case_id in sorted(set(quality_cases) | set(performance_cases)):
        q = quality_cases.get(case_id, {})
        observed = q.get("observed", {}) if isinstance(q, Mapping) else {}
        p = performance_cases.get(case_id, {})
        lines.append(
            "| `{}` | {} | {}/{} | {}/{} | {}/{} | {} |".format(
                case_id,
                "pass" if q.get("pass") else "fail",
                observed.get("spoken_matches", 0),
                observed.get("expected_spoken_count", 0),
                observed.get("token_matches", 0),
                observed.get("expected_token_count", 0),
                observed.get("visual_event_count", 0),
                observed.get("minimum_visual_events", 0),
                p.get("status", "not-compared"),
            )
        )
    identity = report.get("identity", {})
    if isinstance(identity, Mapping) and identity.get("errors"):
        lines.extend(["", "## Identity errors", ""])
        lines.extend(f"- {error}" for error in identity["errors"])
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-markdown", type=Path)
    parser.add_argument("--preset", choices=("strict", "balanced"), default="strict")
    parser.add_argument(
        "--vision-mode", choices=("none", "host-agent", "auto", "local"), default="none"
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--require-performance-baseline", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_corpus(
            args.manifest,
            args.baseline,
            args.output_root,
            preset=args.preset,
            vision_mode=args.vision_mode,
            repeat=args.repeat,
            model_revision=args.model_revision,
            require_performance=args.require_performance_baseline,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if args.report_markdown:
        args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.report_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0 if report.get("gate_pass") else 4


if __name__ == "__main__":
    raise SystemExit(main())
