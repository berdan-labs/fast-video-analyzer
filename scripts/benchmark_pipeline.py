"""Repeatable offline benchmark for the public reconstruction pipeline.

The benchmark deliberately uses the same ``run_pipeline`` entry point as the
CLI.  It never downloads models, enables remote media, or permits external AI;
all timing and resource data comes from the generated run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from video_script_reconstructor.pipeline import run_pipeline
from video_script_reconstructor.resource_usage import resource_snapshot
from video_script_reconstructor.security import safe_slug
from video_script_reconstructor.validate_output import validate_project


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one deterministic, offline reconstruction benchmark."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--subtitle", action="append", type=Path, default=[])
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--preset", choices=("strict", "balanced"), default="strict")
    parser.add_argument(
        "--vision-mode", choices=("none", "host-agent", "auto", "local"), default="none"
    )
    parser.add_argument(
        "--asr-chunk-seconds",
        type=int,
        help="Override the bounded ASR checkpoint window for host-specific comparisons.",
    )
    parser.add_argument(
        "--asr-overlap-seconds",
        type=int,
        help="Override ASR overlap for host-specific comparisons (must be below the window).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Force cold iterations. Repeated cold iterations use isolated child output "
            "directories so stage checkpoints cannot be reused."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the benchmark this many times and report min/median/p95 timing (default: 1).",
    )
    parser.add_argument(
        "--independent-validation",
        action="store_true",
        help=(
            "Run a second public validation pass after the pipeline's final proof; "
            "slower but useful for an external audit."
        ),
    )
    parser.add_argument(
        "--asr-chunk-sweep",
        help=(
            "Run isolated cold ASR benchmarks for comma-separated chunk sizes (for example "
            "150,300,600,900) and recommend a size only when transcript coverage agrees."
        ),
    )
    return parser


def _benchmark_once(
    input_path: Path,
    *,
    output_root: Path,
    subtitles: tuple[Path, ...] = (),
    transcript: Path | None = None,
    preset: str = "strict",
    vision_mode: str = "none",
    asr_chunk_seconds: int | None = None,
    asr_overlap_seconds: int | None = None,
    resume: bool = True,
    asr_adapter: Any | None = None,
    independent_validation: bool = False,
) -> dict[str, Any]:
    manifest_before: tuple[int, int] | None = None
    existing_project = output_root / safe_slug(input_path.stem)
    existing_manifest = existing_project / ".state" / "run-manifest.json"
    if existing_manifest.is_file():
        try:
            stat = existing_manifest.stat()
            manifest_before = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            manifest_before = None
    started = time.perf_counter()
    result = run_pipeline(
        input_path,
        output_root=output_root,
        subtitles=subtitles,
        transcript=transcript,
        preset=preset,
        vision_mode=vision_mode,
        asr_chunk_seconds=asr_chunk_seconds,
        asr_overlap_seconds=asr_overlap_seconds,
        resume=resume,
        asr_adapter=asr_adapter,
        offline=True,
        allow_remote_download=False,
        allow_external_ai=False,
    )
    elapsed = time.perf_counter() - started
    manifest_path = result.project_dir / ".state" / "run-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    validation_started = time.perf_counter()
    pipeline_validation = getattr(result, "validation", None)
    if not independent_validation and pipeline_validation is not None:
        validation = pipeline_validation
        validation_source = "pipeline-final"
    else:
        validation = validate_project(result.project_dir)
        validation_source = "independent-public"
    validation_elapsed = time.perf_counter() - validation_started
    quality = _quality_summary(result.project_dir)
    performance = manifest.get("performance", {})
    if not isinstance(performance, dict):
        performance = {}
    observed_output = resource_snapshot(result.project_dir).get("output")
    recorded_output = performance.get("resource_usage", {})
    if isinstance(recorded_output, dict):
        recorded_output = recorded_output.get("output")
    else:
        recorded_output = None
    manifest_after: tuple[int, int] | None = None
    try:
        stat = manifest_path.stat()
        manifest_after = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        pass
    cache_reused = bool(
        resume
        and manifest_before is not None
        and manifest_after is not None
        and manifest_before == manifest_after
    )
    return {
        "input": str(input_path.resolve()),
        "project_dir": str(result.project_dir),
        "status": result.status,
        "exit_code": result.exit_code,
        "asr_chunk_seconds": asr_chunk_seconds,
        "asr_overlap_seconds": asr_overlap_seconds,
        "validation_valid": validation.valid,
        "validation_errors": list(validation.errors),
        "validation_source": validation_source,
        "elapsed_seconds": round(elapsed, 6),
        "validation_elapsed_seconds": round(validation_elapsed, 6),
        "total_elapsed_seconds": round(time.perf_counter() - started, 6),
        "run_cache_key": manifest.get("run_cache_key"),
        "cache_reused": cache_reused,
        "resource_output": observed_output,
        "resource_output_matches_disk": (
            recorded_output == observed_output if recorded_output is not None else None
        ),
        "performance": performance,
        "performance_summary": _performance_summary(
            manifest,
            elapsed_seconds=elapsed,
            cache_reused=cache_reused,
        ),
        "quality": quality,
    }


def _performance_summary(
    manifest: Mapping[str, Any],
    *,
    elapsed_seconds: float,
    cache_reused: bool = False,
) -> dict[str, Any]:
    """Expose comparable critical-path/resource facts beside the raw manifest.

    Stage records can overlap (for example, an opt-in visual survey running
    during ASR), so a caller must not infer wall time from one stage alone. The
    summary keeps both the measured wall clock and the sum of recorded stages,
    plus bounded resource/scheduler fields useful for host comparisons.
    """

    raw_records = manifest.get("stage_records", [])
    stage_elapsed: dict[str, float] = {}
    stage_sum = 0.0
    if isinstance(raw_records, list):
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                continue
            name = raw.get("name")
            elapsed_ms = raw.get("elapsed_ms")
            if not isinstance(name, str) or not isinstance(elapsed_ms, (int, float)):
                continue
            value = max(0.0, float(elapsed_ms) / 1000.0)
            stage_elapsed[name] = round(value, 6)
            stage_sum += value

    performance = manifest.get("performance", {})
    if not isinstance(performance, Mapping):
        performance = {}
    resources = performance.get("resource_usage", {})
    if not isinstance(resources, Mapping):
        resources = {}
    memory = resources.get("memory", {})
    if not isinstance(memory, Mapping):
        memory = {}
    output = resources.get("output", {})
    if not isinstance(output, Mapping):
        output = {}
    scheduling = performance.get("scheduling", {})
    if not isinstance(scheduling, Mapping):
        scheduling = {}
    visual_events = performance.get("visual_events", [])
    survey_elapsed: float | None = None
    if isinstance(visual_events, list):
        for event in visual_events:
            if not isinstance(event, Mapping) or event.get("event") != "survey_parallel_completed":
                continue
            value = event.get("elapsed_seconds")
            if isinstance(value, (int, float)):
                survey_elapsed = round(max(0.0, float(value)), 6)

    return {
        "wall_elapsed_seconds": round(max(0.0, float(elapsed_seconds)), 6),
        "measurement_mode": "warm-cache-hit" if cache_reused else "pipeline-execution",
        "stage_telemetry_source": (
            "previous-run-manifest" if cache_reused else "current-run-manifest"
        ),
        "stage_telemetry_current": not cache_reused,
        "stage_elapsed_seconds": stage_elapsed,
        "stage_sum_seconds": round(stage_sum, 6),
        "stage_sum_minus_wall_seconds": round(stage_sum - max(0.0, float(elapsed_seconds)), 6),
        "peak_rss_bytes": memory.get("peak_rss_bytes"),
        "output_bytes": output.get("bytes"),
        "output_file_count": output.get("file_count"),
        "parallel_visual_survey": scheduling.get("parallel_visual_survey"),
        "survey_parallel_elapsed_seconds": survey_elapsed,
    }


def _quality_summary(project_dir: Path) -> dict[str, Any]:
    """Extract comparable transcript-coverage facts without judging correctness."""

    canonical_path = project_dir / ".state" / "canonical-project.json"
    try:
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "available": False,
            "substantive_segment_count": 0,
            "word_count": 0,
            "ordered_text_sha256": None,
            "blocking_failure_count": None,
        }
    if not isinstance(canonical, dict):
        return {
            "available": False,
            "substantive_segment_count": 0,
            "word_count": 0,
            "ordered_text_sha256": None,
            "blocking_failure_count": None,
        }
    segments = canonical.get("transcript_segments", [])
    if not isinstance(segments, list):
        segments = []
    substantive = [
        item
        for item in segments
        if isinstance(item, dict) and bool(item.get("substantive"))
    ]
    words = [
        word
        for segment in substantive
        for word in (segment.get("words", []) if isinstance(segment.get("words"), list) else [])
        if isinstance(word, dict)
    ]
    ordered_text = " ".join(
        str(segment.get("normalized_text") or segment.get("raw_text") or "")
        for segment in substantive
    ).strip()
    audit = canonical.get("audit", {})
    blocking = audit.get("blocking_failures", []) if isinstance(audit, dict) else []
    return {
        "available": True,
        "substantive_segment_count": len(substantive),
        "word_count": len(words),
        "ordered_text_sha256": hashlib.sha256(ordered_text.encode("utf-8")).hexdigest(),
        "blocking_failure_count": len(blocking) if isinstance(blocking, list) else None,
    }


def summarize_elapsed(values: Sequence[float]) -> dict[str, Any]:
    """Return robust timing statistics without pretending one run is stable."""

    if not values:
        raise ValueError("at least one elapsed timing is required")
    ordered = sorted(float(value) for value in values)
    # Nearest-rank p95 is intentionally conservative for small samples: with
    # fewer than 20 runs it reports the maximum rather than interpolating away
    # a real slow outlier.
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "count": len(ordered),
        "min_seconds": round(ordered[0], 6),
        "median_seconds": round(statistics.median(ordered), 6),
        "p95_seconds": round(ordered[p95_index], 6),
        "max_seconds": round(ordered[-1], 6),
    }


def _cold_iteration_root(output_root: Path, index: int) -> Path:
    """Choose a fresh child directory without deleting prior benchmark evidence."""

    base = output_root / f"cold-{index + 1:03d}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"cold-{index + 1:03d}-{suffix:03d}"
        suffix += 1
    return candidate


def benchmark(
    input_path: Path,
    *,
    output_root: Path,
    subtitles: tuple[Path, ...] = (),
    transcript: Path | None = None,
    preset: str = "strict",
    vision_mode: str = "none",
    asr_chunk_seconds: int | None = None,
    asr_overlap_seconds: int | None = None,
    resume: bool = True,
    repeat: int = 1,
    asr_adapter: Any | None = None,
    independent_validation: bool = False,
) -> dict[str, Any]:
    """Run one or more public-pipeline iterations with explicit cache semantics.

    Warm iterations intentionally share ``output_root`` and exercise the real
    resume path. Cold iterations are isolated below ``output_root`` so survey,
    frame, OCR, and ASR checkpoints cannot accidentally make a cold timing look
    warm. The returned top-level fields retain the historical single-run shape;
    ``iterations`` and ``timing_summary`` provide the richer multi-run report.
    """

    if repeat <= 0:
        raise ValueError("repeat must be positive")
    iterations: list[dict[str, Any]] = []
    for index in range(repeat):
        iteration_root = output_root
        if not resume:
            iteration_root = _cold_iteration_root(output_root, index)
        iterations.append(
            _benchmark_once(
                input_path,
                output_root=iteration_root,
                subtitles=subtitles,
                transcript=transcript,
                preset=preset,
                vision_mode=vision_mode,
                asr_chunk_seconds=asr_chunk_seconds,
                asr_overlap_seconds=asr_overlap_seconds,
                resume=resume,
                asr_adapter=asr_adapter,
                independent_validation=independent_validation,
            )
        )
    latest = dict(iterations[-1])
    latest["iterations"] = iterations
    latest["timing_summary"] = summarize_elapsed(
        [float(item["elapsed_seconds"]) for item in iterations]
    )
    latest["validation_valid"] = all(bool(item["validation_valid"]) for item in iterations)
    return latest


def _resolve_reusable_asr_adapter() -> Any | None:
    """Return one verified faster-whisper adapter for a cold sweep, if present.

    A sweep compares checkpoint geometry, not independent model initialization.
    Reusing the native large-v3 object removes repeated multi-gigabyte model
    loads while keeping each candidate's output/checkpoint tree isolated. Other
    local backends are left to the normal per-run resolver because their worker
    lifecycle and statefulness differ.
    """

    try:
        from video_script_reconstructor.pipeline import _auto_asr_adapters

        adapters = _auto_asr_adapters(language=None, compare_candidates=False)
    except Exception:
        return None
    if len(adapters) != 1:
        for adapter in adapters:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
        return None
    adapter = adapters[0]
    backend = str(getattr(adapter, "backend_name", ""))
    if backend != "faster-whisper":
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
        return None
    return adapter


def _close_reusable_asr_adapter(adapter: Any | None) -> None:
    if adapter is None:
        return
    close = getattr(adapter, "close", None)
    if callable(close):
        close()


def _parse_chunk_sweep(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid ASR chunk size in sweep: {token!r}") from exc
        if value <= 0:
            raise ValueError("ASR chunk sweep sizes must be positive")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("ASR chunk sweep must contain at least one positive size")
    return tuple(values)


def benchmark_asr_chunk_sweep(
    input_path: Path,
    *,
    output_root: Path,
    chunk_seconds: Sequence[int],
    overlap_seconds: int | None = None,
    subtitles: tuple[Path, ...] = (),
    transcript: Path | None = None,
    preset: str = "strict",
    vision_mode: str = "none",
    independent_validation: bool = False,
) -> dict[str, Any]:
    """Compare cold ASR windows and recommend only coverage-equivalent speedups."""

    chunks = tuple(int(value) for value in chunk_seconds)
    if not chunks or any(value <= 0 for value in chunks):
        raise ValueError("ASR chunk sweep sizes must be positive")
    overlap = 15 if overlap_seconds is None else int(overlap_seconds)
    if overlap < 0 or any(overlap >= value for value in chunks):
        raise ValueError("ASR overlap must be non-negative and smaller than every sweep size")
    results: list[dict[str, Any]] = []
    # A supplied transcript/subtitle means no ASR work is needed; avoid even
    # constructing a model adapter for a visual-only benchmark.
    reusable_adapter = (
        _resolve_reusable_asr_adapter()
        if transcript is None and not subtitles
        else None
    )
    try:
        for chunk in chunks:
            report = benchmark(
                input_path,
                output_root=output_root / f"asr-sweep-{chunk:06d}",
                subtitles=subtitles,
                transcript=transcript,
                preset=preset,
                vision_mode=vision_mode,
                asr_chunk_seconds=chunk,
                asr_overlap_seconds=overlap,
                resume=False,
                repeat=1,
                asr_adapter=reusable_adapter,
                independent_validation=independent_validation,
            )
            results.append(
                {
                    "chunk_seconds": chunk,
                    "overlap_seconds": overlap,
                    "elapsed_seconds": report.get("elapsed_seconds"),
                    "timing_summary": report.get("timing_summary"),
                    "status": report.get("status"),
                    "validation_valid": report.get("validation_valid"),
                    "project_dir": report.get("project_dir"),
                    "quality": report.get("quality", {}),
                    "performance_summary": report.get("performance_summary", {}),
                }
            )
    finally:
        _close_reusable_asr_adapter(reusable_adapter)
    valid = [
        item
        for item in results
        if item.get("validation_valid")
        and item.get("status") != "blocked"
        and isinstance(item.get("quality"), dict)
        and item["quality"].get("available")
    ]
    recommendation: dict[str, Any] = {
        "chunk_seconds": None,
        "reason": "No coverage-equivalent valid candidate was available.",
    }
    if valid:
        max_segments = max(int(item["quality"].get("substantive_segment_count", 0)) for item in valid)
        max_words = max(int(item["quality"].get("word_count", 0)) for item in valid)
        coverage_equivalent = [
            item
            for item in valid
            if int(item["quality"].get("substantive_segment_count", 0)) == max_segments
            and int(item["quality"].get("word_count", 0)) == max_words
        ]
        if coverage_equivalent:
            fastest = min(coverage_equivalent, key=lambda item: float(item["elapsed_seconds"]))
            recommendation = {
                "chunk_seconds": fastest["chunk_seconds"],
                "reason": (
                    "Fastest valid candidate among sweep results with the maximum observed "
                    "substantive-segment and word coverage; this remains a host-specific "
                    "recommendation, not an accuracy proof."
                ),
                "max_substantive_segment_count": max_segments,
                "max_word_count": max_words,
            }
    return {
        "schema_version": "1.0",
        "input": str(input_path.resolve()),
        "output_root": str(output_root.resolve()),
        "resume": False,
        "reused_faster_whisper_adapter": reusable_adapter is not None,
        "results": results,
        "recommendation": recommendation,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.asr_chunk_sweep and args.asr_chunk_seconds is not None:
        raise SystemExit("--asr-chunk-sweep cannot be combined with --asr-chunk-seconds")
    if args.asr_chunk_sweep:
        report = benchmark_asr_chunk_sweep(
            args.input,
            output_root=args.output,
            chunk_seconds=_parse_chunk_sweep(args.asr_chunk_sweep),
            overlap_seconds=args.asr_overlap_seconds,
            subtitles=tuple(args.subtitle),
            transcript=args.transcript,
            preset=args.preset,
            vision_mode=args.vision_mode,
            independent_validation=args.independent_validation,
        )
        valid = all(bool(item.get("validation_valid")) for item in report["results"])
    else:
        report = benchmark(
            args.input,
            output_root=args.output,
            subtitles=tuple(args.subtitle),
            transcript=args.transcript,
            preset=args.preset,
            vision_mode=args.vision_mode,
            asr_chunk_seconds=args.asr_chunk_seconds,
            asr_overlap_seconds=args.asr_overlap_seconds,
            resume=not args.no_resume,
            repeat=args.repeat,
            independent_validation=args.independent_validation,
        )
        valid = bool(report["validation_valid"])
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    # ``review_required`` is a valid accuracy-first outcome, not a benchmark
    # failure.  Fail the harness only when the produced artifact is invalid.
    return 0 if valid else 4


if __name__ == "__main__":
    raise SystemExit(main())
