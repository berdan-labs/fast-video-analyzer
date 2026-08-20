from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .errors import BlockedError, InputError, ReviewRequired, ValidationFailure


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def _sample_values(values: Any, *, limit: int = 6) -> list[str]:
    """Return a bounded, deterministic sample for potentially huge ID lists."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    normalized = [str(value) for value in values]
    if len(normalized) <= limit:
        return normalized
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return [*normalized[:head], "...", *normalized[-tail:]]


def _compact_semantic_batch_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Bound batch CLI output without discarding the underlying project state.

    A long semantic continuation can defer thousands of packet IDs and return
    one ingestion record per applied packet.  Printing those records is not
    useful for routine monitoring and can consume more memory than the work
    itself.  The API still returns the complete structure; this presentation
    helper only replaces large per-project lists with counts and samples.
    """

    compact = dict(result)
    projects = result.get("projects")
    if not isinstance(projects, Sequence) or isinstance(projects, (str, bytes, bytearray)):
        return compact
    compact_projects: list[Any] = []
    for item in projects:
        if not isinstance(item, Mapping):
            compact_projects.append(item)
            continue
        project_item = dict(item)
        summary = project_item.get("summary")
        if isinstance(summary, Mapping):
            summary_compact = {
                key: value
                for key, value in summary.items()
                if key
                not in {
                    "applied",
                    "skipped_event_ids",
                    "semantic_deferred_event_ids",
                    "semantic_provider_failures",
                }
            }
            applied = summary.get("applied", [])
            skipped = summary.get("skipped_event_ids", [])
            deferred = summary.get("semantic_deferred_event_ids", [])
            failures = summary.get("semantic_provider_failures", [])
            applied_ids = [
                str(entry.get("observation_id"))
                for entry in applied
                if isinstance(entry, Mapping) and entry.get("observation_id")
            ] if isinstance(applied, Sequence) else []
            failure_sample = [
                {
                    "candidate_id": entry.get("candidate_id"),
                    "error": str(entry.get("error", ""))[:240],
                }
                for entry in failures[:3]
                if isinstance(entry, Mapping)
            ] if isinstance(failures, Sequence) else []
            summary_compact.update(
                {
                    "applied_count": len(applied) if isinstance(applied, Sequence) else 0,
                    "applied_observation_id_sample": _sample_values(applied_ids),
                    "skipped_event_count": len(skipped) if isinstance(skipped, Sequence) else 0,
                    "skipped_event_id_sample": _sample_values(skipped),
                    "semantic_deferred_event_count": (
                        len(deferred) if isinstance(deferred, Sequence) else 0
                    ),
                    "semantic_deferred_event_id_sample": _sample_values(deferred),
                    "semantic_provider_failure_count": (
                        len(failures) if isinstance(failures, Sequence) else 0
                    ),
                    "semantic_provider_failure_sample": failure_sample,
                }
            )
            project_item["summary"] = summary_compact
        compact_projects.append(project_item)
    compact["projects"] = compact_projects
    compact["output_mode"] = "compact"
    return compact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fast-video-analyzer",
        description=(
            "Fast Video Analyzer: turn video and optional subtitles into structured Markdown and JSON reports "
            "with transcript context, OCR, visual evidence, and provenance metadata."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser(
        "doctor", help="Report local prerequisites using pipeline resolution logic."
    )
    doctor.add_argument(
        "--output", type=Path, help="Output location whose disk/write access should be checked."
    )
    doctor.add_argument("--offline", action="store_true", help="Report the strict offline policy.")

    diagnostic_bundle = commands.add_parser(
        "diagnostic-bundle",
        aliases=("diagnostics", "support-bundle"),
        help="Write a sanitized support bundle without copying user media or credentials.",
    )
    diagnostic_bundle.add_argument("--output", required=True, type=Path, help="Output .zip path.")
    diagnostic_bundle.add_argument(
        "--force", action="store_true", help="Explicitly replace an existing output archive."
    )

    plan = commands.add_parser(
        "plan", help="Inspect lightly and print a no-download, no-full-processing plan."
    )
    plan.add_argument("input")
    plan.add_argument("--output", type=Path)
    plan.add_argument("--subtitle", action="append", type=Path, default=[])
    plan.add_argument("--transcript", type=Path)
    plan.add_argument("--preset", choices=("strict", "balanced"), default="strict")
    plan.add_argument("--config", type=Path)
    plan.add_argument(
        "--vision-mode",
        choices=("auto", "host-agent", "local", "external", "none"),
        default="host-agent",
        help="Use offline Codex/subagent review bundles by default; auto is a compatibility alias.",
    )
    plan.add_argument("--offline", action="store_true")

    run = commands.add_parser("run", help="Run the long-video analysis pipeline.")
    run.add_argument("input")
    run.add_argument(
        "--output",
        type=Path,
        help=(
            "Output directory. If omitted, create '<video stem> (Analyzer Outputs)' "
            "beside the source video."
        ),
    )
    run.add_argument("--subtitle", action="append", type=Path, default=[])
    run.add_argument("--transcript", type=Path)
    run.add_argument("--preset", choices=("strict", "balanced"), default="strict")
    run.add_argument("--config", type=Path)
    run.add_argument(
        "--subtitle-mode",
        choices=("auto", "provided-only", "force-asr", "compare-all"),
        default="auto",
    )
    run.add_argument("--language")
    run.add_argument(
        "--fidelity-mode",
        choices=("verbatim", "clean-verbatim", "production-script"),
        default="verbatim",
    )
    run.add_argument(
        "--vision-mode",
        choices=("auto", "host-agent", "local", "external", "none"),
        default="host-agent",
        help="Use offline Codex/subagent review bundles by default; local Qwen is explicit legacy opt-in.",
    )
    run.add_argument(
        "--asr-chunk-seconds",
        type=int,
        help="Override the bounded ASR checkpoint window (accuracy is unchanged; smaller windows resume sooner).",
    )
    run.add_argument(
        "--asr-overlap-seconds",
        type=int,
        help="Override ASR overlap between checkpoint windows (must be smaller than the window).",
    )
    run.add_argument(
        "--semantic-max-packets",
        type=int,
        help=(
            "Bound Codex/subagent visual-review packets in the one-prompt handoff; "
            "larger values reduce resume passes without copying pixels."
        ),
    )
    resume = run.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    run.add_argument("--offline", action="store_true")
    run.add_argument("--allow-remote-download", action="store_true")
    run.add_argument("--allow-external-ai", action="store_true")
    progress = run.add_mutually_exclusive_group()
    progress.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Print bounded ASR progress/ETA to stderr (the JSON result remains on stdout).",
    )
    progress.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Suppress live ASR progress output.",
    )

    batch = commands.add_parser(
        "batch", help="Run a storage-aware sequential batch over a source-video directory."
    )
    batch.add_argument("source_root", type=Path)
    batch.add_argument(
        "--output",
        type=Path,
        help=(
            "Output root for batch projects. Without this option, each report is placed beside "
            "its source video; VSR_OUTPUT_ROOT is an explicit shared-root override."
        ),
    )
    batch.add_argument("--preset", choices=("strict", "balanced"), default="strict")
    batch.add_argument(
        "--vision-mode",
        choices=("auto", "host-agent", "local", "external", "none"),
        default="host-agent",
        help="Use bounded offline Codex/subagent review bundles by default; local Qwen is legacy opt-in.",
    )
    batch.add_argument("--min-free-bytes", type=int, default=10 * 1024**3)
    batch.add_argument("--max-project-bytes", type=int, default=8 * 1024**3)
    batch.add_argument(
        "--semantic-max-packets",
        type=int,
        help="Bound expensive semantic packets per video while retaining deterministic evidence.",
    )
    batch.add_argument(
        "--language",
        help=(
            "Optional Whisper language hint applied to every source (for example, 'fil' "
            "for Filipino); omitted keeps independent per-chunk detection."
        ),
    )
    batch.add_argument(
        "--compare-sidecars",
        action="store_true",
        help=(
            "Run Whisper alongside adjacent SRT/VTT/ASS sidecars and preserve "
            "candidate disagreements for review."
        ),
    )
    batch.add_argument("--history-root", action="append", type=Path, default=[])
    batch.add_argument("--continue-on-blocked", action="store_true")
    batch.add_argument("--dry-run", action="store_true")
    batch_progress = batch.add_mutually_exclusive_group()
    batch_progress.add_argument("--progress", dest="progress", action="store_true", default=True)
    batch_progress.add_argument("--no-progress", dest="progress", action="store_false")
    batch_resume = batch.add_mutually_exclusive_group()
    batch_resume.add_argument("--resume", dest="resume", action="store_true", default=True)
    batch_resume.add_argument("--no-resume", dest="resume", action="store_false")

    validate = commands.add_parser(
        "validate", help="Validate the exact output, links, state, and embedded metadata."
    )
    validate.add_argument("project_dir", type=Path)

    semantic = commands.add_parser(
        "semantic",
        help="Legacy local semantic continuation; use review bundle for Codex/subagent visual review.",
    )
    semantic.add_argument("project_dir", type=Path)
    semantic.add_argument(
        "--max-packets",
        type=int,
        help="Bound pending local semantic packets for this continuation pass.",
    )
    semantic.add_argument(
        "--vision-mode",
        choices=("local",),
        default="local",
        help="Use the legacy verified offline local Qwen3-VL observer (explicit opt-in only).",
    )
    semantic.add_argument(
        "--retry-fallbacks",
        action="store_true",
        help="Retry only prior semantic fallbacks caused by a local HTTP 400 transport rejection.",
    )
    semantic.add_argument(
        "--retry-semantic-pending",
        action="store_true",
        help=(
            "Retry semantic_pending observations only when their stored prompt revision "
            "differs from the current local adapter."
        ),
    )
    semantic.add_argument(
        "--workers",
        type=int,
        default=1,
        choices=(1, 2),
        help="Bound concurrent local VLM requests (default: 1; use 2 only after a memory benchmark).",
    )

    semantic_batch = commands.add_parser(
        "semantic-batch",
        help="Legacy local semantic batch; use review bundle for Codex/subagent visual review.",
    )
    semantic_batch.add_argument("output_root", type=Path)
    semantic_batch.add_argument(
        "--max-packets-per-project",
        type=int,
        default=32,
        help="Bound semantic packets for each project (default: 32).",
    )
    semantic_batch.add_argument(
        "--min-free-bytes",
        type=int,
        default=10 * 1024**3,
        help="Stop before starting a project below this free-space reserve.",
    )
    semantic_batch.add_argument(
        "--retry-fallbacks",
        action="store_true",
        help="Retry only prior semantic fallbacks caused by local HTTP 400 transport rejection.",
    )
    semantic_batch.add_argument(
        "--retry-semantic-pending",
        action="store_true",
        help=(
            "Retry semantic_pending observations only when their stored prompt revision "
            "differs from the current local adapter."
        ),
    )
    semantic_batch.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later projects after one semantic pass error.",
    )
    semantic_batch.add_argument(
        "--workers",
        type=int,
        default=1,
        choices=(1, 2),
        help="Bound concurrent local VLM requests (default: 1; use 2 only after a memory benchmark).",
    )
    semantic_batch.add_argument(
        "--full-output",
        action="store_true",
        help="Emit complete per-packet semantic records instead of bounded counts/samples.",
    )

    cache = commands.add_parser("cache", help="Manage only validated project-local caches.")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    cache_purge = cache_commands.add_parser(
        "purge", help="Purge one analyzer project's contained stage cache."
    )
    cache_purge.add_argument("project_dir", type=Path)
    cache_compact = cache_commands.add_parser(
        "compact",
        help="Show or explicitly remove completed visual/OCR checkpoints while preserving ASR state.",
    )
    cache_compact.add_argument("project_dir", type=Path)
    cache_compact.add_argument(
        "--apply",
        action="store_true",
        help="Apply the validated compaction plan; default is a read-only estimate.",
    )

    retention = commands.add_parser(
        "retention", help="Report or safely prune old generated analyzer runs."
    )
    retention_commands = retention.add_subparsers(dest="retention_command", required=True)
    retention_report_command = retention_commands.add_parser(
        "report", help="Report bytes and files for recognized generated projects."
    )
    retention_report_command.add_argument("root", type=Path)
    retention_orphans_command = retention_commands.add_parser(
        "orphans",
        help="Report generated-looking incomplete trees without deleting anything.",
    )
    retention_orphans_command.add_argument("root", type=Path)
    retention_prune = retention_commands.add_parser(
        "prune", help="Plan old-run deletion; add --apply to perform the deletion."
    )
    retention_prune.add_argument("root", type=Path)
    retention_prune.add_argument("--keep", type=int, default=1)
    retention_prune.add_argument(
        "--apply", action="store_true", help="Actually delete planned old project directories."
    )
    retention_prune_orphans = retention_commands.add_parser(
        "prune-orphans",
        help="Plan incomplete-footprint deletion; add --apply only after reviewing the dry run.",
    )
    retention_prune_orphans.add_argument("root", type=Path)
    retention_prune_orphans.add_argument(
        "--apply", action="store_true", help="Actually delete planned orphan directories."
    )

    models = commands.add_parser(
        "models", help="Explicitly manage optional local model weights; normal runs never download."
    )
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_report = model_commands.add_parser(
        "report", help="Report model-store bytes and explicit removal commands without deleting."
    )
    model_report.add_argument("--root", type=Path)
    model_report.add_argument(
        "--full",
        action="store_true",
        help="Re-hash every recorded model file before reporting storage.",
    )
    model_report.add_argument(
        "--with-workers",
        action="store_true",
        help="Probe isolated worker runtimes to distinguish verified weights from runnable packs.",
    )
    model_list = model_commands.add_parser("list", help="List local model capability status.")
    model_list.add_argument("--root", type=Path)
    model_fetch = model_commands.add_parser(
        "fetch", help="Explicitly download and hash one public optional model."
    )
    model_fetch.add_argument("name")
    model_fetch.add_argument("--root", type=Path)
    model_fetch.add_argument("--revision")
    model_verify = model_commands.add_parser(
        "verify", help="Verify local model files against their pinned manifest."
    )
    model_verify.add_argument("name", nargs="?")
    model_verify.add_argument("--root", type=Path)
    model_verify.add_argument(
        "--full",
        action="store_true",
        help="Re-hash every recorded model file instead of using an unchanged stat-bound receipt.",
    )
    model_remove = model_commands.add_parser(
        "remove", help="Remove exactly one manifest-verified optional model directory."
    )
    model_remove.add_argument("name")
    model_remove.add_argument("--root", type=Path)

    workers = commands.add_parser(
        "workers", help="Explicitly install and verify isolated heavyweight local runtimes."
    )
    worker_commands = workers.add_subparsers(dest="workers_command", required=True)
    worker_list = worker_commands.add_parser("list")
    worker_list.add_argument("--root", type=Path)
    worker_install = worker_commands.add_parser("install")
    worker_install.add_argument("name", choices=("qwen-speech", "moss-speech", "paddle-ocr"))
    worker_install.add_argument("--root", type=Path)
    worker_verify = worker_commands.add_parser("verify")
    worker_verify.add_argument("name", nargs="?")
    worker_verify.add_argument("--root", type=Path)

    evidence = commands.add_parser("evidence", help="Inspect and enrich evidence without a UI.")
    evidence_commands = evidence.add_subparsers(dest="evidence_command", required=True)
    metadata = evidence_commands.add_parser("metadata")
    metadata_commands = metadata.add_subparsers(dest="metadata_command", required=True)
    metadata_show = metadata_commands.add_parser(
        "show", help="Read metadata from the image bytes and compare canonical state."
    )
    metadata_show.add_argument("project_dir", type=Path)
    metadata_show.add_argument("image_id")
    metadata_show.add_argument("--json", dest="json_output", action="store_true")
    metadata_verify = metadata_commands.add_parser(
        "verify", help="Fail on stripped, corrupt, stale, or mismatched metadata."
    )
    metadata_verify.add_argument("project_dir", type=Path)
    metadata_verify.add_argument("image_id", nargs="?")

    packet = evidence_commands.add_parser("packet")
    packet_commands = packet.add_subparsers(dest="packet_command", required=True)
    packet_show = packet_commands.add_parser("show", help="Show a host-agent evidence packet.")
    packet_show.add_argument("project_dir", type=Path)
    packet_show.add_argument("event_or_frame_id")
    packet_show.add_argument("--json", dest="json_output", action="store_true")

    observation = evidence_commands.add_parser("observation")
    observation_commands = observation.add_subparsers(dest="observation_command", required=True)
    observation_ingest = observation_commands.add_parser(
        "ingest", help="Append and reconcile a schema-valid visual observation."
    )
    observation_ingest.add_argument("project_dir", type=Path)
    observation_ingest.add_argument("--input", required=True, type=Path)
    observation_ingest.add_argument("--base-revision", required=True)

    ocr = evidence_commands.add_parser(
        "ocr", help="Repair OCR projections from existing evidence images only."
    )
    ocr_commands = ocr.add_subparsers(dest="ocr_command", required=True)
    ocr_refresh = ocr_commands.add_parser(
        "refresh",
        help="Re-run local Tesseract OCR without decoding source media or rerunning ASR.",
    )
    ocr_refresh.add_argument("project_dir", type=Path)
    ocr_refresh.add_argument("--workers", type=int)
    ocr_refresh.add_argument("--language")
    ocr_refresh.add_argument(
        "--packets-only",
        action="store_true",
        help="Repair packet schema projections without invoking Tesseract.",
    )

    review = commands.add_parser(
        "review", help="List, inspect, and apply attributable non-UI review decisions."
    )
    review_commands = review.add_subparsers(dest="review_command", required=True)
    review_list = review_commands.add_parser("list")
    review_list.add_argument("project_dir", type=Path)
    review_show = review_commands.add_parser("show")
    review_show.add_argument("project_dir", type=Path)
    review_show.add_argument("review_id")
    review_apply = review_commands.add_parser("apply")
    review_apply.add_argument("project_dir", type=Path)
    review_apply.add_argument("review_id")
    review_apply.add_argument("--reviewer", required=True)
    review_apply.add_argument("--decision", required=True)
    review_apply.add_argument("--replacement")
    review_apply.add_argument("--rationale", required=True)
    review_bundle = review_commands.add_parser(
        "bundle", help="Create or apply a bounded file-based Codex/subagent visual review bundle."
    )
    review_bundle_commands = review_bundle.add_subparsers(
        dest="review_bundle_command", required=True
    )
    review_bundle_create = review_bundle_commands.add_parser(
        "create", help="Write packet/script requests without copying evidence images."
    )
    review_bundle_create.add_argument("project_dir", type=Path)
    review_bundle_create.add_argument("--output", type=Path)
    review_bundle_create.add_argument("--max-packets", type=int, default=8)
    review_bundle_create.add_argument(
        "--include-provider",
        dest="include_annotation_providers",
        action="append",
        default=[],
        help=(
            "Explicitly re-review observed events from this exact provider ID; "
            "repeat for multiple IDs (default reviews only the pending frontier)."
        ),
    )
    review_bundle_create_all = review_bundle_commands.add_parser(
        "create-all",
        aliases=("batch-create", "create-batch"),
        help="Create bounded no-copy review bundles for every pending project below a root.",
    )
    review_bundle_create_all.add_argument("projects_root", type=Path)
    review_bundle_create_all.add_argument("--output-root", type=Path)
    review_bundle_create_all.add_argument("--max-packets-per-project", type=int, default=8)
    review_bundle_create_all.add_argument("--min-free-bytes", type=int, default=10 * 1024**3)
    review_bundle_create_all.add_argument("--max-bundle-bytes", type=int, default=64 * 1024**2)
    review_bundle_create_all.add_argument("--continue-on-error", action="store_true")
    review_bundle_create_all.add_argument("--dry-run", action="store_true")
    review_bundle_create_all.add_argument(
        "--include-provider",
        dest="include_annotation_providers",
        action="append",
        default=[],
        help="Explicitly re-review observed events from this exact provider ID.",
    )
    review_bundle_apply = review_bundle_commands.add_parser(
        "apply", help="Validate completed subagent annotations and ingest them atomically."
    )
    review_bundle_apply.add_argument("project_dir", type=Path)
    review_bundle_apply.add_argument("--bundle", required=True, type=Path)
    review_bundle_apply.add_argument(
        "--workers",
        type=int,
        default=1,
        choices=(1, 2),
        help=(
            "Bound concurrent file-only Codex response preparation (default: 1); "
            "canonical commits remain deterministic."
        ),
    )
    review_bundle_apply.add_argument(
        "--accept-partial",
        action="store_true",
        help=(
            "Return success when this apply committed at least one response with "
            "no missing responses or validation errors, even if review remains pending. "
            "The default keeps exit code 3 for review_required."
        ),
    )

    finalize = commands.add_parser(
        "finalize", help="Apply a gated attributable final human sign-off."
    )
    finalize.add_argument("project_dir", type=Path)
    finalize.add_argument("--reviewer", required=True)
    finalize.add_argument("--rationale", required=True)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "doctor":
        from .pipeline import doctor_report

        _json(doctor_report(output_path=args.output, offline=args.offline or True))
        return 0
    if args.command in {"diagnostic-bundle", "diagnostics", "support-bundle"}:
        from .diagnostics import create_diagnostic_bundle

        _json(create_diagnostic_bundle(args.output, force=args.force))
        return 0
    if args.command == "plan":
        from .pipeline import plan_input

        _json(
            plan_input(
                args.input,
                output_root=args.output,
                subtitles=args.subtitle,
                transcript=args.transcript,
                preset=args.preset,
                config_path=args.config,
                vision_mode=args.vision_mode,
                offline=args.offline or True,
            )
        )
        return 0
    if args.command == "run":
        from .pipeline import run_pipeline

        explicitly_networked = args.allow_remote_download or args.allow_external_ai
        offline = args.offline or not explicitly_networked

        def progress_callback(payload: Mapping[str, Any]) -> None:
            if not args.progress or payload.get("event") != "chunk_completed":
                return
            completed = int(payload.get("completed_chunks", 0))
            total = int(payload.get("total_chunks", 0))
            remaining = float(payload.get("estimated_remaining_seconds", 0.0))
            remaining_text = (
                f"~{remaining / 60:.1f} min remaining" if remaining > 0 else "complete"
            )
            print(
                f"[ASR] chunk {completed}/{total} ({float(payload.get('fraction', 0.0)):.0%}) — "
                f"{remaining_text}",
                file=sys.stderr,
                flush=True,
            )

        run_result = run_pipeline(
            args.input,
            output_root=args.output,
            subtitles=args.subtitle,
            transcript=args.transcript,
            preset=args.preset,
            config_path=args.config,
            subtitle_mode=args.subtitle_mode,
            language=args.language,
            fidelity_mode=args.fidelity_mode,
            vision_mode=args.vision_mode,
            asr_chunk_seconds=args.asr_chunk_seconds,
            asr_overlap_seconds=args.asr_overlap_seconds,
            semantic_max_packets=args.semantic_max_packets,
            progress_callback=progress_callback,
            resume=args.resume,
            offline=offline,
            allow_remote_download=args.allow_remote_download,
            allow_external_ai=args.allow_external_ai,
        )
        _json(
            {
                "project_dir": str(run_result.project_dir),
                "markdown": str(run_result.markdown_path),
                "status": run_result.status,
                "exit_code": run_result.exit_code,
                "validation": run_result.validation.checks if run_result.validation else None,
                "validation_errors": run_result.validation.errors if run_result.validation else [],
            }
        )
        return run_result.exit_code
    if args.command == "batch":
        from .batch import run_batch

        def batch_progress(payload: Mapping[str, Any]) -> None:
            if not args.progress or payload.get("event") != "chunk_completed":
                return
            completed = int(payload.get("completed_chunks", 0))
            total = int(payload.get("total_chunks", 0))
            remaining = float(payload.get("estimated_remaining_seconds", 0.0))
            remaining_text = f"~{remaining / 60:.1f} min remaining" if remaining > 0 else "complete"
            print(
                f"[batch ASR] chunk {completed}/{total} ({float(payload.get('fraction', 0.0)):.0%}) — {remaining_text}",
                file=sys.stderr,
                flush=True,
            )

        summary = run_batch(
            args.source_root,
            output_root=args.output,
            preset=args.preset,
            vision_mode=args.vision_mode,
            language=args.language,
            compare_sidecars=args.compare_sidecars,
            resume=args.resume,
            min_free_bytes=args.min_free_bytes,
            max_project_bytes=args.max_project_bytes,
            semantic_max_packets=args.semantic_max_packets,
            history_roots=tuple(args.history_root),
            stop_on_blocked=not args.continue_on_blocked,
            dry_run=args.dry_run,
            progress_callback=batch_progress,
        )
        _json(summary)
        if summary.get("blocked"):
            return 4
        if any(item.get("status") == "review_required" for item in summary.get("executed", [])):
            return 3
        return 0
    if args.command == "validate":
        from .validate_output import validate_project

        validation_result = validate_project(args.project_dir)
        _json(
            {
                "valid": validation_result.valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "checks": validation_result.checks,
            }
        )
        return 0 if validation_result.valid else 4
    if args.command == "semantic":
        from .providers.llama_cpp import LlamaCppVisionProvider
        from .semantic_pipeline import run_semantic_pass

        provider = LlamaCppVisionProvider(parallel_slots=args.workers)
        try:
            result = run_semantic_pass(
                args.project_dir,
                provider,
                semantic_max_packets=args.max_packets,
                retry_fallbacks=args.retry_fallbacks,
                retry_semantic_pending=args.retry_semantic_pending,
                semantic_workers=args.workers,
            )
        finally:
            provider.close()
        _json(result)
        status = result.get("status")
        return 0 if status in {"automatically_checked", "human_reviewed", "fully_verified"} else (
            3 if status == "review_required" else 4
        )
    if args.command == "semantic-batch":
        from .providers.llama_cpp import LlamaCppVisionProvider
        from .semantic_pipeline import run_semantic_batch

        provider = LlamaCppVisionProvider(parallel_slots=args.workers)
        try:
            result = run_semantic_batch(
                args.output_root,
                provider,
                semantic_max_packets=args.max_packets_per_project,
                retry_fallbacks=args.retry_fallbacks,
                retry_semantic_pending=args.retry_semantic_pending,
                min_free_bytes=args.min_free_bytes,
                continue_on_error=args.continue_on_error,
                semantic_workers=args.workers,
            )
        finally:
            provider.close()
        _json(result if args.full_output else _compact_semantic_batch_result(result))
        return 4 if result.get("blocked") else (3 if result.get("status") == "review_required" else 0)
    if args.command == "cache" and args.cache_command == "purge":
        from .cache import purge_project_cache

        removed = purge_project_cache(args.project_dir)
        _json({"project_dir": str(args.project_dir.resolve()), "removed_cache_files": removed})
        return 0
    if args.command == "cache" and args.cache_command == "compact":
        from .cache import compact_completed_checkpoints, completed_checkpoint_compaction_plan
        from .pipeline import _publish_resource_telemetry
        from .validate_output import (
            refresh_validation_receipt_signature,
            validate_project,
            write_validation_receipt,
        )

        plan = completed_checkpoint_compaction_plan(args.project_dir)
        if not args.apply:
            _json({"project_dir": str(args.project_dir.resolve()), **plan})
            return 0
        validation_result = validate_project(args.project_dir, verify_metadata=True)
        if not validation_result.valid:
            _json(
                {
                    "project_dir": str(args.project_dir.resolve()),
                    "dry_run": False,
                    "applied": False,
                    "validation_errors": validation_result.errors,
                    "plan": plan,
                }
            )
            return 4
        result = compact_completed_checkpoints(args.project_dir)
        project_dir = args.project_dir.expanduser().resolve(strict=True)
        canonical_path = project_dir / ".state" / "canonical-project.json"
        try:
            project = json.loads(canonical_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _json(
                {
                    "project_dir": str(project_dir),
                    "dry_run": False,
                    "applied": True,
                    "validation_errors": [f"Unable to refresh project telemetry: {exc}"],
                    **result,
                }
            )
            return 4
        if not isinstance(project, dict):
            _json(
                {
                    "project_dir": str(project_dir),
                    "dry_run": False,
                    "applied": True,
                    "validation_errors": ["Canonical project root must be an object"],
                    **result,
                }
            )
            return 4
        manifest = project.setdefault("manifest", {})
        if not isinstance(manifest, dict):
            _json(
                {
                    "project_dir": str(project_dir),
                    "dry_run": False,
                    "applied": True,
                    "validation_errors": ["Canonical project manifest must be an object"],
                    **result,
                }
            )
            return 4
        performance = manifest.setdefault("performance", {})
        if not isinstance(performance, dict):
            performance = {}
            manifest["performance"] = performance
        performance["checkpoint_compaction"] = result
        _publish_resource_telemetry(project_dir, project, manifest)
        post_validation = validate_project(project_dir, verify_metadata=True)
        validation_errors = list(post_validation.errors)
        if post_validation.valid:
            run_cache_key = str(manifest.get("run_cache_key") or "")
            write_validation_receipt(
                project_dir,
                project,
                run_cache_key=run_cache_key,
                validation=post_validation,
            )
            refresh_validation_receipt_signature(project_dir)
        _json(
            {
                "project_dir": str(project_dir),
                "dry_run": False,
                "applied": True,
                "validation_errors": validation_errors,
                "checks": post_validation.checks,
                "resource_usage": manifest.get("performance", {}).get("resource_usage"),
                **result,
            }
        )
        return 0 if post_validation.valid else 4
    if args.command == "retention":
        from .retention import orphan_report, prune_orphans, prune_runs, retention_report

        if args.retention_command == "report":
            _json(retention_report(args.root))
            return 0
        if args.retention_command == "orphans":
            _json(orphan_report(args.root))
            return 0
        if args.retention_command == "prune":
            _json(prune_runs(args.root, keep=args.keep, apply=args.apply))
            return 0
        if args.retention_command == "prune-orphans":
            _json(prune_orphans(args.root, apply=args.apply))
            return 0
    if args.command == "models":
        from .model_store import fetch_model, list_models, model_report, remove_model, verify_model

        if args.models_command == "report":
            _json(
                model_report(
                    args.root,
                    force_full=args.full,
                    include_workers=args.with_workers,
                )
            )
            return 0
        if args.models_command == "list":
            _json(list_models(args.root))
            return 0
        if args.models_command == "fetch":
            _json(fetch_model(args.name, args.root, revision=args.revision))
            return 0
        if args.models_command == "verify":
            _json(
                verify_model(args.name, args.root, force_full=args.full)
                if args.name
                else list_models(args.root, force_full=args.full)
            )
            return 0
        if args.models_command == "remove":
            _json(remove_model(args.name, args.root))
            return 0
    if args.command == "workers":
        from .worker_store import install_worker, list_workers, verify_worker

        if args.workers_command == "list":
            _json(list_workers(args.root))
            return 0
        if args.workers_command == "install":
            _json(install_worker(args.name, args.root))
            return 0
        if args.workers_command == "verify":
            _json(
                verify_worker(args.name, args.root)
                if args.name
                else list_workers(args.root, force=True)
            )
            return 0
    if args.command == "evidence":
        from .evidence import (
            ingest_project_observation,
            show_image_metadata,
            show_packet,
            verify_image_metadata,
        )

        if args.evidence_command == "metadata" and args.metadata_command == "show":
            value = show_image_metadata(args.project_dir, args.image_id)
            if args.json_output:
                _json(value)
            else:
                print(f"Image: {args.image_id}")
                print(f"Revision: {value['latest_revision_id']} (#{value['revision_number']})")
                print(f"Semantic status: {value['semantic_status']}")
                print(
                    f"Description: {value['current_factual_description'] or '[semantic description pending]'}"
                )
                print("Claims:")
                for claim in value["claims"]:
                    print(f"  {claim['claim_id']} [{claim['status']}] {claim['statement']}")
                print("Unknowns: " + ("; ".join(value["explicit_unknowns"]) or "none"))
                print("Unanswered: " + ("; ".join(value["unanswered_questions"]) or "none"))
                print("Canonical match: yes")
            return 0
        if args.evidence_command == "metadata" and args.metadata_command == "verify":
            _json(verify_image_metadata(args.project_dir, args.image_id))
            return 0
        if args.evidence_command == "packet" and args.packet_command == "show":
            value = show_packet(args.project_dir, args.event_or_frame_id)
            _json(value)
            return 0
        if args.evidence_command == "observation" and args.observation_command == "ingest":
            _json(
                ingest_project_observation(
                    args.project_dir, args.input, base_revision=args.base_revision
                )
            )
            return 0
        if args.evidence_command == "ocr" and args.ocr_command == "refresh":
            from .ocr_refresh import refresh_project_ocr, repair_project_ocr_packets

            if args.packets_only:
                result = repair_project_ocr_packets(args.project_dir)
            else:
                result = refresh_project_ocr(
                    args.project_dir,
                    workers=args.workers,
                    language=args.language,
                )
            _json(result)
            return 0 if not result.get("validation_errors") else 4
    if args.command == "review":
        from .review import apply_review, list_review_items, show_review_item

        if args.review_command == "list":
            _json(list_review_items(args.project_dir))
            return 0
        if args.review_command == "show":
            _json(show_review_item(args.project_dir, args.review_id))
            return 0
        if args.review_command == "apply":
            _json(
                apply_review(
                    args.project_dir,
                    args.review_id,
                    reviewer=args.reviewer,
                    decision=args.decision,
                    replacement=args.replacement,
                    rationale=args.rationale,
                )
            )
            return 0
        if args.review_command == "bundle":
            from .subagent_review import apply_review_bundle, create_review_bundle

            if args.review_bundle_command == "create":
                _json(
                    create_review_bundle(
                        args.project_dir,
                        output_dir=args.output,
                        max_packets=args.max_packets,
                        include_annotation_providers=args.include_annotation_providers,
                    )
                )
                return 0
            if args.review_bundle_command == "apply":
                result = apply_review_bundle(
                    args.project_dir,
                    args.bundle,
                    semantic_workers=args.workers,
                )
                _json(result)
                if args.accept_partial:
                    bundle_summary_value: Any = result.get("summary")
                    applied_value: Any = (
                        bundle_summary_value.get("applied")
                        if isinstance(bundle_summary_value, Mapping)
                        else None
                    )
                    validation_errors_value: Any = result.get("validation_errors")
                    missing_value: Any = result.get("missing_candidate_ids")
                    if (
                        isinstance(applied_value, list)
                        and applied_value
                        and isinstance(validation_errors_value, list)
                        and not validation_errors_value
                        and isinstance(missing_value, list)
                        and not missing_value
                    ):
                        return 0
                return 0 if result.get("status") in {
                    "automatically_checked",
                    "human_reviewed",
                    "fully_verified",
                } else 3
            if args.review_bundle_command in {"create-all", "batch-create", "create-batch"}:
                from .review_batch import create_review_bundles

                result = create_review_bundles(
                    args.projects_root,
                    output_root=args.output_root,
                    max_packets_per_project=args.max_packets_per_project,
                    min_free_bytes=args.min_free_bytes,
                    max_bundle_bytes=args.max_bundle_bytes,
                    continue_on_error=args.continue_on_error,
                    dry_run=args.dry_run,
                    include_annotation_providers=args.include_annotation_providers,
                )
                _json(result)
                return 4 if result.get("blocked") else 0
    if args.command == "finalize":
        from .review import finalize_project

        _json(finalize_project(args.project_dir, reviewer=args.reviewer, rationale=args.rationale))
        return 0
    raise InputError("Unknown command")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        code = _run(args)
    except (InputError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 2
    except ReviewRequired as exc:
        print(f"review required: {exc}", file=sys.stderr)
        code = 3
    except (BlockedError, ValidationFailure) as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        code = 4
    except Exception as exc:
        print(f"internal failure: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


__all__ = ["main"]
