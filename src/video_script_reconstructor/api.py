"""Stable, synchronous Python facade for the Fast Video Analyzer.

The CLI remains the broad compatibility surface.  This module intentionally
exposes only the small workflow that is already exercised by the installed
package: plan one input, run one input, validate one project, and inspect its
review queue.  Pipeline stages, provider adapters, persisted dictionaries,
and mutation-heavy review operations remain private implementation details.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from .errors import InputError
from .pipeline import plan_input as _plan_input
from .pipeline import run_pipeline as _run_pipeline
from .review import list_review_items as _list_review_items
from .review import show_review_item as _show_review_item
from .validate_output import validate_project as _validate_project

ProjectStatus = Literal[
    "automatically_checked",
    "review_required",
    "human_reviewed",
    "fully_verified",
    "blocked",
]
InputClassification = Literal["video", "audio", "transcript", "remote_media"]
ExitCode = Literal[0, 1, 2, 3, 4]
Preset = Literal["strict", "balanced"]
SubtitleMode = Literal["auto", "provided-only", "force-asr", "compare-all"]
FidelityMode = Literal["verbatim", "clean-verbatim", "production-script"]
VisionMode = Literal["auto", "host-agent", "local", "external", "none"]


def _freeze(value: Any) -> Any:
    """Freeze JSON-shaped payloads so result snapshots cannot be mutated."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _mapping(value: Any, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Stable API payload field {field_name!r} is not an object")
    frozen = _freeze(value)
    if not isinstance(frozen, Mapping):
        raise RuntimeError(f"Stable API payload field {field_name!r} is not an object")
    return cast(Mapping[str, object], frozen)


def _strings(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RuntimeError(f"Stable API payload field {field_name!r} is not a list")
    return tuple(str(item) for item in value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _status(value: Any) -> ProjectStatus:
    allowed = {
        "automatically_checked",
        "review_required",
        "human_reviewed",
        "fully_verified",
        "blocked",
    }
    text = str(value)
    if text not in allowed:
        raise RuntimeError(f"Stable API received unknown project status: {text!r}")
    return cast(ProjectStatus, text)


def _optional_status(value: Any) -> ProjectStatus | None:
    return None if value is None else _status(value)


def _exit_code(value: Any) -> ExitCode:
    number = int(value)
    if number not in {0, 1, 2, 3, 4}:
        raise RuntimeError(f"Stable API received unknown exit code: {number}")
    return cast(ExitCode, number)


def _path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _path_tuple(values: Sequence[str | Path]) -> tuple[Path, ...]:
    return tuple(_path(value) for value in values)


@dataclass(frozen=True, slots=True)
class Plan:
    """No-download, no-full-processing plan for one input."""

    schema_version: str
    input_classification: InputClassification
    probe: Mapping[str, object] | None
    likely_transcript_sources: tuple[str, ...]
    planned_stages: tuple[str, ...]
    strict_prerequisites: tuple[str, ...]
    asr_expected: bool
    ocr_expected: bool
    visual_review_expected: bool
    image_metadata_plan: str
    semantic_pending_possible: bool
    network_actions_requiring_permission: tuple[str, ...]
    estimated_evidence_images: int
    estimated_disk_bytes: int
    asr_plan: Mapping[str, object]
    output_path: Path
    output_contract: str
    no_full_processing_statement: str
    offline: bool


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Independent validation report for one generated project."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    checks: Mapping[str, object]
    project_status: ProjectStatus | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    """Result of one synchronous analysis run."""

    project_dir: Path
    markdown_path: Path
    status: ProjectStatus
    exit_code: ExitCode
    validation: ValidationResult | None


@dataclass(frozen=True, slots=True)
class CompetingEvidence:
    """One alternative claim attached to a detailed review item."""

    claim_id: str
    statement: str | None
    status: str | None
    alternatives: tuple[str, ...]


def _empty_source_ids() -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ReviewItem:
    """Typed review-queue item; detail-only fields are populated by ``show``."""

    review_id: str
    severity: str
    category: str
    problem: str
    required_action: str
    blocking: bool
    start_ms: int | None = None
    end_ms: int | None = None
    block_ids: tuple[str, ...] = ()
    segment_ids: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()
    frame_ids: tuple[str, ...] = ()
    ocr_observation_ids: tuple[str, ...] = ()
    image_claim_ids: tuple[str, ...] = ()
    metadata_revision_ids: tuple[str, ...] = ()
    sufficiency_decision_ids: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    decision: str | None = None
    replacement: str | None = None
    reviewer: str | None = None
    decision_timestamp_utc: str | None = None
    rationale: str | None = None
    image_paths: tuple[Path, ...] = ()
    source_ids: Mapping[str, tuple[str, ...]] = field(default_factory=_empty_source_ids)
    competing_evidence: tuple[CompetingEvidence, ...] = ()


def _plan_from_payload(payload: Mapping[str, Any]) -> Plan:
    try:
        classification = str(payload["input_classification"])
        if classification not in {"video", "audio", "transcript", "remote_media"}:
            raise RuntimeError(
                f"Stable API received unknown input classification: {classification!r}"
            )
        probe_value = payload.get("probe")
        probe = None if probe_value is None else _mapping(probe_value, field_name="probe")
        return Plan(
            schema_version=str(payload["schema_version"]),
            input_classification=cast(InputClassification, classification),
            probe=probe,
            likely_transcript_sources=_strings(
                payload.get("likely_transcript_sources"), field_name="likely_transcript_sources"
            ),
            planned_stages=_strings(payload.get("planned_stages"), field_name="planned_stages"),
            strict_prerequisites=_strings(
                payload.get("strict_prerequisites"), field_name="strict_prerequisites"
            ),
            asr_expected=bool(payload["asr_expected"]),
            ocr_expected=bool(payload["ocr_expected"]),
            visual_review_expected=bool(payload["visual_review_expected"]),
            image_metadata_plan=str(payload["image_metadata_plan"]),
            semantic_pending_possible=bool(payload["semantic_pending_possible"]),
            network_actions_requiring_permission=_strings(
                payload.get("network_actions_requiring_permission"),
                field_name="network_actions_requiring_permission",
            ),
            estimated_evidence_images=int(payload["estimated_evidence_images"]),
            estimated_disk_bytes=int(payload["estimated_disk_bytes"]),
            asr_plan=_mapping(payload["asr_plan"], field_name="asr_plan"),
            output_path=Path(str(payload["output_path"])),
            output_contract=str(payload["output_contract"]),
            no_full_processing_statement=str(payload["no_full_processing_statement"]),
            offline=bool(payload["offline"]),
        )
    except KeyError as exc:
        raise RuntimeError(f"Stable API plan payload is missing {exc.args[0]!r}") from exc


def _validation_from_internal(value: Any) -> ValidationResult:
    return ValidationResult(
        valid=bool(value.valid),
        errors=_strings(value.errors, field_name="validation.errors"),
        warnings=_strings(value.warnings, field_name="validation.warnings"),
        checks=_mapping(value.checks, field_name="validation.checks"),
        project_status=_optional_status(value.project_status),
    )


def _run_result_from_internal(value: Any) -> RunResult:
    validation = None if value.validation is None else _validation_from_internal(value.validation)
    return RunResult(
        project_dir=Path(value.project_dir),
        markdown_path=Path(value.markdown_path),
        status=_status(value.status),
        exit_code=_exit_code(value.exit_code),
        validation=validation,
    )


def _review_from_payload(payload: Mapping[str, Any]) -> ReviewItem:
    time_range = payload.get("time_range_ms")
    if isinstance(time_range, Mapping):
        start_value = time_range.get("start")
        end_value = time_range.get("end")
    else:
        start_value = payload.get("start_ms")
        end_value = payload.get("end_ms")

    raw_source_ids = payload.get("source_ids")
    source_ids: Mapping[str, tuple[str, ...]]
    if raw_source_ids is None:
        source_ids = _empty_source_ids()
    elif isinstance(raw_source_ids, Mapping):
        source_ids = MappingProxyType(
            {
                str(key): _strings(value, field_name=f"review.source_ids.{key}")
                for key, value in raw_source_ids.items()
            }
        )
    else:
        raise RuntimeError("Stable API review payload field 'source_ids' is not an object")

    raw_competing = payload.get("competing_evidence", ())
    if isinstance(raw_competing, (str, bytes)) or not isinstance(raw_competing, Sequence):
        raise RuntimeError("Stable API review payload field 'competing_evidence' is not a list")
    competing: list[CompetingEvidence] = []
    for item in raw_competing:
        if not isinstance(item, Mapping):
            raise RuntimeError("Stable API competing evidence item is not an object")
        competing.append(
            CompetingEvidence(
                claim_id=str(item.get("claim_id", "")),
                statement=_optional_string(item.get("statement")),
                status=_optional_string(item.get("status")),
                alternatives=_strings(item.get("alternatives"), field_name="review.alternatives"),
            )
        )

    raw_paths = payload.get("image_paths", ())
    image_paths = tuple(Path(str(value)) for value in _strings(raw_paths, field_name="image_paths"))
    try:
        return ReviewItem(
            review_id=str(payload["review_id"]),
            severity=str(payload["severity"]),
            category=str(payload["category"]),
            problem=str(payload["problem"]),
            required_action=str(payload["required_action"]),
            blocking=bool(payload["blocking"]),
            start_ms=None if start_value is None else int(start_value),
            end_ms=None if end_value is None else int(end_value),
            block_ids=_strings(payload.get("block_ids"), field_name="review.block_ids"),
            segment_ids=_strings(payload.get("segment_ids"), field_name="review.segment_ids"),
            event_ids=_strings(payload.get("event_ids"), field_name="review.event_ids"),
            frame_ids=_strings(payload.get("frame_ids"), field_name="review.frame_ids"),
            ocr_observation_ids=_strings(
                payload.get("ocr_observation_ids"), field_name="review.ocr_observation_ids"
            ),
            image_claim_ids=_strings(
                payload.get("image_claim_ids"), field_name="review.image_claim_ids"
            ),
            metadata_revision_ids=_strings(
                payload.get("metadata_revision_ids"), field_name="review.metadata_revision_ids"
            ),
            sufficiency_decision_ids=_strings(
                payload.get("sufficiency_decision_ids"),
                field_name="review.sufficiency_decision_ids",
            ),
            alternatives=_strings(payload.get("alternatives"), field_name="review.alternatives"),
            decision=_optional_string(payload.get("decision")),
            replacement=_optional_string(payload.get("replacement")),
            reviewer=_optional_string(payload.get("reviewer")),
            decision_timestamp_utc=_optional_string(payload.get("decision_timestamp_utc")),
            rationale=_optional_string(payload.get("rationale")),
            image_paths=image_paths,
            source_ids=source_ids,
            competing_evidence=tuple(competing),
        )
    except KeyError as exc:
        raise RuntimeError(f"Stable API review payload is missing {exc.args[0]!r}") from exc


def plan(
    input_value: str | Path,
    *,
    output_root: str | Path | None = None,
    subtitles: Sequence[str | Path] = (),
    transcript: str | Path | None = None,
    preset: Preset = "strict",
    config_path: str | Path | None = None,
    vision_mode: VisionMode = "host-agent",
    offline: bool = True,
) -> Plan:
    """Inspect one input without downloading models or processing all media."""

    try:
        payload = _plan_input(
            input_value,
            output_root=None if output_root is None else _path(output_root),
            subtitles=_path_tuple(subtitles),
            transcript=None if transcript is None else _path(transcript),
            preset=preset,
            config_path=None if config_path is None else _path(config_path),
            vision_mode=vision_mode,
            offline=offline,
        )
    except FileNotFoundError as exc:
        raise InputError(str(exc)) from exc
    return _plan_from_payload(payload)


def run(
    input_value: str | Path,
    *,
    output_root: str | Path | None = None,
    subtitles: Sequence[str | Path] = (),
    transcript: str | Path | None = None,
    preset: Preset = "strict",
    config_path: str | Path | None = None,
    subtitle_mode: SubtitleMode = "auto",
    language: str | None = None,
    fidelity_mode: FidelityMode = "verbatim",
    vision_mode: VisionMode = "host-agent",
    asr_chunk_seconds: int | None = None,
    asr_overlap_seconds: int | None = None,
    semantic_max_packets: int | None = None,
    resume: bool = True,
    offline: bool = True,
    allow_remote_download: bool = False,
    allow_external_ai: bool = False,
) -> RunResult:
    """Run one input synchronously and return its typed, auditable result.

    Review-required and blocked outcomes are returned in ``RunResult`` with
    their documented status and exit code.  Expected input/configuration
    failures raise the package's typed exceptions.
    """

    try:
        result = _run_pipeline(
            input_value,
            output_root=None if output_root is None else _path(output_root),
            subtitles=_path_tuple(subtitles),
            transcript=None if transcript is None else _path(transcript),
            preset=preset,
            config_path=None if config_path is None else _path(config_path),
            subtitle_mode=subtitle_mode,
            language=language,
            fidelity_mode=fidelity_mode,
            vision_mode=vision_mode,
            asr_chunk_seconds=asr_chunk_seconds,
            asr_overlap_seconds=asr_overlap_seconds,
            semantic_max_packets=semantic_max_packets,
            resume=resume,
            offline=offline,
            allow_remote_download=allow_remote_download,
            allow_external_ai=allow_external_ai,
        )
    except FileNotFoundError as exc:
        raise InputError(str(exc)) from exc
    return _run_result_from_internal(result)


def validate(
    project_dir: str | Path,
    *,
    verify_metadata: bool = True,
) -> ValidationResult:
    """Validate one generated project independently of a prior run."""

    try:
        result = _validate_project(_path(project_dir), verify_metadata=verify_metadata)
    except FileNotFoundError as exc:
        raise InputError(str(exc)) from exc
    return _validation_from_internal(result)


def list_review_items(project_dir: str | Path) -> tuple[ReviewItem, ...]:
    """Return an immutable snapshot of the project's review queue."""

    try:
        payload = _list_review_items(_path(project_dir))
    except FileNotFoundError as exc:
        raise InputError(str(exc)) from exc
    return tuple(_review_from_payload(item) for item in payload)


def show_review_item(project_dir: str | Path, review_id: str) -> ReviewItem:
    """Return one review item with evidence paths and competing claims."""

    try:
        payload = _show_review_item(_path(project_dir), review_id)
    except FileNotFoundError as exc:
        raise InputError(str(exc)) from exc
    return _review_from_payload(payload)


__all__ = [
    "CompetingEvidence",
    "ExitCode",
    "FidelityMode",
    "InputClassification",
    "Plan",
    "Preset",
    "ProjectStatus",
    "ReviewItem",
    "RunResult",
    "SubtitleMode",
    "ValidationResult",
    "VisionMode",
    "list_review_items",
    "plan",
    "run",
    "show_review_item",
    "validate",
]
