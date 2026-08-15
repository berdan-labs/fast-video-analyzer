from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from .errors import InputError, ValidationFailure
from .scene_detection import SurveyCandidate
from .schemas import VisualEvent
from .vision_packets import VisionAnnotation, VisionPacket, validate_annotation_for_packet

SEMANTIC_PENDING_DESCRIPTION = "[visual evidence retained; semantic description pending review]"


@dataclass(frozen=True)
class VisualEventBuildResult:
    event: VisualEvent
    review_required: bool
    unresolved_questions: tuple[str, ...]


def _event_id(index: int) -> str:
    if index <= 0:
        raise InputError("visual event index must be positive")
    return f"V{index:06d}"


def _event_interval(packet: VisionPacket) -> tuple[int, int]:
    times = [frame.actual_ms for frame in packet.frames]
    return min(times), max(times)


def annotation_to_visual_event(
    annotation: VisionAnnotation | dict[str, object],
    packet: VisionPacket,
    *,
    event_index: int,
    annotation_provider: str,
    scene_or_state_id: str | None = None,
    ocr_observation_ids: Sequence[str] | None = None,
    image_claim_ids: Sequence[str] = (),
    metadata_revision_ids: Sequence[str] = (),
) -> VisualEventBuildResult:
    parsed = validate_annotation_for_packet(annotation, packet)
    start_ms, end_ms = _event_interval(packet)
    roles: dict[str, list[str]] = {}
    for frame_id, role in parsed.before_action_after_roles.items():
        roles.setdefault(role, []).append(frame_id)
    linked_ocr = (
        list(ocr_observation_ids)
        if ocr_observation_ids is not None
        else [
            observation.observation_id
            for observation in packet.raw_ocr
            if observation.frame_id in parsed.evidence_frame_ids
        ]
    )
    consequential = bool(parsed.consequential_changes)
    importance: Literal["consequential", "supporting"] = (
        "consequential" if consequential else "supporting"
    )
    event = VisualEvent(
        event_id=_event_id(event_index),
        start_ms=start_ms,
        end_ms=end_ms,
        event_type=parsed.event_type,
        scene_or_state_id=scene_or_state_id,
        evidence_frame_ids=list(parsed.evidence_frame_ids),
        before_action_after_roles=roles,
        ocr_observation_ids=linked_ocr,
        factual_grounded_description=parsed.factual_visible_description,
        importance=importance,
        confidence=parsed.confidence,
        uncertainty=list(parsed.uncertainty),
        annotation_provider=annotation_provider,
        review_status="review_required" if parsed.uncertainty else "automatically_checked",
        image_claim_ids=list(image_claim_ids),
        metadata_revision_ids=list(metadata_revision_ids),
    )
    return VisualEventBuildResult(event, bool(parsed.uncertainty), tuple(parsed.uncertainty))


def pending_visual_event(
    packet: VisionPacket,
    *,
    event_index: int,
    scene_or_state_id: str | None = None,
) -> VisualEventBuildResult:
    start_ms, end_ms = _event_interval(packet)
    frame_ids = [frame.frame_id for frame in packet.frames]
    roles: dict[str, list[str]] = {}
    for frame in packet.frames:
        role = "action" if frame.role == "focus" else frame.role
        roles.setdefault(role, []).append(frame.frame_id)
    questions = tuple(packet.questions)
    event = VisualEvent(
        event_id=_event_id(event_index),
        start_ms=start_ms,
        end_ms=end_ms,
        event_type="semantic_annotation_pending",
        scene_or_state_id=scene_or_state_id,
        evidence_frame_ids=frame_ids,
        before_action_after_roles=roles,
        ocr_observation_ids=[item.observation_id for item in packet.raw_ocr],
        factual_grounded_description=SEMANTIC_PENDING_DESCRIPTION,
        importance="supporting",
        confidence=None,
        uncertainty=list(questions) or ["No semantic annotator was available"],
        annotation_provider=None,
        review_status="review_required",
        image_claim_ids=[],
        metadata_revision_ids=[],
    )
    return VisualEventBuildResult(event, True, questions)


def deterministic_events_from_candidates(
    candidates: Iterable[SurveyCandidate],
    *,
    first_event_number: int = 1,
) -> tuple[VisualEventBuildResult, ...]:
    """Retain candidate evidence without fabricating a semantic description."""
    results: list[VisualEventBuildResult] = []
    for offset, candidate in enumerate(sorted(candidates, key=lambda item: item.requested_ms)):
        time_ms = candidate.actual_ms if candidate.actual_ms is not None else candidate.requested_ms
        event = VisualEvent(
            event_id=_event_id(first_event_number + offset),
            start_ms=time_ms,
            end_ms=time_ms,
            event_type="survey_candidate",
            scene_or_state_id=None,
            evidence_frame_ids=[],
            before_action_after_roles={},
            ocr_observation_ids=[],
            factual_grounded_description=SEMANTIC_PENDING_DESCRIPTION,
            importance="supporting",
            confidence=None,
            uncertainty=[
                f"Semantic meaning unresolved for candidate reasons: {', '.join(candidate.reasons)}"
            ],
            annotation_provider="deterministic-survey",
            review_status="review_required",
            image_claim_ids=[],
            metadata_revision_ids=[],
        )
        results.append(VisualEventBuildResult(event, True, tuple(event.uncertainty)))
    return tuple(results)


def validate_event_grounding(event: VisualEvent, available_frame_ids: Iterable[str]) -> None:
    available = set(available_frame_ids)
    missing = set(event.evidence_frame_ids) - available
    if missing:
        raise ValidationFailure(
            f"Visual event {event.event_id} references unavailable frames: {sorted(missing)}"
        )
    role_frames = {
        frame_id for frames in event.before_action_after_roles.values() for frame_id in frames
    }
    if role_frames - set(event.evidence_frame_ids):
        raise ValidationFailure(
            f"Visual event {event.event_id} assigns roles to frames outside its evidence list"
        )
