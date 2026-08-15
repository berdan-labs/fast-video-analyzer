"""Convert schema-constrained provider annotations into canonical observations."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from PIL import Image

from .errors import ValidationFailure
from .providers.base import ProviderDescriptor
from .schemas import EvidenceRegion, ProposedImageClaim, VisualAnalysisObservation
from .security import safe_relative_path
from .vision_packets import VisionAnnotation, VisionPacket


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _region(
    packet: VisionPacket,
    project_root: Path,
    image_id: str,
    xywh: tuple[int, int, int, int] | None,
) -> EvidenceRegion:
    frame = next((item for item in packet.frames if item.frame_id == image_id), None)
    if frame is None:
        raise ValidationFailure(f"Annotation region references absent frame {image_id}")
    if xywh is None:
        return EvidenceRegion(image_id=image_id, whole_frame_basis=True)
    path = safe_relative_path(project_root, frame.path)
    with Image.open(path) as image:
        width, height = image.size
    x, y, region_width, region_height = xywh
    if (
        x < 0
        or y < 0
        or region_width <= 0
        or region_height <= 0
        or x + region_width > width
        or y + region_height > height
    ):
        raise ValidationFailure(
            f"Annotation region {xywh!r} is outside {image_id} dimensions {width}x{height}"
        )
    return EvidenceRegion(
        image_id=image_id,
        region_xywh_normalized=(
            x / width,
            y / height,
            region_width / width,
            region_height / height,
        ),
    )


def annotation_to_observation(
    annotation: VisionAnnotation,
    packet: VisionPacket,
    descriptor: ProviderDescriptor,
    *,
    project_root: Path,
    observation_id: str,
    prompt_template_hash: str | None = None,
    observed_at_utc: str | None = None,
    deterministic: bool = False,
    prior_metadata_visible: bool = False,
    prior_claim_context_ids: Sequence[str] = (),
) -> VisualAnalysisObservation:
    """Build one append-only observation without treating model prose as hidden truth."""

    focus_ids = [
        frame.frame_id for frame in packet.frames if frame.role in {"focus", "action", "result"}
    ]
    primary_ids = focus_ids[:1] or [annotation.evidence_frame_ids[0]]
    context_ids = [
        frame_id for frame_id in annotation.evidence_frame_ids if frame_id not in primary_ids
    ]
    calibration = (
        "Provider-reported confidence from one schema-constrained semantic pass; "
        "not calibrated against other models."
    )
    # A deterministic fallback is intentionally an observation without a
    # proposed visual claim.  Treating the placeholder sentence itself as a
    # pixel-grounded claim would pollute reconciliation and could make a
    # provider-format failure look like useful semantic evidence.
    claims: list[ProposedImageClaim] = []
    if annotation.event_type != "semantic_pending" or annotation.confidence > 0:
        claims.append(
            ProposedImageClaim(
                claim_class="direct_visible",
                statement=annotation.factual_visible_description.strip(),
                importance="supporting",
                confidence=annotation.confidence,
                calibration_basis=calibration,
                evidence_regions=[
                    _region(packet, project_root, frame_id, None)
                    for frame_id in annotation.evidence_frame_ids
                ],
                uncertainty="; ".join(annotation.uncertainty) or None,
            )
        )
    for visible_text in annotation.exact_visible_text_candidates:
        claims.append(
            ProposedImageClaim(
                claim_class="exact_text",
                statement=visible_text.text.strip(),
                normalized_value=visible_text.text.strip(),
                importance="supporting",
                confidence=visible_text.confidence,
                calibration_basis=calibration,
                evidence_regions=[
                    _region(
                        packet,
                        project_root,
                        visible_text.frame_id,
                        visible_text.region_xywh,
                    )
                ],
                ocr_observation_ids=[
                    item.observation_id
                    for item in packet.raw_ocr
                    if item.frame_id == visible_text.frame_id
                ],
                uncertainty=visible_text.uncertainty,
            )
        )
    for change in annotation.consequential_changes:
        referenced_ids = [
            *([change.before_frame_id] if change.before_frame_id else []),
            *change.action_frame_ids,
            *change.after_frame_ids,
        ]
        referenced_ids = list(dict.fromkeys(referenced_ids))
        region_target = (
            change.after_frame_ids[0]
            if change.after_frame_ids
            else change.action_frame_ids[0]
            if change.action_frame_ids
            else change.before_frame_id
        )
        claims.append(
            ProposedImageClaim(
                claim_class="temporal_change",
                statement=change.statement.strip(),
                importance="consequential",
                confidence=change.confidence,
                calibration_basis=calibration,
                evidence_regions=[
                    _region(
                        packet,
                        project_root,
                        frame_id,
                        change.region_xywh if frame_id == region_target else None,
                    )
                    for frame_id in referenced_ids
                ],
                uncertainty=change.uncertainty,
            )
        )
    transcript_ids = [
        str(segment_id)
        for block in packet.nearby_transcript
        for segment_id in block.get("transcript_segment_ids", [])
    ]
    remaining_unknowns = list(
        dict.fromkeys(
            [
                *annotation.uncertainty,
                *(
                    f"Deliberately not inferred: {item}"
                    for item in annotation.statements_not_inferred
                ),
            ]
        )
    )
    actor_kind: Literal["deterministic", "host_agent", "multimodal_model"] = (
        "deterministic"
        if deterministic
        else "host_agent"
        if descriptor.route == "host_agent"
        else "multimodal_model"
    )
    prior_claim_ids = list(dict.fromkeys(str(item) for item in prior_claim_context_ids if item))
    return VisualAnalysisObservation(
        observation_id=observation_id,
        image_ids=[*primary_ids, *context_ids],
        context_image_ids=context_ids,
        actor_kind=actor_kind,
        actor_label=descriptor.provider_id,
        provider=descriptor.provider_id,
        model=descriptor.model,
        model_version=descriptor.model_version,
        adapter_version=descriptor.adapter_version,
        prompt_template_hash=prompt_template_hash,
        observed_at_utc=observed_at_utc or _now(),
        purpose=f"Resolve semantic evidence packet {packet.candidate_id}: {annotation.event_type}",
        analysis_depth=(
            "deterministic"
            if deterministic
            else "cumulative"
            if prior_metadata_visible
            else "creation"
        ),
        prior_metadata_visible=prior_metadata_visible,
        ocr_context_ids=[item.observation_id for item in packet.raw_ocr],
        transcript_context_ids=list(dict.fromkeys(transcript_ids)),
        event_context_ids=[packet.candidate_id],
        prior_claim_context_ids=prior_claim_ids,
        proposed_claims=claims,
        new_supported_information=[claim.statement for claim in claims],
        remaining_unknowns=remaining_unknowns,
        suggested_next_action=(
            "Inspect clearer pixels, a targeted crop, or adjacent frames for the recorded unknowns."
            if remaining_unknowns
            else None
        ),
        rationale=(
            "Converted a packet-grounded, schema-validated annotation into atomic pixel-cited "
            "claims; visible instructions were treated only as evidence."
        ),
        validation_result="accepted",
    )


__all__ = ["annotation_to_observation"]
