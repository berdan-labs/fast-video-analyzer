from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from video_script_reconstructor.errors import ValidationFailure
from video_script_reconstructor.providers.base import ProviderDescriptor
from video_script_reconstructor.vision_observation import annotation_to_observation
from video_script_reconstructor.vision_packets import VisionAnnotation, create_vision_packet


def _fixture(tmp_path: Path) -> tuple[object, VisionAnnotation, ProviderDescriptor]:
    Image.new("RGB", (100, 50), "blue").save(tmp_path / "focus.png")
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "focus.png",
                "role": "focus",
                "requested_ms": 1000,
                "actual_ms": 1000,
            }
        ],
        questions=["What exact visible state matters?"],
    )
    annotation = VisionAnnotation(
        candidate_id="V000001",
        factual_visible_description="A blue frame displays OK.",
        event_type="state",
        evidence_frame_ids=["F000001"],
        before_action_after_roles={"F000001": "context"},
        exact_visible_text_candidates=[
            {
                "text": "OK",
                "frame_id": "F000001",
                "region_xywh": (10, 5, 20, 10),
                "confidence": 0.9,
            }
        ],
        consequential_changes=[],
        confidence=0.8,
        uncertainty=[],
        statements_not_inferred=["No speech was inferred."],
    )
    descriptor = ProviderDescriptor(
        provider_id="local-fixture",
        route="local",
        model="fixture-vlm",
        model_version="revision-1",
        adapter_version="1.0",
        network_required=False,
    )
    return packet, annotation, descriptor


def test_annotation_becomes_atomic_pixel_grounded_observation(tmp_path: Path) -> None:
    packet, annotation, descriptor = _fixture(tmp_path)
    observation = annotation_to_observation(
        annotation,
        packet,  # type: ignore[arg-type]
        descriptor,
        project_root=tmp_path,
        observation_id="VA000001",
        observed_at_utc="2026-08-09T00:00:00Z",
    )
    assert observation.actor_kind == "multimodal_model"
    assert observation.image_ids == ["F000001"]
    assert [claim.claim_class for claim in observation.proposed_claims] == [
        "direct_visible",
        "exact_text",
    ]
    assert observation.proposed_claims[1].evidence_regions[0].region_xywh_normalized == (
        0.1,
        0.1,
        0.2,
        0.2,
    )
    assert observation.remaining_unknowns == ["Deliberately not inferred: No speech was inferred."]


def test_deterministic_annotation_is_attributed_as_deterministic(
    tmp_path: Path,
) -> None:
    packet, annotation, descriptor = _fixture(tmp_path)
    observation = annotation_to_observation(
        annotation,
        packet,  # type: ignore[arg-type]
        descriptor,
        project_root=tmp_path,
        observation_id="VA000001",
        deterministic=True,
    )

    assert observation.actor_kind == "deterministic"
    assert observation.analysis_depth == "deterministic"


def test_cumulative_host_observation_records_visible_prior_claim_context(
    tmp_path: Path,
) -> None:
    packet, annotation, _descriptor = _fixture(tmp_path)
    descriptor = ProviderDescriptor(
        provider_id="codex-subagent",
        route="host_agent",
        model="codex-subagent",
        model_version=None,
        adapter_version="1.0",
        network_required=False,
    )
    observation = annotation_to_observation(
        annotation,
        packet,  # type: ignore[arg-type]
        descriptor,
        project_root=tmp_path,
        observation_id="VA000002",
        prior_metadata_visible=True,
        prior_claim_context_ids=("IC000001", "IC000001", "IC000002"),
    )

    assert observation.actor_kind == "host_agent"
    assert observation.analysis_depth == "cumulative"
    assert observation.prior_metadata_visible is True
    assert observation.prior_claim_context_ids == ["IC000001", "IC000002"]


def test_annotation_region_must_be_inside_real_image(tmp_path: Path) -> None:
    packet, annotation, descriptor = _fixture(tmp_path)
    broken = annotation.model_copy(
        update={
            "exact_visible_text_candidates": [
                annotation.exact_visible_text_candidates[0].model_copy(
                    update={"region_xywh": (90, 0, 20, 10)}
                )
            ]
        }
    )
    with pytest.raises(ValidationFailure, match="outside"):
        annotation_to_observation(
            broken,
            packet,  # type: ignore[arg-type]
            descriptor,
            project_root=tmp_path,
            observation_id="VA000001",
        )
