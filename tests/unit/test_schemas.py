from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_script_reconstructor.schemas import (
    CanonicalProject,
    EmbeddedSufficiency,
    EvidenceQuestion,
    ImageClaim,
    ImageKnowledge,
    MetadataSufficiencyDecision,
    TranscriptSegment,
)


@pytest.mark.parametrize(
    "status",
    [
        "processing",
        "blocked",
        "review_required",
        "automatically_checked",
        "human_reviewed",
        "fully_verified",
        "failed",
    ],
)
def test_canonical_project_accepts_every_contract_verification_state(status: str) -> None:
    assert CanonicalProject(project_status=status).project_status == status


def test_all_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TranscriptSegment(segment_id="T000001", raw_text="hello", misspelled=True)


def test_nullable_timing_stays_nullable_and_order_is_checked() -> None:
    segment = TranscriptSegment(segment_id="T000001", raw_text="hello")
    assert segment.start_ms is None and segment.end_ms is None
    assert segment.human_verified_text is None
    with pytest.raises(ValidationError, match="must not precede"):
        TranscriptSegment(segment_id="T000001", raw_text="hello", start_ms=20, end_ms=10)


def test_supported_direct_claim_requires_pixel_basis_and_indexes_match() -> None:
    with pytest.raises(ValidationError, match="requires pixel support"):
        ImageClaim(
            claim_id="IC000001",
            claim_class="direct_visible",
            statement="The box is checked.",
            status="supported",
        )
    claim = ImageClaim(
        claim_id="IC000001",
        claim_class="direct_visible",
        statement="The box is checked.",
        status="supported",
        supporting_image_ids=["F000001"],
        region_xywh_normalized=(0.1, 0.1, 0.2, 0.2),
        supporting_observation_ids=["VA000001"],
    )
    knowledge = ImageKnowledge(
        selection_reason="The changed state is visible.",
        claims=[claim],
        supported_claim_ids=[claim.claim_id],
    )
    assert knowledge.supported_claim_ids == ["IC000001"]
    with pytest.raises(ValidationError, match="do not match"):
        ImageKnowledge(selection_reason="bad index", claims=[claim], supported_claim_ids=[])


def test_sufficiency_schema_requires_a_complete_question_partition() -> None:
    question = EvidenceQuestion(
        question_id="Q000001",
        question="What is the value?",
        required_precision="exact token",
        modality="ocr",
    )
    with pytest.raises(ValidationError, match="every evaluated question"):
        MetadataSufficiencyDecision(
            decision_id="MS000001",
            questions=[question],
            status="insufficient",
            decided_by="test",
            decided_at_utc="2026-01-01T00:00:00Z",
            rationale="not answered",
        )
    embedded = EmbeddedSufficiency(status="semantic_observer_unavailable")
    assert embedded.status == "semantic_observer_unavailable"
