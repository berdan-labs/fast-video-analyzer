from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from video_script_reconstructor.image_metadata import create_creation_metadata
from video_script_reconstructor.metadata_reconcile import (
    AppendOnlyViolationError,
    SemanticBatchContext,
    StaleBaseRevisionError,
    evaluate_sufficiency,
    ingest_observation,
    reconcile_observation,
    should_request_another_pass,
)
from video_script_reconstructor.schemas import (
    EvidenceQuestion,
    EvidenceRegion,
    ImageClaim,
    ProposedImageClaim,
    VisualAnalysisObservation,
)


def observation(
    observation_id: str,
    *,
    base_revision_id: str = "MR000001",
    proposed_claims: list[ProposedImageClaim] | None = None,
    depth: str = "cumulative",
    prior_visible: bool = True,
    actor_kind: str = "multimodal_model",
    actor_label: str = "provider/model@1",
    provider: str | None = "provider",
    model: str | None = "model",
) -> VisualAnalysisObservation:
    return VisualAnalysisObservation(
        observation_id=observation_id,
        image_ids=["F000001"],
        base_revision_id=base_revision_id,
        actor_kind=actor_kind,
        actor_label=actor_label,
        reviewer_name="Reviewer" if actor_kind == "human" else None,
        provider=provider,
        model=model,
        observed_at_utc="2026-01-01T00:00:00Z",
        purpose="Answer Q000001.",
        analysis_depth=depth,
        prior_metadata_visible=prior_visible,
        proposed_claims=proposed_claims or [],
        rationale="Concise pixel-grounded rationale.",
        validation_result="accepted",
    )


def proposed(statement: str, **updates: object) -> ProposedImageClaim:
    raw = {
        "claim_class": "direct_visible",
        "statement": statement,
        "evidence_regions": [{"image_id": "F000001", "whole_frame_basis": True}],
    }
    raw.update(updates)
    return ProposedImageClaim.model_validate(raw)


def test_equivalent_claims_deduplicate_without_averaging_confidence() -> None:
    existing = ImageClaim(
        claim_id="IC000001",
        claim_class="direct_visible",
        statement="The Example option is unchecked.",
        status="supported",
        confidence=0.91,
        supporting_image_ids=["F000001"],
        evidence_regions=[EvidenceRegion(image_id="F000001", whole_frame_basis=True)],
        supporting_observation_ids=["VA000001"],
    )
    second = observation(
        "VA000002",
        proposed_claims=[proposed("  The Example option is unchecked  ", confidence=0.5)],
    )
    result = reconcile_observation(
        [existing], second, revision_id="MR000002", current_revision_id="MR000001"
    )
    assert len(result.claims) == 1
    assert result.claims[0].supporting_observation_ids == ["VA000001", "VA000002"]
    assert result.claims[0].confidence == 0.91


def test_explicit_contradiction_preserves_both_claims_as_disputed() -> None:
    existing = ImageClaim(
        claim_id="IC000001",
        claim_class="exact_text",
        statement="The value is 42.",
        status="supported",
        supporting_image_ids=["F000001"],
        evidence_regions=[EvidenceRegion(image_id="F000001", whole_frame_basis=True)],
        supporting_observation_ids=["VA000001"],
    )
    conflict = observation(
        "VA000002",
        proposed_claims=[
            proposed(
                "The value is 43.",
                claim_class="exact_text",
                relationship="contradict",
                related_claim_ids=["IC000001"],
            )
        ],
    )
    result = reconcile_observation(
        [existing], conflict, revision_id="MR000002", current_revision_id="MR000001"
    )
    assert {claim.status for claim in result.claims} == {"disputed"}
    assert {claim.statement for claim in result.claims} == {"The value is 42.", "The value is 43."}
    assert result.disputed_claim_ids == ["IC000001", "IC000002"]


def test_stale_base_is_rejected_or_explicitly_reconciled() -> None:
    stale = observation(
        "VA000003", base_revision_id="MR000001", proposed_claims=[proposed("A box is visible.")]
    )
    with pytest.raises(StaleBaseRevisionError):
        reconcile_observation(
            [],
            stale,
            revision_id="MR000003",
            current_revision_id="MR000002",
            allow_stale_reconcile=False,
        )
    merged = reconcile_observation(
        [],
        stale,
        revision_id="MR000003",
        current_revision_id="MR000002",
        allow_stale_reconcile=True,
    )
    assert merged.stale_base_reconciled
    assert merged.method == "stale-base-explicit-merge-v1"


def test_ingestion_is_append_only_and_revision_is_monotonic(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    Image.new("RGB", (3, 3), "white").save(path)
    metadata = create_creation_metadata(
        path,
        image_id="F000001",
        media_id="M1234567890ABCDEF",
        origin="extracted_full_frame",
        derivation_method="fixture",
        requested_ms=0,
        actual_ms=10,
        pts_value=1,
        time_base="1/100",
        pts_source="measured",
        role="context",
        selection_reason="state",
        revision_id="MR000001",
        canonical_revision_locator=".state/vision/image-observations.json#MR000001",
        canonical_revision_digest="a" * 64,
    )
    item = observation("VA000001", proposed_claims=[proposed("A white field is visible.")])
    result = ingest_observation(metadata, [], [], item, now_utc="2026-01-01T00:00:01Z")
    assert result.revision.revision_id == "MR000002"
    assert result.metadata.analysis.revision_number == 2
    assert result.metadata.analysis.observation_history[0].observation_id == "VA000001"
    with pytest.raises(AppendOnlyViolationError):
        ingest_observation(result.metadata, result.observations, [result.revision], item)


def test_semantic_batch_context_reuses_history_indexes(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    Image.new("RGB", (3, 3), "white").save(path)
    metadata = create_creation_metadata(
        path,
        image_id="F000001",
        media_id="M1234567890ABCDEF",
        origin="extracted_full_frame",
        derivation_method="fixture",
        requested_ms=0,
        actual_ms=10,
        pts_value=1,
        time_base="1/100",
        pts_source="measured",
        role="context",
        selection_reason="state",
        revision_id="MR000001",
        canonical_revision_locator=".state/vision/image-observations.json#MR000001",
        canonical_revision_digest="a" * 64,
    )
    first = observation("VA000001", proposed_claims=[proposed("A white field is visible.")])
    first_result = ingest_observation(metadata, [], [], first)
    context = SemanticBatchContext.from_project(
        {
            "visual_observations": [
                item.model_dump(mode="json") for item in first_result.observations
            ],
            "metadata_revisions": [first_result.revision.model_dump(mode="json")],
            "image_claims": [
                claim.model_dump(mode="json")
                for claim in first_result.metadata.knowledge.claims
            ],
        }
    )
    second = observation("VA000002", base_revision_id="MR000002")
    second_result = ingest_observation(
        first_result.metadata,
        context.observations,
        context.revisions,
        second,
        batch_context=context,
    )
    assert second_result.revision.revision_id == "MR000003"
    assert context.latest_revision_for_image("F000001") is not None
    context.commit(second_result.observations[-1], second_result.revision)
    assert len(context.observations) == 2
    assert context.observations_by_id["VA000002"].ingestion_result == "accepted in MR000003"


def test_semantic_batch_context_indexes_project_relationships() -> None:
    """Batch relationship lookups are built once and retain live dictionaries."""

    frame = {"frame_id": "F000001", "full_frame_path": "evidence/full/F000001.png"}
    block = {"block_id": "B000001", "frame_ids": ["F000001"]}
    event = {"event_id": "V000001", "evidence_frame_ids": ["F000001"]}
    review = {
        "review_id": "R000001",
        "category": "high_impact_image_claim",
        "frame_ids": ["F000001"],
        "event_ids": ["V000001"],
        "image_claim_ids": ["IC000001"],
        "sufficiency_decision_ids": ["MS000001"],
    }
    payload = {"image": {"image_id": "F000001"}}
    context = SemanticBatchContext.from_project(
        {
            "frames": [frame],
            "script_blocks": [block],
            "visual_events": [event],
            "review_items": [review],
            "evidence_image_metadata": [payload],
        }
    )

    assert context.frames_by_id["F000001"] is frame
    assert context.blocks_by_frame_id["F000001"] == [block]
    assert context.events_by_frame_id["F000001"] == [event]
    assert context.reviews_by_frame_id["F000001"] == [review]
    assert context.reviews_by_event_id["V000001"] == [review]
    assert context.reviews_by_claim_id["IC000001"] == [review]
    assert context.reviews_by_decision_id["MS000001"] == [review]
    assert context.guarded_review_claim_ids == {"IC000001"}
    assert context.payload_index_by_image_id == {"F000001": 0}

    # Newly appended guarded reviews are indexed immediately for the next
    # observation in the same batch.
    new_review = {
        "review_id": "R000002",
        "category": "disputed_image_claim",
        "frame_ids": ["F000001"],
        "image_claim_ids": ["IC000002"],
    }
    context.record_reviews([new_review])
    assert context.reviews_by_claim_id["IC000002"] == [new_review]
    assert "IC000002" in context.guarded_review_claim_ids


def test_high_impact_claim_needs_meaningfully_independent_support() -> None:
    claim = ImageClaim(
        claim_id="IC000001",
        claim_class="exact_text",
        statement="The displayed value is 42.",
        status="supported",
        importance="high_impact",
        high_impact_token=True,
        supporting_image_ids=["F000001"],
        evidence_regions=[EvidenceRegion(image_id="F000001", whole_frame_basis=True)],
        supporting_observation_ids=["VA000001", "VA000002"],
    )
    first = observation("VA000001")
    repeated = observation("VA000002")
    question = EvidenceQuestion(
        question_id="Q000001",
        question="What is the exact value?",
        importance="high_impact",
        required_precision="exact token",
        modality="ocr",
        candidate_claim_ids=[claim.claim_id],
    )
    insufficient = evaluate_sufficiency(
        decision_id="MS000001",
        questions=[question],
        claims=[claim],
        observations=[first, repeated],
        image_ids=["F000001"],
        metadata_revision_ids=["MR000002"],
        unattempted_evidence_actions=["Run a blind check."],
        now_utc="2026-01-01T00:00:02Z",
    )
    assert insufficient.status == "insufficient"
    assert should_request_another_pass(
        insufficient, consecutive_passes_without_new_supported_information=0
    )
    blind = observation(
        "VA000002",
        depth="blind",
        prior_visible=False,
        actor_kind="human",
        actor_label="Reviewer",
        provider=None,
        model=None,
    )
    sufficient = evaluate_sufficiency(
        decision_id="MS000002",
        questions=[question],
        claims=[claim],
        observations=[first, blind],
        image_ids=["F000001"],
        metadata_revision_ids=["MR000002"],
        now_utc="2026-01-01T00:00:03Z",
    )
    assert sufficient.status == "sufficient"


def test_new_high_impact_claim_stays_proposed_until_independent_confirmation() -> None:
    first = observation(
        "VA000010",
        proposed_claims=[
            proposed(
                "The displayed value is 42.",
                claim_class="exact_text",
                importance="high_impact",
                high_impact_token=True,
            )
        ],
    )
    initial = reconcile_observation(
        [], first, revision_id="MR000002", current_revision_id="MR000001"
    )
    assert initial.claims[0].status == "proposed"
    repeated = observation(
        "VA000011",
        base_revision_id="MR000002",
        proposed_claims=[
            proposed(
                "The displayed value is 42.",
                claim_class="exact_text",
                importance="high_impact",
                high_impact_token=True,
            )
        ],
    )
    correlated = reconcile_observation(
        initial.claims, repeated, revision_id="MR000003", current_revision_id="MR000002"
    )
    assert correlated.claims[0].status == "proposed"
    blind = observation(
        "VA000012",
        base_revision_id="MR000003",
        depth="blind",
        prior_visible=False,
        proposed_claims=[],
    ).model_copy(update={"independently_confirmed_claim_ids": ["IC000001"]})
    confirmed = reconcile_observation(
        correlated.claims, blind, revision_id="MR000004", current_revision_id="MR000003"
    )
    assert confirmed.claims[0].status == "supported"


def test_attributable_human_can_directly_support_a_new_high_impact_claim() -> None:
    human = observation(
        "VA000020",
        actor_kind="human",
        actor_label="Ada Reviewer",
        provider=None,
        model=None,
        proposed_claims=[
            proposed(
                "The displayed value is 42.",
                claim_class="exact_text",
                importance="high_impact",
                high_impact_token=True,
            )
        ],
    )
    result = reconcile_observation(
        [], human, revision_id="MR000002", current_revision_id="MR000001"
    )
    assert result.claims[0].status == "supported"


def test_stopping_after_two_no_information_passes_never_claims_sufficiency() -> None:
    question = EvidenceQuestion(
        question_id="Q000001",
        question="What is hidden?",
        required_precision="exact",
        modality="visual",
    )
    decision = evaluate_sufficiency(
        decision_id="MS000001",
        questions=[question],
        claims=[],
        observations=[],
        image_ids=["F000001"],
        metadata_revision_ids=["MR000001"],
        unattempted_evidence_actions=["Inspect adjacent frame."],
        now_utc="2026-01-01T00:00:00Z",
    )
    assert decision.status == "insufficient"
    assert not should_request_another_pass(
        decision,
        consecutive_passes_without_new_supported_information=2,
        stop_after_no_new_supported_information_passes=2,
    )
