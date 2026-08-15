"""Canonical, schema-validated records used by the reconstruction pipeline.

Every model deliberately rejects unknown fields.  This makes adapter drift visible
at ingestion time instead of allowing misspelled evidence fields to disappear.
Timeline values are integer milliseconds; unavailable timing remains ``None``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base for persisted contracts: unknown input is always an error."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TimeInterval(StrictModel):
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def ordered(self) -> TimeInterval:
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self


class MediaStream(StrictModel):
    index: int = Field(ge=0)
    codec: str | None = None
    language: str | None = None
    disposition: dict[str, bool] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)


class MediaChapter(TimeInterval):
    chapter_id: str
    title: str | None = None


class MediaIdentity(StrictModel):
    schema_version: str = "1.0"
    media_id: str
    original_source_reference: str
    local_preserved_reference: str | None = None
    content_hash: str
    byte_size: int = Field(ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    container: str | None = None
    video_streams: list[MediaStream] = Field(default_factory=list)
    audio_streams: list[MediaStream] = Field(default_factory=list)
    subtitle_streams: list[MediaStream] = Field(default_factory=list)
    frame_rate: str | None = None
    average_frame_rate: str | None = None
    time_base: str | None = None
    variable_frame_rate: bool | None = None
    resolution: tuple[int, int] | None = None
    sample_aspect_ratio: str | None = None
    rotation: int | None = None
    chapters: list[MediaChapter] = Field(default_factory=list)
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    acquisition_provenance: dict[str, Any] = Field(default_factory=dict)


class TranscriptWord(StrictModel):
    word_id: str
    text: str
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None
    language: str | None = None
    uncertainty_flags: list[str] = Field(default_factory=list)


class TranscriptSegment(StrictModel):
    segment_id: str
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    timing_provenance: str | None = None
    raw_text: str
    normalized_text: str | None = None
    repaired_text: str | None = None
    human_verified_text: str | None = None
    speaker_label: str | None = None
    language: str | None = None
    words: list[TranscriptWord] = Field(default_factory=list)
    source_candidate_id: str | None = None
    source_track: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    verification_status: str = "unverified"
    repair_record_ids: list[str] = Field(default_factory=list)
    uncertainty_items: list[str] = Field(default_factory=list)
    substantive: bool = True

    @model_validator(mode="after")
    def ordered(self) -> TranscriptSegment:
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("segment end_ms must not precede start_ms")
        return self


class TranscriptSourceCandidate(StrictModel):
    candidate_id: str
    source_type: str
    origin: str
    language: str | None = None
    authorship: Literal["human", "auto_generated", "unknown"] = "unknown"
    human_authored: bool | None = None
    auto_generated: bool | None = None
    raw_preservation_path: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    quality_metrics: dict[str, float | int | bool | None] = Field(default_factory=dict)
    reliable_intervals: list[TimeInterval] = Field(default_factory=list)
    unreliable_intervals: list[TimeInterval] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    selection_score: float | None = None
    selected_intervals: list[TimeInterval] = Field(default_factory=list)
    decision_rationale: str | None = None


class RepairRecord(StrictModel):
    record_id: str
    source_segment_ids: list[str]
    before_text: str
    candidate_after_text: str | None = None
    action: Literal["retain", "replace", "insert", "split", "merge", "unresolved"]
    audio_interval: TimeInterval | None = None
    context_padding_ms: int = Field(default=0, ge=0)
    asr_model: str | None = None
    asr_settings: dict[str, Any] = Field(default_factory=dict)
    alignment_evidence: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str
    alternatives: list[str] = Field(default_factory=list)
    actor: str
    created_at_utc: str


class PTSReference(StrictModel):
    value: int | None = None
    time_base: str | None = None
    source: str


class PixelHash(StrictModel):
    algorithm: Literal["sha256-rgba8-srgb-v1"] = "sha256-rgba8-srgb-v1"
    value: str = Field(pattern=r"^[0-9a-f]{64}$")


class ImageDerivation(StrictModel):
    method: str
    transformation_ids: list[str] = Field(default_factory=list)


class CropRecord(StrictModel):
    crop_id: str
    parent_full_frame_id: str
    crop_xywh: tuple[int, int, int, int]
    path: str
    reason: str
    method: str


class FrameObservation(StrictModel):
    frame_id: str
    requested_ms: int = Field(ge=0)
    actual_ms: int = Field(ge=0)
    pts: int | None = None
    time_base: str | None = None
    frame_index: int | None = Field(default=None, ge=0)
    offset_ms: int | None = None
    timestamp_source: str | None = None
    timing_estimated: bool = False
    full_frame_path: str
    parent_full_frame_id: str | None = None
    crop_xywh: tuple[int, int, int, int] | None = None
    crops: list[CropRecord] = Field(default_factory=list)
    scene_id: str | None = None
    quality_scores: dict[str, float] = Field(default_factory=dict)
    perceptual_hashes: dict[str, str] = Field(default_factory=dict)
    region_hashes: dict[str, str] = Field(default_factory=dict)
    pixel_hash: PixelHash
    file_hash: str | None = None
    metadata_payload_digest: str | None = None
    latest_revision_id: str | None = None
    metadata_sufficiency_state: str | None = None
    ocr_observation_ids: list[str] = Field(default_factory=list)
    selection_reason: str
    evidence_role: str
    linked_event_ids: list[str] = Field(default_factory=list)
    linked_block_ids: list[str] = Field(default_factory=list)
    verification_status: str = "unverified"
    # Canonical renderer conveniences remain explicit and must agree with the
    # authoritative fields above during audits.
    path: str | None = None
    final: bool = False
    description: str | None = None
    supported_claim_ids: list[str] = Field(default_factory=list)
    disputed_claim_ids: list[str] = Field(default_factory=list)
    unresolved_claim_ids: list[str] = Field(default_factory=list)
    metadata: EvidenceImageMetadata | None = None

    @model_validator(mode="after")
    def offset_matches(self) -> FrameObservation:
        measured = self.actual_ms - self.requested_ms
        if self.offset_ms is None:
            self.offset_ms = measured
        elif self.offset_ms != measured:
            raise ValueError("offset_ms must equal actual_ms - requested_ms")
        return self


Snapshot = FrameObservation


ClaimClass = Literal[
    "direct_visible",
    "exact_text",
    "temporal_change",
    "cross_modal_corroboration",
    "contextual_inference",
    "absence",
    "unresolved",
]
ClaimStatus = Literal["proposed", "supported", "disputed", "rejected", "superseded", "unresolved"]
ClaimImportance = Literal["incidental", "supporting", "consequential", "high_impact"]


class EvidenceRegion(StrictModel):
    image_id: str
    region_xywh_normalized: tuple[float, float, float, float] | None = None
    whole_frame_basis: bool = False

    @model_validator(mode="after")
    def valid_basis(self) -> EvidenceRegion:
        if self.region_xywh_normalized is None and not self.whole_frame_basis:
            raise ValueError("a claim requires a normalized region or whole-frame basis")
        if self.region_xywh_normalized is not None:
            x, y, width, height = self.region_xywh_normalized
            if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
                raise ValueError("normalized region values must be between 0 and 1")
            if width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
                raise ValueError("normalized region is outside the image")
        return self


class ImageClaim(StrictModel):
    claim_id: str
    claim_class: ClaimClass
    statement: str = Field(min_length=1)
    normalized_value: str | None = None
    status: ClaimStatus = "proposed"
    importance: ClaimImportance = "supporting"
    high_impact_token: bool = False
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_basis: str | None = None
    supporting_image_ids: list[str] = Field(default_factory=list)
    region_xywh_normalized: tuple[float, float, float, float] | None = None
    evidence_regions: list[EvidenceRegion] = Field(default_factory=list)
    ocr_observation_ids: list[str] = Field(default_factory=list)
    adjacent_frame_relationship: str | None = None
    supporting_observation_ids: list[str] = Field(default_factory=list)
    contradicting_observation_ids: list[str] = Field(default_factory=list)
    first_seen_revision_id: str | None = None
    last_updated_revision_id: str | None = None
    uncertainty: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    supersedes_claim_ids: list[str] = Field(default_factory=list)
    superseded_by_claim_id: str | None = None
    consumed_by_block_statement_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def region_bounds(self) -> ImageClaim:
        if self.region_xywh_normalized is not None:
            x, y, width, height = self.region_xywh_normalized
            if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
                raise ValueError("normalized region values must be between 0 and 1")
            if width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
                raise ValueError("normalized region is outside the image")
        if self.claim_class == "direct_visible" and self.status == "supported":
            if self.region_xywh_normalized is None and not self.evidence_regions:
                raise ValueError("a supported direct-visible claim requires pixel support")
        return self


class ProposedImageClaim(StrictModel):
    claim_id: str | None = None
    claim_class: ClaimClass
    statement: str = Field(min_length=1)
    normalized_value: str | None = None
    importance: ClaimImportance = "supporting"
    high_impact_token: bool = False
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_basis: str | None = None
    evidence_regions: list[EvidenceRegion]
    ocr_observation_ids: list[str] = Field(default_factory=list)
    relationship: Literal["new", "confirm", "narrow", "contradict", "reject", "unresolved"] = "new"
    related_claim_ids: list[str] = Field(default_factory=list)
    uncertainty: str | None = None
    alternatives: list[str] = Field(default_factory=list)


class VisualAnalysisObservation(StrictModel):
    observation_id: str
    image_ids: list[str]
    context_image_ids: list[str] = Field(default_factory=list)
    base_revision_id: str | None = None
    actor_kind: Literal["deterministic", "multimodal_model", "host_agent", "human"]
    actor_label: str
    reviewer_name: str | None = None
    provider: str | None = None
    model: str | None = None
    model_version: str | None = None
    adapter_version: str | None = None
    prompt_template_hash: str | None = None
    observed_at_utc: str
    purpose: str
    targeted_question_ids: list[str] = Field(default_factory=list)
    analysis_depth: Literal["creation", "deterministic", "cumulative", "blind"] = "cumulative"
    prior_metadata_visible: bool
    ocr_context_ids: list[str] = Field(default_factory=list)
    transcript_context_ids: list[str] = Field(default_factory=list)
    audio_context: list[TimeInterval] = Field(default_factory=list)
    event_context_ids: list[str] = Field(default_factory=list)
    prior_claim_context_ids: list[str] = Field(default_factory=list)
    proposed_claims: list[ProposedImageClaim] = Field(default_factory=list)
    independently_confirmed_claim_ids: list[str] = Field(default_factory=list)
    narrowed_claim_ids: list[str] = Field(default_factory=list)
    contradicted_claim_ids: list[str] = Field(default_factory=list)
    rejected_claim_ids: list[str] = Field(default_factory=list)
    new_supported_information: list[str] = Field(default_factory=list)
    remaining_unknowns: list[str] = Field(default_factory=list)
    suggested_next_action: str | None = None
    rationale: str
    validation_result: Literal["accepted", "rejected", "pending"] = "pending"
    ingestion_result: str | None = None

    @model_validator(mode="after")
    def human_attribution(self) -> VisualAnalysisObservation:
        if self.actor_kind == "human" and not self.reviewer_name:
            raise ValueError("human observations require reviewer_name")
        if self.analysis_depth == "blind" and self.prior_metadata_visible:
            raise ValueError("a blind observation cannot see prior metadata")
        return self


class BeforeActionAfter(StrictModel):
    group_id: str
    before_image_id: str | None = None
    action_image_ids: list[str] = Field(default_factory=list)
    after_image_ids: list[str] = Field(default_factory=list)
    supported_change_claim_ids: list[str] = Field(default_factory=list)


class ImageIdentity(StrictModel):
    image_id: str
    media_id: str
    parent_full_frame_id: str | None = None
    origin: Literal["extracted_full_frame", "derived_crop", "candidate", "diagnostic_overlay"]
    derivation: ImageDerivation
    requested_ms: int = Field(ge=0)
    actual_ms: int = Field(ge=0)
    pts: PTSReference
    role: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    orientation: int = 0
    crop_xywh: tuple[int, int, int, int] | None = None
    pixel_hash: PixelHash

    @model_validator(mode="after")
    def crop_relationship(self) -> ImageIdentity:
        if self.origin == "derived_crop":
            if self.parent_full_frame_id is None or self.crop_xywh is None:
                raise ValueError("derived crops require a full-frame parent and crop_xywh")
        elif self.crop_xywh is not None:
            raise ValueError("only derived crops may carry crop_xywh")
        if self.crop_xywh is not None:
            x, y, width, height = self.crop_xywh
            if min(x, y) < 0 or width <= 0 or height <= 0:
                raise ValueError("invalid crop_xywh")
        return self


class ImageLinks(StrictModel):
    chapter_ids: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    visual_event_ids: list[str] = Field(default_factory=list)
    ocr_observation_ids: list[str] = Field(default_factory=list)
    review_item_ids: list[str] = Field(default_factory=list)
    neighbor_image_ids: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)


class ImageKnowledge(StrictModel):
    selection_reason: str
    why_it_matters: str | None = None
    current_factual_description: str | None = None
    claims: list[ImageClaim] = Field(default_factory=list)
    supported_claim_ids: list[str] = Field(default_factory=list)
    disputed_claim_ids: list[str] = Field(default_factory=list)
    rejected_claim_ids: list[str] = Field(default_factory=list)
    unresolved_claim_ids: list[str] = Field(default_factory=list)
    explicit_unknowns: list[str] = Field(default_factory=list)
    statements_not_inferred: list[str] = Field(default_factory=list)
    before_action_after: BeforeActionAfter | None = None

    @model_validator(mode="after")
    def claim_indexes_match(self) -> ImageKnowledge:
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim IDs must be unique")
        expected = {
            "supported": {c.claim_id for c in self.claims if c.status == "supported"},
            "disputed": {c.claim_id for c in self.claims if c.status == "disputed"},
            "rejected": {c.claim_id for c in self.claims if c.status in {"rejected", "superseded"}},
            "unresolved": {
                c.claim_id for c in self.claims if c.status in {"proposed", "unresolved"}
            },
        }
        actual = {
            "supported": set(self.supported_claim_ids),
            "disputed": set(self.disputed_claim_ids),
            "rejected": set(self.rejected_claim_ids),
            "unresolved": set(self.unresolved_claim_ids),
        }
        for status, status_ids in actual.items():
            if status_ids != expected[status]:
                raise ValueError(f"{status}_claim_ids do not match current claims")
        return self


class EmbeddedSufficiency(StrictModel):
    status: Literal[
        "sufficient",
        "insufficient",
        "no_further_evidence",
        "limit_reached",
        "semantic_observer_unavailable",
    ]
    evaluated_question_ids: list[str] = Field(default_factory=list)
    answered_question_ids: list[str] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)
    recommended_next_action: str | None = None


class ObservationHistoryEntry(StrictModel):
    observation_id: str
    actor_kind: str
    actor_label: str
    observed_at_utc: str
    prior_metadata_visible: bool
    purpose: str
    outcome: str


class DeterministicDifferenceRegion(StrictModel):
    neighbor_image_id: str
    xywh: tuple[int, int, int, int]
    changed_ratio: float = Field(ge=0.0, le=1.0)
    mean_difference: float = Field(ge=0.0, le=1.0)


class ImageAnalysis(StrictModel):
    enrichment_level: Literal["creation", "deterministic", "semantic"]
    semantic_status: Literal["unobserved", "deterministic_only", "observed", "review_required"]
    sufficiency: EmbeddedSufficiency
    latest_revision_id: str
    revision_number: int = Field(ge=1)
    observation_history: list[ObservationHistoryEntry] = Field(default_factory=list)
    frame_quality: dict[str, float] = Field(default_factory=dict)
    scene_relationships: list[str] = Field(default_factory=list)
    difference_regions: list[DeterministicDifferenceRegion] = Field(default_factory=list)
    ocr_observation_ids: list[str] = Field(default_factory=list)
    neighbor_image_ids: list[str] = Field(default_factory=list)
    before_action_after_membership: str | None = None


class ImageIntegrity(StrictModel):
    previous_revision_id: str | None = None
    previous_payload_digest: str | None = None
    payload_digest_algorithm: Literal["sha256-canonical-json-with-digest-omitted-v1"] = (
        "sha256-canonical-json-with-digest-omitted-v1"
    )
    payload_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_revision_locator: str
    canonical_revision_digest: str


class EvidenceImageMetadata(StrictModel):
    schema_name: Literal["video-script-reconstructor.evidence-image"] = (
        "video-script-reconstructor.evidence-image"
    )
    schema_version: Literal["1.0"] = "1.0"
    image: ImageIdentity
    links: ImageLinks = Field(default_factory=ImageLinks)
    knowledge: ImageKnowledge
    analysis: ImageAnalysis
    integrity: ImageIntegrity


class ImageMetadataRevision(StrictModel):
    revision_id: str
    revision_number: int = Field(ge=1)
    image_id: str
    base_revision_id: str | None = None
    observation_ids: list[str] = Field(default_factory=list)
    added_claim_ids: list[str] = Field(default_factory=list)
    confirmed_claim_ids: list[str] = Field(default_factory=list)
    narrowed_claim_ids: list[str] = Field(default_factory=list)
    disputed_claim_ids: list[str] = Field(default_factory=list)
    rejected_claim_ids: list[str] = Field(default_factory=list)
    superseded_claim_ids: list[str] = Field(default_factory=list)
    unresolved_claim_ids: list[str] = Field(default_factory=list)
    previous_payload_digest: str | None = None
    new_payload_digest: str
    reconciliation_method: str
    actor: str
    stale_base_reconciled: bool = False
    pixel_invariance_verified: bool
    embedded_write_verified: bool
    read_back_verified: bool
    canonical_mirror_committed: bool
    created_at_utc: str


class EvidenceQuestion(StrictModel):
    question_id: str
    question: str
    importance: ClaimImportance = "supporting"
    required_precision: str
    modality: Literal["visual", "ocr", "audio", "transcript", "temporal", "cross_modal"]
    candidate_claim_ids: list[str] = Field(default_factory=list)


class MetadataSufficiencyDecision(StrictModel):
    decision_id: str
    image_ids: list[str] = Field(default_factory=list)
    visual_event_ids: list[str] = Field(default_factory=list)
    script_block_ids: list[str] = Field(default_factory=list)
    metadata_revision_ids: list[str] = Field(default_factory=list)
    questions: list[EvidenceQuestion]
    answered_question_ids: list[str] = Field(default_factory=list)
    supporting_claim_ids: list[str] = Field(default_factory=list)
    unanswered_question_ids: list[str] = Field(default_factory=list)
    exact_gaps: list[str] = Field(default_factory=list)
    unattempted_evidence_actions: list[str] = Field(default_factory=list)
    status: Literal[
        "sufficient",
        "insufficient",
        "no_further_evidence",
        "limit_reached",
        "semantic_observer_unavailable",
    ]
    recommended_next_action: str | None = None
    decided_by: str
    decided_at_utc: str
    rationale: str

    @model_validator(mode="after")
    def question_partition(self) -> MetadataSufficiencyDecision:
        question_ids = {question.question_id for question in self.questions}
        answered = set(self.answered_question_ids)
        unanswered = set(self.unanswered_question_ids)
        if answered & unanswered:
            raise ValueError("a question cannot be both answered and unanswered")
        if answered | unanswered != question_ids:
            raise ValueError("every evaluated question must be answered or unanswered")
        if self.status == "sufficient" and unanswered:
            raise ValueError("a sufficient decision cannot have unanswered questions")
        return self


class OCRAlternative(StrictModel):
    token_or_character: str
    alternatives: list[str]


class OCRObservation(StrictModel):
    observation_id: str
    frame_id: str
    crop_id: str | None = None
    bounding_region: tuple[float, float, float, float] | None = None
    raw_engine_text: str
    normalized_interpretation: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    alternatives: list[OCRAlternative] = Field(default_factory=list)
    language: str | None = None
    uncertain_characters: list[str] = Field(default_factory=list)
    engine: str
    engine_version: str | None = None
    human_decision: str | None = None


class VisualEvent(TimeInterval):
    event_id: str
    event_type: str
    scene_or_state_id: str | None = None
    evidence_frame_ids: list[str] = Field(default_factory=list)
    before_action_after_roles: dict[str, list[str]] = Field(default_factory=dict)
    ocr_observation_ids: list[str] = Field(default_factory=list)
    factual_grounded_description: str
    importance: ClaimImportance = "supporting"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: list[str] = Field(default_factory=list)
    annotation_provider: str | None = None
    review_status: str = "unverified"
    image_claim_ids: list[str] = Field(default_factory=list)
    metadata_revision_ids: list[str] = Field(default_factory=list)


class ScriptBlock(TimeInterval):
    block_id: str
    chapter_id: str
    speaker: str | None = None
    spoken_text: str = ""
    visual_description: str | None = None
    on_screen_text: list[str] = Field(default_factory=list)
    relevant_non_speech_audio: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    transcript_segment_ids: list[str] = Field(default_factory=list)
    visual_event_ids: list[str] = Field(default_factory=list)
    image_claim_ids: list[str] = Field(default_factory=list)
    metadata_revision_ids: list[str] = Field(default_factory=list)
    metadata_sufficiency_decision_ids: list[str] = Field(default_factory=list)
    transformation_ids: list[str] = Field(default_factory=list)
    fidelity_mode: Literal["verbatim", "clean-verbatim", "production-script"] = "verbatim"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    verification_status: str = "unverified"
    uncertainty: list[str] = Field(default_factory=list)
    residual_source_text: str | None = None


class Chapter(TimeInterval):
    chapter_id: str
    title: str
    block_ids: list[str] = Field(default_factory=list)
    source_authored: bool = False


class ReviewItem(TimeInterval):
    review_id: str
    severity: Literal["low", "medium", "high", "critical"]
    category: str
    block_ids: list[str] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    ocr_observation_ids: list[str] = Field(default_factory=list)
    image_claim_ids: list[str] = Field(default_factory=list)
    metadata_revision_ids: list[str] = Field(default_factory=list)
    sufficiency_decision_ids: list[str] = Field(default_factory=list)
    problem: str
    alternatives: list[str] = Field(default_factory=list)
    required_action: str
    blocking: bool
    decision: str | None = None
    replacement: str | None = None
    reviewer: str | None = None
    decision_timestamp_utc: str | None = None
    rationale: str | None = None


class StageRecord(StrictModel):
    name: str
    status: Literal["pending", "running", "completed", "failed", "blocked", "skipped"]
    started_at_utc: str | None = None
    ended_at_utc: str | None = None
    elapsed_ms: int | None = Field(default=None, ge=0)
    detail: str | None = None


class RunManifest(StrictModel):
    schema_version: str = "1.0"
    run_id: str
    input_identity: dict[str, Any]
    source_hashes: dict[str, str] = Field(default_factory=dict)
    source_config_hash: str | None = None
    configuration_hash: str
    code_version: str
    runtime: dict[str, Any] = Field(default_factory=dict)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    commands: list[list[str]] = Field(default_factory=list)
    exact_commands: list[list[str]] = Field(default_factory=list)
    stages: list[StageRecord] | dict[str, Any] = Field(default_factory=list)
    stage_records: list[StageRecord] = Field(default_factory=list)
    cache_keys: dict[str, str] = Field(default_factory=dict)
    checkpoints: list[str] = Field(default_factory=list)
    network_activity: list[dict[str, Any]] = Field(default_factory=list)
    provider_usage: list[dict[str, Any]] = Field(default_factory=list)
    degradations: list[str] = Field(default_factory=list)
    performance: dict[str, Any] = Field(default_factory=dict)
    generated_artifacts: list[str] = Field(default_factory=list)
    reproducibility: dict[str, Any] = Field(default_factory=dict)
    written_at_utc: str | None = None
    run_cache_key: str | None = None


class CoverageCounts(StrictModel):
    total: int = Field(ge=0)
    covered: int = Field(ge=0)
    missing_ids: list[str] = Field(default_factory=list)
    partial_ids: list[str] = Field(default_factory=list)
    duplicate_ids: list[str] = Field(default_factory=list)


class OrderedMeaningCoverage(StrictModel):
    total_segments: int = Field(ge=0)
    exact_segments: int = Field(ge=0)


class VisualEvidenceCoverage(StrictModel):
    total_final_frames: int = Field(ge=0)
    used_frames: int = Field(ge=0)
    total_generated_images: int = Field(default=0, ge=0)
    embedded_metadata_images: int = Field(default=0, ge=0)
    semantically_analyzed_images: int = Field(default=0, ge=0)
    markdown_consumed_images: int = Field(default=0, ge=0)


class AuditReport(StrictModel):
    schema_version: str = "1.0"
    source_segment_coverage: float | CoverageCounts
    ordered_token_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    meaning_unit_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    ordered_meaning_coverage: OrderedMeaningCoverage | None = None
    missing_segment_ids: list[str] = Field(default_factory=list)
    partial_segment_ids: list[str] = Field(default_factory=list)
    duplicate_output_ids: list[str] = Field(default_factory=list)
    residual_text_items: list[str] = Field(default_factory=list)
    unsupported_spoken_statements: list[str] = Field(default_factory=list)
    unsupported_visual_statements: list[str] = Field(default_factory=list)
    high_impact_token_discrepancies: list[dict[str, Any]] | Literal["[REDACTED]"] = Field(
        default_factory=list
    )
    timeline_errors: list[str] = Field(default_factory=list)
    visual_event_coverage: float | CoverageCounts | None = Field(default=None)
    visual_evidence: VisualEvidenceCoverage | None = None
    image_metadata_coverage: dict[str, int] = Field(default_factory=dict)
    screenshot_checks: list[str] = Field(default_factory=list)
    image_metadata_checks: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    stale_claim_ids: list[str] = Field(default_factory=list)
    disputed_claim_ids: list[str] = Field(default_factory=list)
    unresolved_claim_ids: list[str] = Field(default_factory=list)
    guarded_unresolved_claim_ids: list[str] = Field(default_factory=list)
    ocr_uncertainty: list[str] = Field(default_factory=list)
    anchor_navigation_checks: list[str] = Field(default_factory=list)
    output_contract_checks: list[str] = Field(default_factory=list)
    unresolved_review_item_ids: list[str] = Field(default_factory=list)
    unresolved_review_items: list[str] = Field(default_factory=list)
    blocking_failures: list[str] = Field(default_factory=list)
    final_project_status: Literal[
        "processing",
        "blocked",
        "review_required",
        "automatically_checked",
        "human_reviewed",
        "fully_verified",
        "failed",
    ]


class CanonicalProject(StrictModel):
    schema_version: str = "1.0"
    source_title: str = "Untitled source"
    project_status: Literal[
        "processing",
        "blocked",
        "review_required",
        "automatically_checked",
        "human_reviewed",
        "fully_verified",
        "failed",
    ] = "review_required"
    status_reason: str | None = None
    generated_at_utc: str | None = None
    fidelity_mode: Literal["verbatim", "clean-verbatim", "production-script"] = "verbatim"
    primary_language: str | None = None
    visual_source_available: bool = True
    transcript_source_decision: str | dict[str, Any] = Field(default_factory=dict)
    input_reference: str | None = None
    code_version: str | None = None
    config_hash: str | None = None
    tools_models_summary: str | None = None
    manifest: RunManifest | dict[str, Any] | None = None
    media: MediaIdentity | None = None
    chapters: list[Chapter] = Field(default_factory=list)
    transcript_candidates: list[TranscriptSourceCandidate] = Field(default_factory=list)
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    repairs: list[RepairRecord] = Field(default_factory=list)
    frames: list[FrameObservation] = Field(default_factory=list)
    ocr_observations: list[OCRObservation] = Field(default_factory=list)
    visual_observations: list[VisualAnalysisObservation] = Field(default_factory=list)
    image_claims: list[ImageClaim] = Field(default_factory=list)
    evidence_image_metadata: list[EvidenceImageMetadata] = Field(default_factory=list)
    metadata_revisions: list[ImageMetadataRevision] = Field(default_factory=list)
    sufficiency_decisions: list[MetadataSufficiencyDecision] = Field(default_factory=list)
    visual_events: list[VisualEvent] = Field(default_factory=list)
    script_blocks: list[ScriptBlock] = Field(default_factory=list)
    review_items: list[ReviewItem] = Field(default_factory=list)
    corrections: list[dict[str, Any]] = Field(default_factory=list)
    final_signoffs: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    state_metadata: dict[str, Any] = Field(default_factory=dict)
    audit: AuditReport | None = None
    state_transitions: list[dict[str, Any]] = Field(default_factory=list)


__all__ = [
    name
    for name, value in globals().copy().items()
    if isinstance(value, type) and issubclass(value, BaseModel)
]
