from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from .errors import BlockedError, InputError
from .frame_quality import (
    FrameQuality,
    assess_frame_quality,
    deduplication_decision,
    normalize_ocr_for_comparison,
    perceptual_dhash,
    perceptual_hamming,
)


@dataclass(frozen=True)
class FrameCandidate:
    frame_id: str
    path: Path
    actual_ms: int
    requested_ms: int | None = None
    relevance: float = 0.0
    importance: float = 0.0
    temporal_proximity: float = 0.0
    stability: float = 0.5
    novelty: float = 0.5
    ocr_readability: float = 0.0
    full_state_completeness: float = 0.5
    transition_risk: float | None = None
    evidence_role: str = "context"
    ocr_text: str | None = None
    consequential_change: bool = False
    mandatory: bool = False
    quality: FrameQuality | None = None
    pixel_hash: str | None = None
    # ``analyze_frame_sequence_with_hash`` already computes a dHash while the
    # PNG is decoded for quality/difference analysis.  Carrying it forward
    # avoids reopening every evidence PNG during selection.  ``None`` keeps
    # the public helper backwards-compatible for callers that construct
    # candidates independently.
    perceptual_hash: str | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ScoredFrame:
    candidate: FrameCandidate
    score: float
    selected_reason: str
    # IDs covered by this representative after low-importance deduplication.
    # Keeping this on the scored record makes the coverage available to callers
    # without requiring them to join a second ledger themselves.
    covered_frame_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FrameSelectionProvenance:
    """Auditable disposition for one input frame.

    A duplicate always points at the selected representative that covered it;
    a low-score frame has no representative because it was rejected by the
    duration-aware budget.  Similarity measurements are optional for selected
    and budget-rejected records, but are populated for duplicate decisions.
    """

    frame_id: str
    status: Literal["selected", "duplicate", "low_score"]
    reason: str
    importance: float
    importance_tier: str
    score: float
    representative_frame_id: str | None = None
    time_delta_ms: int | None = None
    perceptual_hamming: int | None = None
    changed_pixel_ratio: float | None = None
    ocr_changed: bool = False
    protected_reasons: tuple[str, ...] = field(default_factory=tuple)
    covered_frame_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def representative_id(self) -> str | None:
        """Short alias for integrations that use generic cluster ledgers."""

        return self.representative_frame_id

    @property
    def cluster_frame_ids(self) -> tuple[str, ...]:
        """Representative plus all low-importance frames it covers."""

        return self.covered_frame_ids


# A descriptive alias is useful to downstream callers that only care about
# deduplication receipts.  It intentionally remains the same immutable record
# so old code can continue to consume ``FrameSelectionProvenance``.
FrameDeduplicationRecord = FrameSelectionProvenance


@dataclass(frozen=True)
class FrameSelectionResult:
    selected: tuple[ScoredFrame, ...]
    duplicate_frame_ids: tuple[str, ...]
    low_score_frame_ids: tuple[str, ...]
    target_budget: int
    review_required: bool
    review_reason: str | None
    # New audit fields are appended with defaults so positional construction by
    # existing integrations remains valid.
    provenance: tuple[FrameSelectionProvenance, ...] = field(default_factory=tuple)
    coverage: Mapping[str, Any] = field(default_factory=dict)

    @property
    def selection_provenance(self) -> tuple[FrameSelectionProvenance, ...]:
        """Backward/forward-friendly name for the immutable audit ledger."""

        return self.provenance

    @property
    def provenance_by_frame_id(self) -> dict[str, FrameSelectionProvenance]:
        """Convenient lookup view for callers joining frame records."""

        return {item.frame_id: item for item in self.provenance}

    @property
    def coverage_report(self) -> Mapping[str, Any]:
        """Alias used by audit/reporting callers."""

        return self.coverage

    @property
    def audit(self) -> dict[str, Any]:
        """Return JSON-friendly selection audit data without mutating state."""

        return {
            "coverage": dict(self.coverage),
            "provenance": [
                {
                    "frame_id": item.frame_id,
                    "status": item.status,
                    "reason": item.reason,
                    "importance": item.importance,
                    "importance_tier": item.importance_tier,
                    "score": item.score,
                    "representative_frame_id": item.representative_frame_id,
                    "time_delta_ms": item.time_delta_ms,
                    "perceptual_hamming": item.perceptual_hamming,
                    "changed_pixel_ratio": item.changed_pixel_ratio,
                    "ocr_changed": item.ocr_changed,
                    "protected_reasons": list(item.protected_reasons),
                    "covered_frame_ids": list(item.covered_frame_ids),
                }
                for item in self.provenance
            ],
        }


# Importance bands deliberately match the safety-limit threshold used by the
# legacy selector.  Very-high frames are consequential transitions in the
# persisted corpus; high frames are the remaining protected change candidates.
HIGH_IMPORTANCE_THRESHOLD = 0.75
VERY_HIGH_IMPORTANCE_THRESHOLD = 0.90

# Compression noise and timestamp-adjacent resampling can make two static
# states differ slightly more than the strict duplicate test allows.  These
# bounds are used only for *low-importance* candidates after all semantic and
# boundary protections have been checked.  They are intentionally much tighter
# than a general visual-similarity threshold.
NEAR_DUPLICATE_HAMMING_THRESHOLD = 8
NEAR_DUPLICATE_CHANGED_RATIO = 0.05
NEAR_DUPLICATE_MEAN_DIFFERENCE = 0.03

_SEQUENCE_ROLES = {"before", "action", "after", "result"}
_BOUNDARY_REASON_TOKENS = {
    "chapter",
    "deictic",
    "speech",
    "subtitle",
    "transcript",
    "boundary",
    "meaning",
}


def importance_tier(
    importance: float,
    *,
    high_threshold: float = HIGH_IMPORTANCE_THRESHOLD,
    very_high_threshold: float = VERY_HIGH_IMPORTANCE_THRESHOLD,
) -> str:
    """Return the stable audit label for a unit-interval importance value."""

    _unit(importance, "importance")
    _unit(high_threshold, "high_threshold")
    _unit(very_high_threshold, "very_high_threshold")
    if very_high_threshold < high_threshold:
        raise InputError("very_high_threshold cannot be lower than high_threshold")
    if importance >= very_high_threshold:
        return "very_high"
    if importance >= high_threshold:
        return "high"
    if importance >= 0.45:
        return "supporting"
    return "low"


def _reason_tokens(candidate: FrameCandidate) -> set[str]:
    return {
        token.strip().casefold().replace("-", "_")
        for reason in candidate.reasons
        for token in str(reason).replace("-", "_").split()
        if token.strip()
    }


def _has_transcript_boundary(candidate: FrameCandidate) -> bool:
    """Detect explicit timing/context boundaries without guessing semantics."""

    reasons = " ".join(str(reason).casefold().replace("-", "_") for reason in candidate.reasons)
    if any(token in reasons for token in _BOUNDARY_REASON_TOKENS):
        return True
    return bool(
        {
            "deictic_speech_reference",
            "chapter_boundary",
            "transcript_change",
            "subtitle_boundary",
            "ocr_change",
        }
        & _reason_tokens(candidate)
    )


def _dedup_protected_reasons(
    candidate: FrameCandidate,
    *,
    high_threshold: float,
    very_high_threshold: float,
    preserve_periodic: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if candidate.mandatory:
        reasons.append("mandatory")
    if candidate.consequential_change:
        reasons.append("consequential_change")
    if candidate.importance >= very_high_threshold:
        reasons.append("very_high_importance")
    elif candidate.importance >= high_threshold:
        reasons.append("high_importance")
    if candidate.evidence_role in _SEQUENCE_ROLES:
        reasons.append("before_action_after_role")
    if _has_transcript_boundary(candidate):
        reasons.append("transcript_or_context_boundary")
    if preserve_periodic and any(
        "periodic_safety" in str(reason).casefold().replace("-", "_")
        for reason in candidate.reasons
    ):
        reasons.append("periodic_coverage")
    return tuple(dict.fromkeys(reasons))


def _low_importance_near_duplicate(
    decision: Any,
    *,
    candidate: FrameCandidate,
    prior: FrameCandidate,
    high_threshold: float,
    very_high_threshold: float,
    preserve_periodic: bool,
) -> bool:
    """Apply a conservative relaxed duplicate test for repetitive context.

    ``deduplication_decision`` remains authoritative for OCR/role/semantic
    changes.  The relaxed path only ignores its localized-pixel protection when
    both frames are low/supporting context and the measured change is bounded by
    all near-duplicate thresholds.  This avoids turning one-character/OCR
    changes into duplicates while removing codec jitter from long static runs.
    """

    if _dedup_protected_reasons(
        candidate,
        high_threshold=high_threshold,
        very_high_threshold=very_high_threshold,
        preserve_periodic=preserve_periodic,
    ):
        return False
    if _dedup_protected_reasons(
        prior,
        high_threshold=high_threshold,
        very_high_threshold=very_high_threshold,
        preserve_periodic=preserve_periodic,
    ):
        return False
    protected = set(getattr(decision, "protected_reasons", ()))
    # OCR and sequence-role changes are never relaxed.  A semantic reason is
    # preserved even when no OCR text was available to corroborate it.
    if protected & {
        "ocr_change",
        "consequential_event",
        "before_action_after_role",
    }:
        return False
    difference = decision.difference
    return bool(
        difference.perceptual_hamming <= NEAR_DUPLICATE_HAMMING_THRESHOLD
        and difference.changed_pixel_ratio <= NEAR_DUPLICATE_CHANGED_RATIO
        and difference.mean_pixel_difference <= NEAR_DUPLICATE_MEAN_DIFFERENCE
        and difference.edge_difference <= NEAR_DUPLICATE_MEAN_DIFFERENCE
    )


def _unit(value: float, name: str) -> float:
    if not 0 <= value <= 1:
        raise InputError(f"{name} must be between 0 and 1")
    return value


def adaptive_selection_budget(
    duration_ms: int,
    *,
    important_event_count: int,
    evidence_density_per_minute: float,
    minimum: int = 1,
) -> int:
    if duration_ms < 0 or important_event_count < 0 or evidence_density_per_minute < 0:
        raise InputError("duration, event count, and evidence density cannot be negative")
    minutes = max(duration_ms / 60_000, 1 / 60)
    baseline = math.ceil(minutes * max(0.5, min(6.0, evidence_density_per_minute)))
    # Important events can each require a before/action/after group.
    event_allowance = important_event_count * 3
    return max(minimum, baseline, event_allowance)


def score_candidate(candidate: FrameCandidate) -> float:
    quality = candidate.quality or assess_frame_quality(candidate.path)
    transition_risk = (
        quality.transition_risk if candidate.transition_risk is None else candidate.transition_risk
    )
    fields = {
        "relevance": candidate.relevance,
        "importance": candidate.importance,
        "temporal_proximity": candidate.temporal_proximity,
        "stability": candidate.stability,
        "novelty": candidate.novelty,
        "ocr_readability": candidate.ocr_readability,
        "full_state_completeness": candidate.full_state_completeness,
        "transition_risk": transition_risk,
    }
    for name, value in fields.items():
        _unit(value, name)
    role_bonus = 0.08 if candidate.evidence_role in {"before", "action", "after", "result"} else 0.0
    small_change_bonus = 0.10 if candidate.consequential_change else 0.0
    raw = (
        0.18 * candidate.relevance
        + 0.17 * candidate.importance
        + 0.09 * candidate.temporal_proximity
        + 0.12 * quality.sharpness
        + 0.09 * candidate.stability
        + 0.10 * candidate.novelty
        + 0.09 * candidate.ocr_readability
        + 0.08 * candidate.full_state_completeness
        + role_bonus
        + small_change_bonus
        - 0.12 * transition_risk
    )
    return max(0.0, min(1.0, raw))


def _coerce(candidate: FrameCandidate | Mapping[str, object]) -> FrameCandidate:
    if isinstance(candidate, FrameCandidate):
        return candidate
    values = dict(candidate)
    values["path"] = Path(cast(str, values["path"]))
    if "reasons" in values:
        values["reasons"] = tuple(cast(Iterable[str], values["reasons"]))
    return FrameCandidate(**cast(Any, values))


def select_frames(
    candidates: Iterable[FrameCandidate | Mapping[str, object]],
    *,
    duration_ms: int,
    important_event_count: int | None = None,
    evidence_density_per_minute: float | None = None,
    safety_limit: int | None = None,
    protect_small_changes: bool = True,
    deduplicate: bool = True,
    importance_aware: bool = True,
    high_importance_threshold: float = HIGH_IMPORTANCE_THRESHOLD,
    very_high_importance_threshold: float = VERY_HIGH_IMPORTANCE_THRESHOLD,
    preserve_periodic_coverage: bool = True,
) -> FrameSelectionResult:
    materialized = [_coerce(candidate) for candidate in candidates]
    if not materialized:
        return FrameSelectionResult(
            (),
            (),
            (),
            0,
            False,
            None,
            (),
            {
                "candidate_count": 0,
                "selected_count": 0,
                "duplicate_count": 0,
                "low_score_count": 0,
                "importance_candidate_counts": {},
                "importance_selected_counts": {},
                "importance_coverage": {},
                "max_selected_gap_ms": None,
            },
        )
    _unit(high_importance_threshold, "high_importance_threshold")
    _unit(very_high_importance_threshold, "very_high_importance_threshold")
    if very_high_importance_threshold < high_importance_threshold:
        raise InputError(
            "very_high_importance_threshold cannot be lower than high_importance_threshold"
        )
    important_count = important_event_count
    if important_count is None:
        important_count = sum(
            item.importance >= 0.75 or item.consequential_change for item in materialized
        )
    density = evidence_density_per_minute
    if density is None:
        density = len(materialized) / max(duration_ms / 60_000, 1 / 60)
    budget = adaptive_selection_budget(
        duration_ms,
        important_event_count=important_count,
        evidence_density_per_minute=density,
    )
    if safety_limit is not None and safety_limit <= 0:
        raise InputError("safety_limit must be positive when configured")

    # Score once and reuse the result for both ranking and the selected record.
    # The quality object is normally already populated, but this also avoids
    # repeating validation/arithmetic for custom candidates.
    scores = {candidate.frame_id: score_candidate(candidate) for candidate in materialized}
    ranked = sorted(
        materialized,
        key=lambda item: (item.mandatory, scores[item.frame_id], -item.actual_ms),
        reverse=True,
    )
    selected: list[ScoredFrame] = []
    duplicates: list[str] = []
    low_score: list[str] = []
    hashes: dict[str, str] = {}
    provenance_by_id: dict[str, FrameSelectionProvenance] = {}
    covered_by_id: dict[str, list[str]] = {}
    for candidate in ranked:
        score = scores[candidate.frame_id]
        duplicate = False
        duplicate_representative: FrameCandidate | None = None
        duplicate_reason = ""
        duplicate_decision: Any | None = None
        duplicate_hamming: int | None = None
        # A candidate already marked as a consequential pixel/OCR transition
        # is explicitly protected by the upstream survey. Running the expensive
        # full-image comparison against every nearby selected frame cannot turn
        # it into a duplicate: ``deduplication_decision`` would protect it for
        # the same reason. Skip that redundant work while retaining the exact
        # candidate and its measured transition metadata.
        protected_reasons = (
            _dedup_protected_reasons(
                candidate,
                high_threshold=high_importance_threshold,
                very_high_threshold=very_high_importance_threshold,
                preserve_periodic=preserve_periodic_coverage,
            )
            if importance_aware
            else ()
        )
        # Protected frames are never sent through the duplicate path.  This is
        # both a safety guarantee and a measurable decode saving for the common
        # consequential-change case.
        can_compare_for_duplicate = (
            (not protected_reasons)
            if importance_aware
            else (not candidate.consequential_change)
        )
        if deduplicate and can_compare_for_duplicate:
            for prior in selected:
                # Far-apart frames are separate chronological evidence even
                # when a static slide looks identical; only compare locally
                # adjacent candidates for expensive pixel-level deduplication.
                if abs(candidate.actual_ms - prior.candidate.actual_ms) > 120_000:
                    continue
                candidate_hash = hashes.get(candidate.frame_id) or candidate.perceptual_hash
                if candidate_hash is None:
                    candidate_hash = perceptual_dhash(candidate.path)
                    hashes[candidate.frame_id] = candidate_hash
                prior_hash = hashes.get(prior.candidate.frame_id) or prior.candidate.perceptual_hash
                if prior_hash is None:
                    prior_hash = perceptual_dhash(prior.candidate.path)
                    hashes[prior.candidate.frame_id] = prior_hash
                duplicate_hamming = perceptual_hamming(candidate_hash, prior_hash)
                if duplicate_hamming > 5:
                    # The strict fast-path threshold is retained for exact
                    # duplicates.  Low-importance context gets a bounded
                    # relaxed comparison below to absorb codec jitter.
                    if not importance_aware:
                        continue
                    if duplicate_hamming > NEAR_DUPLICATE_HAMMING_THRESHOLD:
                        continue
                # The pipeline already computes a canonical normalized pixel
                # hash while creating each evidence image. Exact equality is a
                # safe fast path: it avoids reopening/decoding two large PNGs
                # for the common static-slide case, while still protecting
                # OCR changes, sequence roles, and consequential events.
                if (
                    candidate.pixel_hash is not None
                    and prior.candidate.pixel_hash is not None
                    and candidate.pixel_hash == prior.candidate.pixel_hash
                ):
                    ocr_changed = bool(
                        normalize_ocr_for_comparison(candidate.ocr_text)
                        or normalize_ocr_for_comparison(prior.candidate.ocr_text)
                    ) and normalize_ocr_for_comparison(candidate.ocr_text) != normalize_ocr_for_comparison(
                        prior.candidate.ocr_text
                    )
                    sequence_roles = {"before", "action", "after", "result"}
                    role_change = (
                        candidate.evidence_role in sequence_roles
                        and prior.candidate.evidence_role in sequence_roles
                        and candidate.evidence_role != prior.candidate.evidence_role
                    )
                    protected = (
                        (protect_small_changes and ocr_changed)
                        or candidate.consequential_change
                        or role_change
                    )
                    if importance_aware:
                        protected = protected or bool(protected_reasons)
                    duplicate = not protected
                    if duplicate:
                        duplicate_representative = prior.candidate
                        duplicate_reason = "exact_pixel_hash"
                        break
                    continue
                decision = deduplication_decision(
                    prior.candidate.path,
                    candidate.path,
                    left_ocr=prior.candidate.ocr_text,
                    right_ocr=candidate.ocr_text,
                    left_role=prior.candidate.evidence_role,
                    right_role=candidate.evidence_role,
                    consequential_change=candidate.consequential_change,
                    protect_small_changes=protect_small_changes,
                )
                duplicate_decision = decision
                relaxed_duplicate = (
                    importance_aware
                    and _low_importance_near_duplicate(
                        decision,
                        candidate=candidate,
                        prior=prior.candidate,
                        high_threshold=high_importance_threshold,
                        very_high_threshold=very_high_importance_threshold,
                        preserve_periodic=preserve_periodic_coverage,
                    )
                )
                if decision.is_duplicate or relaxed_duplicate:
                    duplicate = True
                    duplicate_representative = prior.candidate
                    duplicate_reason = (
                        "near_duplicate_low_importance"
                        if relaxed_duplicate and not decision.is_duplicate
                        else "perceptual_duplicate"
                    )
                    break
        if duplicate and not candidate.mandatory:
            duplicates.append(candidate.frame_id)
            representative_id = (
                duplicate_representative.frame_id if duplicate_representative is not None else None
            )
            if representative_id is not None:
                covered_by_id.setdefault(representative_id, []).append(candidate.frame_id)
            difference = getattr(duplicate_decision, "difference", None)
            provenance_by_id[candidate.frame_id] = FrameSelectionProvenance(
                frame_id=candidate.frame_id,
                status="duplicate",
                reason=duplicate_reason or "duplicate",
                importance=candidate.importance,
                importance_tier=importance_tier(
                    candidate.importance,
                    high_threshold=high_importance_threshold,
                    very_high_threshold=very_high_importance_threshold,
                ),
                score=score,
                representative_frame_id=representative_id,
                time_delta_ms=(
                    abs(candidate.actual_ms - duplicate_representative.actual_ms)
                    if duplicate_representative is not None
                    else None
                ),
                perceptual_hamming=(
                    int(difference.perceptual_hamming)
                    if difference is not None
                    else duplicate_hamming
                ),
                changed_pixel_ratio=(
                    float(difference.changed_pixel_ratio) if difference is not None else None
                ),
                ocr_changed=bool(getattr(duplicate_decision, "ocr_changed", False)),
                protected_reasons=tuple(getattr(duplicate_decision, "protected_reasons", ())),
            )
            continue
        budget_protected = (
            candidate.mandatory
            or candidate.consequential_change
            or (
                importance_aware
                and candidate.importance >= high_importance_threshold
            )
        )
        if len(selected) >= budget and not budget_protected:
            low_score.append(candidate.frame_id)
            provenance_by_id[candidate.frame_id] = FrameSelectionProvenance(
                frame_id=candidate.frame_id,
                status="low_score",
                reason="duration_aware_budget_exhausted",
                importance=candidate.importance,
                importance_tier=importance_tier(
                    candidate.importance,
                    high_threshold=high_importance_threshold,
                    very_high_threshold=very_high_importance_threshold,
                ),
                score=score,
                protected_reasons=protected_reasons,
            )
            continue
        reason_parts = list(candidate.reasons)
        if candidate.evidence_role in {"before", "action", "after", "result"}:
            reason_parts.append(f"sequence:{candidate.evidence_role}")
        if candidate.consequential_change:
            reason_parts.append("consequential-small-change")
        selected.append(
            ScoredFrame(
                candidate,
                score,
                "; ".join(dict.fromkeys(reason_parts)) or "highest evidence score",
            )
        )
        covered_by_id.setdefault(candidate.frame_id, [])
        provenance_by_id[candidate.frame_id] = FrameSelectionProvenance(
            frame_id=candidate.frame_id,
            status="selected",
            reason=(
                "; ".join(dict.fromkeys(reason_parts)) or "highest evidence score"
            ),
            importance=candidate.importance,
            importance_tier=importance_tier(
                candidate.importance,
                high_threshold=high_importance_threshold,
                very_high_threshold=very_high_importance_threshold,
            ),
            score=score,
            protected_reasons=protected_reasons,
        )
        if candidate.frame_id not in hashes:
            hashes[candidate.frame_id] = candidate.perceptual_hash or perceptual_dhash(candidate.path)

    if safety_limit is not None and len(selected) > safety_limit:
        pre_safety_selected = tuple(selected)
        required = [
            item
            for item in selected
            if item.candidate.mandatory
            or item.candidate.consequential_change
            or item.candidate.importance >= high_importance_threshold
        ]
        if len(required) > safety_limit:
            raise BlockedError(
                f"Configured image safety limit {safety_limit} cannot cover "
                f"{len(required)} important frames"
            )
        required_ids = {item.candidate.frame_id for item in required}
        optional = sorted(
            (item for item in selected if item.candidate.frame_id not in required_ids),
            key=lambda item: item.score,
            reverse=True,
        )
        selected = required + optional[: safety_limit - len(required)]
        retained_after_safety = {item.candidate.frame_id for item in selected}
        for item in pre_safety_selected:
            frame_id = item.candidate.frame_id
            if frame_id in retained_after_safety:
                continue
            if frame_id not in low_score:
                low_score.append(frame_id)
            provenance_by_id[frame_id] = FrameSelectionProvenance(
                frame_id=frame_id,
                status="low_score",
                reason="safety_limit_exhausted",
                importance=item.candidate.importance,
                importance_tier=importance_tier(
                    item.candidate.importance,
                    high_threshold=high_importance_threshold,
                    very_high_threshold=very_high_importance_threshold,
                ),
                score=item.score,
            )
        review_required = True
        review_reason = (
            "Safety limit stopped selection before the duration-aware target was reached"
        )
    else:
        review_required = False
        review_reason = None
    selected.sort(key=lambda item: (item.candidate.actual_ms, item.candidate.frame_id))
    # Finalize representative coverage after safety pruning.  A duplicate may
    # have been discovered while the representative was ranked, so attach the
    # complete cluster only once chronological ordering is known.
    candidate_by_id = {item.frame_id: item for item in materialized}
    selected_with_coverage: list[ScoredFrame] = []
    for item in selected:
        representative_id = item.candidate.frame_id
        covered_ids = [representative_id, *covered_by_id.get(representative_id, [])]
        covered_ids = sorted(
            dict.fromkeys(covered_ids),
            key=lambda frame_id: (
                candidate_by_id.get(frame_id, item.candidate).actual_ms,
                frame_id,
            ),
        )
        selected_with_coverage.append(
            ScoredFrame(
                candidate=item.candidate,
                score=item.score,
                selected_reason=item.selected_reason,
                covered_frame_ids=tuple(covered_ids),
            )
        )
        existing = provenance_by_id.get(representative_id)
        if existing is not None:
            provenance_by_id[representative_id] = FrameSelectionProvenance(
                frame_id=existing.frame_id,
                status=existing.status,
                reason=existing.reason,
                importance=existing.importance,
                importance_tier=existing.importance_tier,
                score=existing.score,
                representative_frame_id=existing.representative_frame_id,
                time_delta_ms=existing.time_delta_ms,
                perceptual_hamming=existing.perceptual_hamming,
                changed_pixel_ratio=existing.changed_pixel_ratio,
                ocr_changed=existing.ocr_changed,
                protected_reasons=existing.protected_reasons,
                covered_frame_ids=tuple(covered_ids),
            )

    tier_order = ("very_high", "high", "supporting", "low")
    candidate_counts = {tier: 0 for tier in tier_order}
    selected_counts = {tier: 0 for tier in tier_order}
    for candidate in materialized:
        tier = importance_tier(
            candidate.importance,
            high_threshold=high_importance_threshold,
            very_high_threshold=very_high_importance_threshold,
        )
        candidate_counts[tier] += 1
    for item in selected_with_coverage:
        tier = importance_tier(
            item.candidate.importance,
            high_threshold=high_importance_threshold,
            very_high_threshold=very_high_importance_threshold,
        )
        selected_counts[tier] += 1
    importance_coverage = {
        tier: (
            round(selected_counts[tier] / candidate_counts[tier], 6)
            if candidate_counts[tier]
            else 1.0
        )
        for tier in tier_order
    }
    selected_times = [item.candidate.actual_ms for item in selected_with_coverage]
    max_selected_gap_ms = (
        max(
            right - left
            for left, right in zip(selected_times, selected_times[1:], strict=False)
        )
        if len(selected_times) > 1
        else 0
    )
    covered_count = len(
        {
            frame_id
            for item in selected_with_coverage
            for frame_id in item.covered_frame_ids
        }
    )
    coverage: dict[str, Any] = {
        "candidate_count": len(materialized),
        "selected_count": len(selected_with_coverage),
        "duplicate_count": len(duplicates),
        "low_score_count": len(low_score),
        "covered_candidate_count": covered_count,
        "coverage_ratio": round(covered_count / len(materialized), 6) if materialized else 1.0,
        "importance_candidate_counts": candidate_counts,
        "importance_selected_counts": selected_counts,
        "importance_coverage": importance_coverage,
        "max_selected_gap_ms": max_selected_gap_ms,
        "selected_temporal_start_ms": selected_times[0] if selected_times else None,
        "selected_temporal_end_ms": selected_times[-1] if selected_times else None,
    }
    provenance = tuple(
        provenance_by_id[item.frame_id]
        for item in sorted(
            materialized,
            key=lambda candidate: (candidate.actual_ms, candidate.frame_id),
        )
        if item.frame_id in provenance_by_id
    )
    return FrameSelectionResult(
        tuple(selected_with_coverage),
        tuple(duplicates),
        tuple(low_score),
        budget,
        review_required,
        review_reason,
        provenance,
        coverage,
    )
