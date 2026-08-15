"""Append-only visual-observation ingestion, claim reconciliation, and sufficiency."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from .ids import sequential_id
from .image_claims import (
    claim_has_independent_support,
    claim_requires_independent_check,
    clone_claims,
    find_equivalent_claim,
    next_claim_id,
    observation_can_independently_check,
    proposed_to_claim,
    synthesize_factual_description,
)
from .image_metadata import canonical_json_bytes, prepare_metadata_payload
from .schemas import (
    EvidenceImageMetadata,
    EvidenceQuestion,
    ImageClaim,
    ImageMetadataRevision,
    MetadataSufficiencyDecision,
    ObservationHistoryEntry,
    VisualAnalysisObservation,
)


class ReconciliationError(ValueError):
    """Visual evidence cannot be reconciled without violating the ledger contract."""


class StaleBaseRevisionError(ReconciliationError):
    """An observation targets a stale revision and reconciliation was not allowed."""


class AppendOnlyViolationError(ReconciliationError):
    """An update attempts to delete or rewrite preserved observation history."""


@dataclass(frozen=True)
class ReconciliationResult:
    claims: list[ImageClaim]
    added_claim_ids: list[str]
    confirmed_claim_ids: list[str]
    narrowed_claim_ids: list[str]
    disputed_claim_ids: list[str]
    rejected_claim_ids: list[str]
    superseded_claim_ids: list[str]
    unresolved_claim_ids: list[str]
    stale_base_reconciled: bool
    method: str


@dataclass(frozen=True)
class ObservationIngestionResult:
    metadata: EvidenceImageMetadata
    revision: ImageMetadataRevision
    observations: list[VisualAnalysisObservation]
    reconciliation: ReconciliationResult


def _mapping_records(value: object) -> list[Mapping[str, object]]:
    """Normalize an untyped project array for model/index construction."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _sequence_values(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return value


@dataclass
class SemanticBatchContext:
    """Process-local indexes for a sequential semantic observation batch.

    Semantic packets are committed in deterministic order, while the provider
    calls themselves may run concurrently.  The historical ingestion API
    reparsed every observation and metadata revision for each packet.  This
    context owns those parsed models and the append-only counters once per
    batch; callers still persist the same canonical arrays and ledger records.

    The context is intentionally process-local and disposable.  A restart
    rebuilds it from canonical state after the semantic journal is recovered,
    so it is never part of the public on-disk contract.
    """

    observations: list[VisualAnalysisObservation] = field(default_factory=list)
    revisions: list[ImageMetadataRevision] = field(default_factory=list)
    observations_by_id: dict[str, VisualAnalysisObservation] = field(init=False)
    revisions_by_image: dict[str, list[ImageMetadataRevision]] = field(init=False)
    # The semantic batch mutates one image/event at a time, but the canonical
    # project arrays can contain thousands of records.  Keep the relationship
    # indexes process-local so evidence ingestion and semantic-link commits do
    # not rescan every frame/block/event/review for each packet.  The indexes
    # point at the original dictionaries, never copies, so in-place canonical
    # mutations remain visible to the normal journal/recovery path.
    frames_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    blocks_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    events_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    reviews_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    blocks_by_frame_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    events_by_frame_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reviews_by_frame_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reviews_by_event_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reviews_by_claim_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reviews_by_decision_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    guarded_review_claim_ids: set[str] = field(default_factory=set)
    payload_index_by_image_id: dict[str, int] = field(default_factory=dict)
    claim_index_by_id: dict[str, int] = field(default_factory=dict)
    latest_decision_by_image: dict[str, Mapping[str, object]] = field(default_factory=dict)
    max_revision_number: int = field(init=False, default=0)
    max_claim_number: int = field(default=0)
    max_decision_number: int = field(default=0)
    max_review_number: int = field(default=0)

    def __post_init__(self) -> None:
        self.observations_by_id = {
            item.observation_id: item for item in self.observations
        }
        self.revisions_by_image = {}
        for revision in self.revisions:
            self.revisions_by_image.setdefault(revision.image_id, []).append(revision)
            match = re.fullmatch(r"MR([0-9]{6})", revision.revision_id)
            if match:
                self.max_revision_number = max(self.max_revision_number, int(match.group(1)))

    @classmethod
    def from_project(cls, project: Mapping[str, object]) -> SemanticBatchContext:
        """Parse append-only semantic state once and seed batch counters."""

        observations = [
            VisualAnalysisObservation.model_validate(item)
            for item in _mapping_records(project.get("visual_observations", []))
        ]
        revisions = [
            ImageMetadataRevision.model_validate(item)
            for item in _mapping_records(project.get("metadata_revisions", []))
        ]
        context = cls(observations=observations, revisions=revisions)
        context.max_claim_number = _greatest_prefixed_number(
            project.get("image_claims", []), "claim_id", "IC"
        )
        context.max_decision_number = _greatest_prefixed_number(
            project.get("sufficiency_decisions", []), "decision_id", "MS"
        )
        context.max_review_number = _greatest_prefixed_number(
            project.get("review_items", []), "review_id", "R"
        )
        for item in _mapping_records(project.get("frames", [])):
            frame_id = item.get("frame_id") or item.get("image_id")
            if frame_id is not None and isinstance(item, dict):
                context.frames_by_id[str(frame_id)] = item
        for item in _mapping_records(project.get("script_blocks", [])):
            block_id = item.get("block_id")
            if block_id is None or not isinstance(item, dict):
                continue
            context.blocks_by_id[str(block_id)] = item
            for frame_id in _sequence_values(item.get("frame_ids", [])):
                context.blocks_by_frame_id.setdefault(str(frame_id), []).append(item)
        for item in _mapping_records(project.get("visual_events", [])):
            event_id = item.get("event_id")
            if event_id is None or not isinstance(item, dict):
                continue
            context.events_by_id[str(event_id)] = item
            for frame_id in _sequence_values(item.get("evidence_frame_ids", [])):
                context.events_by_frame_id.setdefault(str(frame_id), []).append(item)
        for item in _mapping_records(project.get("review_items", [])):
            review_id = item.get("review_id")
            if review_id is None or not isinstance(item, dict):
                continue
            context._index_review(item)
        for index, item in enumerate(_mapping_records(project.get("evidence_image_metadata", []))):
            image = item.get("image")
            image_id = image.get("image_id") if isinstance(image, Mapping) else None
            if image_id is not None:
                context.payload_index_by_image_id[str(image_id)] = index
        context.claim_index_by_id = {
            str(item.get("claim_id")): index
            for index, item in enumerate(_mapping_records(project.get("image_claims", [])))
            if item.get("claim_id") is not None
        }
        for decision in _mapping_records(project.get("sufficiency_decisions", [])):
            for image_id in _sequence_values(decision.get("image_ids", [])):
                context.latest_decision_by_image[str(image_id)] = decision
        return context

    def _index_review(self, review: dict[str, Any]) -> None:
        """Index one review while preserving canonical insertion order."""

        review_id = review.get("review_id")
        if review_id is None:
            return
        review_key = str(review_id)
        self.reviews_by_id[review_key] = review
        category = str(review.get("category") or "")
        guarded_category = category in {"disputed_image_claim", "high_impact_image_claim"}
        for frame_id in _sequence_values(review.get("frame_ids", [])):
            self.reviews_by_frame_id.setdefault(str(frame_id), []).append(review)
        for event_id in _sequence_values(review.get("event_ids", [])):
            self.reviews_by_event_id.setdefault(str(event_id), []).append(review)
        for claim_id in _sequence_values(review.get("image_claim_ids", [])):
            claim_key = str(claim_id)
            self.reviews_by_claim_id.setdefault(claim_key, []).append(review)
            if guarded_category:
                self.guarded_review_claim_ids.add(claim_key)
        for decision_id in _sequence_values(review.get("sufficiency_decision_ids", [])):
            self.reviews_by_decision_id.setdefault(str(decision_id), []).append(review)

    def latest_revision_for_image(self, image_id: str) -> ImageMetadataRevision | None:
        revisions = self.revisions_by_image.get(image_id)
        if not revisions:
            return None
        # Canonical revisions are append-only; max keeps the invariant explicit
        # even if a legacy project contains an out-of-order revision array.
        return max(revisions, key=lambda item: item.revision_number)

    def allocate_revision_id(self, current_revision_id: str | None = None) -> str:
        greatest = self.max_revision_number
        current_match = re.fullmatch(r"MR([0-9]{6})", current_revision_id or "")
        if current_match:
            greatest = max(greatest, int(current_match.group(1)))
        self.max_revision_number = greatest + 1
        return sequential_id("metadata_revision", self.max_revision_number)

    def allocate_claim_id(self) -> str:
        self.max_claim_number += 1
        return sequential_id("image_claim", self.max_claim_number)

    def allocate_decision_id(self) -> str:
        self.max_decision_number += 1
        return sequential_id("metadata_sufficiency", self.max_decision_number)

    def record_decision(self, image_id: str, decision: Mapping[str, object]) -> None:
        self.latest_decision_by_image[image_id] = decision

    def record_claims(self, claims: Sequence[ImageClaim]) -> None:
        for claim in claims:
            match = re.fullmatch(r"IC([0-9]{6})", claim.claim_id)
            if match:
                self.max_claim_number = max(self.max_claim_number, int(match.group(1)))

    def record_reviews(self, reviews: Sequence[Mapping[str, object]]) -> None:
        for review in reviews:
            value = str(review.get("review_id", ""))
            match = re.fullmatch(r"R([0-9]{6})", value)
            if match:
                self.max_review_number = max(self.max_review_number, int(match.group(1)))
            if isinstance(review, dict):
                self._index_review(review)

    def record_payload(self, image_id: str, index: int) -> None:
        """Record the stable canonical payload slot for a semantic batch."""

        self.payload_index_by_image_id[str(image_id)] = int(index)

    def commit(
        self,
        observation: VisualAnalysisObservation,
        revision: ImageMetadataRevision,
    ) -> None:
        """Record one successful image transaction in the in-memory indexes."""

        self.observations.append(observation)
        self.observations_by_id[observation.observation_id] = observation
        self.revisions.append(revision)
        self.revisions_by_image.setdefault(revision.image_id, []).append(revision)
        match = re.fullmatch(r"MR([0-9]{6})", revision.revision_id)
        if match:
            self.max_revision_number = max(self.max_revision_number, int(match.group(1)))


def _greatest_prefixed_number(
    values: object,
    key: str,
    prefix: str,
) -> int:
    """Return the greatest numeric suffix from a list of mapping records."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return 0
    greatest = 0
    for item in values:
        if not isinstance(item, Mapping):
            continue
        value = str(item.get(key, ""))
        if value.startswith(prefix) and value.removeprefix(prefix).isdigit():
            greatest = max(greatest, int(value.removeprefix(prefix)))
    return greatest


def _append_unique(values: list[str], *new_values: str) -> None:
    for value in new_values:
        if value and value not in values:
            values.append(value)


def _by_id(claims: Sequence[ImageClaim]) -> dict[str, ImageClaim]:
    result: dict[str, ImageClaim] = {}
    for claim in claims:
        if claim.claim_id in result:
            raise ReconciliationError(f"duplicate existing claim ID: {claim.claim_id}")
        result[claim.claim_id] = claim
    return result


def reconcile_observation(
    existing_claims: Sequence[ImageClaim],
    observation: VisualAnalysisObservation,
    *,
    revision_id: str,
    current_revision_id: str | None = None,
    allow_stale_reconcile: bool = True,
) -> ReconciliationResult:
    """Reconcile one accepted observation without deleting historical evidence.

    Stale observations are either rejected explicitly or merged against the current
    claims with ``stale_base_reconciled=True``.  They never overwrite the current
    envelope as a last writer.
    """

    if observation.validation_result != "accepted":
        raise ReconciliationError("only schema-validated accepted observations may be reconciled")
    stale = observation.base_revision_id != current_revision_id
    # Creation may legitimately have both values None.
    if stale and not allow_stale_reconcile:
        detail = f"base {observation.base_revision_id!r}; current {current_revision_id!r}"
        raise StaleBaseRevisionError(f"observation revision is stale: {detail}")
    claims = clone_claims(existing_claims)
    claims_by_id = _by_id(claims)
    added: list[str] = []
    confirmed: list[str] = []
    narrowed: list[str] = []
    disputed: list[str] = []
    rejected: list[str] = []
    superseded: list[str] = []

    def require_claim(claim_id: str) -> ImageClaim:
        try:
            return claims_by_id[claim_id]
        except KeyError as exc:
            raise ReconciliationError(f"observation references unknown claim {claim_id}") from exc

    def may_promote(claim: ImageClaim) -> bool:
        if not claim_requires_independent_check(claim):
            return True
        return observation_can_independently_check(observation)

    for claim_id in observation.independently_confirmed_claim_ids:
        claim = require_claim(claim_id)
        _append_unique(claim.supporting_observation_ids, observation.observation_id)
        claim.last_updated_revision_id = revision_id
        if (
            claim.status in {"proposed", "unresolved"}
            and not claim.contradicting_observation_ids
            and may_promote(claim)
        ):
            claim.status = "supported"
        _append_unique(confirmed, claim_id)

    for claim_id in observation.contradicted_claim_ids:
        claim = require_claim(claim_id)
        _append_unique(claim.contradicting_observation_ids, observation.observation_id)
        claim.status = "disputed"
        claim.last_updated_revision_id = revision_id
        _append_unique(disputed, claim_id)

    for claim_id in observation.rejected_claim_ids:
        claim = require_claim(claim_id)
        _append_unique(claim.contradicting_observation_ids, observation.observation_id)
        claim.status = "rejected"
        claim.last_updated_revision_id = revision_id
        _append_unique(rejected, claim_id)

    for proposed in observation.proposed_claims:
        if (
            proposed.relationship in {"confirm", "contradict", "reject", "narrow"}
            and not proposed.related_claim_ids
        ):
            raise ReconciliationError(
                f"{proposed.relationship} proposal requires related_claim_ids"
            )
        equivalent = find_equivalent_claim(proposed, claims)
        if equivalent is not None and proposed.relationship in {"new", "confirm"}:
            _append_unique(equivalent.supporting_observation_ids, observation.observation_id)
            equivalent.last_updated_revision_id = revision_id
            if (
                equivalent.status in {"proposed", "unresolved"}
                and not equivalent.contradicting_observation_ids
                and may_promote(equivalent)
            ):
                equivalent.status = "supported"
            _append_unique(confirmed, equivalent.claim_id)
            continue

        if proposed.relationship == "confirm":
            for claim_id in proposed.related_claim_ids:
                claim = require_claim(claim_id)
                _append_unique(claim.supporting_observation_ids, observation.observation_id)
                claim.last_updated_revision_id = revision_id
                if (
                    claim.status in {"proposed", "unresolved"}
                    and not claim.contradicting_observation_ids
                    and may_promote(claim)
                ):
                    claim.status = "supported"
                _append_unique(confirmed, claim_id)
            continue

        if proposed.relationship == "reject":
            for claim_id in proposed.related_claim_ids:
                claim = require_claim(claim_id)
                _append_unique(claim.contradicting_observation_ids, observation.observation_id)
                claim.status = "rejected"
                claim.last_updated_revision_id = revision_id
                _append_unique(rejected, claim_id)
            continue

        claim_id = proposed.claim_id or next_claim_id(claims)
        if claim_id in claims_by_id:
            raise ReconciliationError(f"new proposed claim reuses existing ID {claim_id}")
        new_claim = proposed_to_claim(
            proposed,
            observation_id=observation.observation_id,
            image_ids=observation.image_ids,
            revision_id=revision_id,
            claim_id=claim_id,
        )
        if (
            new_claim.status == "proposed"
            and observation.actor_kind == "human"
            and proposed.relationship == "new"
        ):
            # An attributable human inspecting the pixels is itself an eligible
            # independent evidentiary actor. A blind machine pass can confirm an
            # existing guarded claim, but it cannot self-confirm a brand-new one.
            new_claim.status = "supported"
        if proposed.relationship == "contradict":
            new_claim.status = "disputed"
            for related_id in proposed.related_claim_ids:
                old = require_claim(related_id)
                _append_unique(
                    new_claim.contradicting_observation_ids, *old.supporting_observation_ids
                )
                _append_unique(old.contradicting_observation_ids, observation.observation_id)
                old.status = "disputed"
                old.last_updated_revision_id = revision_id
                _append_unique(disputed, related_id)
            _append_unique(disputed, claim_id)
        elif proposed.relationship == "narrow":
            for related_id in proposed.related_claim_ids:
                old = require_claim(related_id)
                old.status = "superseded"
                old.superseded_by_claim_id = claim_id
                old.last_updated_revision_id = revision_id
                _append_unique(superseded, related_id)
            _append_unique(narrowed, claim_id)
        claims.append(new_claim)
        claims_by_id[claim_id] = new_claim
        _append_unique(added, claim_id)

    unresolved = [claim.claim_id for claim in claims if claim.status in {"proposed", "unresolved"}]
    # Stable claim ordering is ID order, never observer output ordering.
    claims.sort(key=lambda item: item.claim_id)
    return ReconciliationResult(
        claims=claims,
        added_claim_ids=sorted(added),
        confirmed_claim_ids=sorted(confirmed),
        narrowed_claim_ids=sorted(narrowed),
        disputed_claim_ids=sorted(set(disputed)),
        rejected_claim_ids=sorted(set(rejected)),
        superseded_claim_ids=sorted(set(superseded)),
        unresolved_claim_ids=sorted(unresolved),
        stale_base_reconciled=stale,
        method="stale-base-explicit-merge-v1" if stale else "atomic-evidence-reconciliation-v1",
    )


def _next_revision_id(
    revisions: Sequence[ImageMetadataRevision], current_revision_id: str | None = None
) -> str:
    greatest = 0
    for revision in revisions:
        match = re.fullmatch(r"MR([0-9]{6})", revision.revision_id)
        if match:
            greatest = max(greatest, int(match.group(1)))
    current_match = re.fullmatch(r"MR([0-9]{6})", current_revision_id or "")
    if current_match:
        greatest = max(greatest, int(current_match.group(1)))
    return sequential_id("metadata_revision", greatest + 1)


def _revision_journal_digest(
    revision_id: str,
    image_id: str,
    base_revision_id: str | None,
    observation_id: str,
    result: ReconciliationResult,
) -> str:
    content = {
        "revision_id": revision_id,
        "image_id": image_id,
        "base_revision_id": base_revision_id,
        "observation_id": observation_id,
        "added": result.added_claim_ids,
        "confirmed": result.confirmed_claim_ids,
        "narrowed": result.narrowed_claim_ids,
        "disputed": result.disputed_claim_ids,
        "rejected": result.rejected_claim_ids,
        "superseded": result.superseded_claim_ids,
        "unresolved": result.unresolved_claim_ids,
        "method": result.method,
    }
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def ingest_observation(
    metadata: EvidenceImageMetadata,
    observations: Sequence[VisualAnalysisObservation],
    revisions: Sequence[ImageMetadataRevision],
    observation: VisualAnalysisObservation,
    *,
    allow_stale_reconcile: bool = True,
    now_utc: str | None = None,
    batch_context: SemanticBatchContext | None = None,
) -> ObservationIngestionResult:
    """Append an observation and prepare one monotonic logical metadata revision.

    The returned revision is intentionally uncommitted.  After the image write and
    canonical two-phase commit succeed, use :func:`mark_revision_committed`.
    """

    if metadata.image.image_id not in observation.image_ids:
        raise ReconciliationError("observation did not inspect the metadata image")
    existing_by_id = (
        batch_context.observations_by_id
        if batch_context is not None
        else {item.observation_id: item for item in observations}
    )
    if observation.observation_id in existing_by_id:
        if existing_by_id[observation.observation_id].model_dump(
            mode="json"
        ) != observation.model_dump(mode="json"):
            detail = f"observation ID {observation.observation_id}"
            raise AppendOnlyViolationError(
                f"{detail} was previously accepted with different content"
            )
        raise AppendOnlyViolationError(
            f"observation ID {observation.observation_id} is already present"
        )
    image_revisions = (
        batch_context.revisions_by_image.get(metadata.image.image_id, [])
        if batch_context is not None
        else sorted(
            (item for item in revisions if item.image_id == metadata.image.image_id),
            key=lambda item: item.revision_number,
        )
    )
    if image_revisions:
        latest = (
            batch_context.latest_revision_for_image(metadata.image.image_id)
            if batch_context is not None
            else image_revisions[-1]
        )
        assert latest is not None
        if (
            latest.revision_id != metadata.analysis.latest_revision_id
            or latest.revision_number != metadata.analysis.revision_number
        ):
            raise ReconciliationError(
                "metadata envelope does not match the latest canonical revision"
            )
    revision_id = (
        batch_context.allocate_revision_id(metadata.analysis.latest_revision_id)
        if batch_context is not None
        else _next_revision_id(revisions, metadata.analysis.latest_revision_id)
    )
    result = reconcile_observation(
        metadata.knowledge.claims,
        observation,
        revision_id=revision_id,
        current_revision_id=metadata.analysis.latest_revision_id,
        allow_stale_reconcile=allow_stale_reconcile,
    )
    raw = deepcopy(metadata.model_dump(mode="json"))
    previous_revision_id = metadata.analysis.latest_revision_id
    previous_digest = metadata.integrity.payload_digest
    raw["knowledge"]["claims"] = [claim.model_dump(mode="json") for claim in result.claims]
    raw["knowledge"]["supported_claim_ids"] = [
        c.claim_id for c in result.claims if c.status == "supported"
    ]
    raw["knowledge"]["disputed_claim_ids"] = [
        c.claim_id for c in result.claims if c.status == "disputed"
    ]
    raw["knowledge"]["rejected_claim_ids"] = [
        c.claim_id for c in result.claims if c.status in {"rejected", "superseded"}
    ]
    raw["knowledge"]["unresolved_claim_ids"] = [
        c.claim_id for c in result.claims if c.status in {"proposed", "unresolved"}
    ]
    raw["knowledge"]["current_factual_description"] = synthesize_factual_description(result.claims)
    raw["analysis"]["enrichment_level"] = "semantic"
    raw["analysis"]["semantic_status"] = "observed"
    raw["analysis"]["latest_revision_id"] = revision_id
    raw["analysis"]["revision_number"] = metadata.analysis.revision_number + 1
    outcome_parts = []
    for label, values in (
        ("added", result.added_claim_ids),
        ("confirmed", result.confirmed_claim_ids),
        ("narrowed", result.narrowed_claim_ids),
        ("disputed", result.disputed_claim_ids),
        ("rejected", result.rejected_claim_ids),
    ):
        if values:
            outcome_parts.append(f"{label} {', '.join(values)}")
    history_entry = ObservationHistoryEntry(
        observation_id=observation.observation_id,
        actor_kind=observation.actor_kind,
        actor_label=observation.actor_label,
        observed_at_utc=observation.observed_at_utc,
        prior_metadata_visible=observation.prior_metadata_visible,
        purpose=observation.purpose,
        outcome="; ".join(outcome_parts) or "No supported information gained.",
    )
    raw["analysis"]["observation_history"].append(history_entry.model_dump(mode="json"))
    raw["integrity"]["previous_revision_id"] = previous_revision_id
    raw["integrity"]["previous_payload_digest"] = previous_digest
    raw["integrity"]["canonical_revision_locator"] = (
        f".state/vision/image-observations.json#{revision_id}"
    )
    raw["integrity"]["canonical_revision_digest"] = _revision_journal_digest(
        revision_id,
        metadata.image.image_id,
        previous_revision_id,
        observation.observation_id,
        result,
    )
    updated = prepare_metadata_payload(raw)
    timestamp = now_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    revision = ImageMetadataRevision(
        revision_id=revision_id,
        revision_number=updated.analysis.revision_number,
        image_id=metadata.image.image_id,
        base_revision_id=previous_revision_id,
        observation_ids=[observation.observation_id],
        added_claim_ids=result.added_claim_ids,
        confirmed_claim_ids=result.confirmed_claim_ids,
        narrowed_claim_ids=result.narrowed_claim_ids,
        disputed_claim_ids=result.disputed_claim_ids,
        rejected_claim_ids=result.rejected_claim_ids,
        superseded_claim_ids=result.superseded_claim_ids,
        unresolved_claim_ids=result.unresolved_claim_ids,
        previous_payload_digest=previous_digest,
        new_payload_digest=updated.integrity.payload_digest,
        reconciliation_method=result.method,
        actor=observation.actor_label,
        stale_base_reconciled=result.stale_base_reconciled,
        pixel_invariance_verified=False,
        embedded_write_verified=False,
        read_back_verified=False,
        canonical_mirror_committed=False,
        created_at_utc=timestamp,
    )
    accepted_observation = observation.model_copy(
        update={"ingestion_result": f"accepted in {revision_id}"}
    )
    return ObservationIngestionResult(
        metadata=updated,
        revision=revision,
        observations=[*observations, accepted_observation],
        reconciliation=result,
    )


def mark_revision_committed(revision: ImageMetadataRevision) -> ImageMetadataRevision:
    """Return the immutable commit-state update after all two-phase checks pass."""

    return revision.model_copy(
        update={
            "pixel_invariance_verified": True,
            "embedded_write_verified": True,
            "read_back_verified": True,
            "canonical_mirror_committed": True,
        }
    )


def evaluate_sufficiency(
    *,
    decision_id: str,
    questions: Sequence[EvidenceQuestion],
    claims: Sequence[ImageClaim],
    observations: Sequence[VisualAnalysisObservation],
    image_ids: Sequence[str],
    metadata_revision_ids: Sequence[str],
    visual_event_ids: Sequence[str] = (),
    script_block_ids: Sequence[str] = (),
    payload_current_and_valid: bool = True,
    unattempted_evidence_actions: Sequence[str] = (),
    semantic_observer_available: bool = True,
    pass_limit_reached: bool = False,
    no_further_evidence: bool = False,
    decided_by: str = "metadata-sufficiency-rule-v1",
    now_utc: str | None = None,
) -> MetadataSufficiencyDecision:
    """Apply deterministic, question-scoped sufficiency and stopping rules."""

    by_id = {claim.claim_id: claim for claim in claims}
    answered: list[str] = []
    supporting: list[str] = []
    unanswered: list[str] = []
    gaps: list[str] = []
    for question in questions:
        candidates = [
            by_id[claim_id] for claim_id in question.candidate_claim_ids if claim_id in by_id
        ]
        usable: list[ImageClaim] = []
        for claim in candidates:
            if claim.status != "supported" or not claim.supporting_observation_ids:
                continue
            has_pixel_basis = bool(claim.region_xywh_normalized or claim.evidence_regions)
            has_image_basis = bool(claim.supporting_image_ids)
            if not has_pixel_basis or not has_image_basis:
                continue
            if claim_requires_independent_check(claim) and not claim_has_independent_support(
                claim, observations
            ):
                continue
            usable.append(claim)
        if usable and payload_current_and_valid:
            answered.append(question.question_id)
            supporting.extend(claim.claim_id for claim in usable)
        else:
            unanswered.append(question.question_id)
            if not payload_current_and_valid:
                gaps.append(
                    f"{question.question_id}: image payload is stale, corrupt, or mismatched"
                )
            elif candidates and any(claim.status == "disputed" for claim in candidates):
                gaps.append(f"{question.question_id}: credible image claims remain disputed")
            elif candidates and any(claim_requires_independent_check(c) for c in candidates):
                gaps.append(
                    f"{question.question_id}: high-impact/disputed evidence lacks "
                    "an independent blind check"
                )
            else:
                gaps.append(
                    f"{question.question_id}: no current supported claim answers "
                    f"at {question.required_precision}"
                )
    if not unanswered:
        status: Literal[
            "sufficient",
            "insufficient",
            "no_further_evidence",
            "limit_reached",
            "semantic_observer_unavailable",
        ] = "sufficient"
        next_action = None
        rationale = "Every evaluated question has current pixel-grounded supported evidence."
    elif not semantic_observer_available:
        status = "semantic_observer_unavailable"
        next_action = "Obtain attributable human or configured semantic visual review."
        rationale = "Creation/deterministic metadata cannot answer all semantic questions."
    elif pass_limit_reached:
        status = "limit_reached"
        next_action = "Create an exact review item for each unanswered consequential question."
        rationale = "The pass limit is not evidence of sufficiency."
    elif no_further_evidence:
        status = "no_further_evidence"
        next_action = "Retain uncertainty and route consequential gaps to review."
        rationale = "No available evidence action is likely to reduce the recorded uncertainty."
    else:
        status = "insufficient"
        next_action = (
            unattempted_evidence_actions[0]
            if unattempted_evidence_actions
            else "Perform a targeted pass using better pixels, a justified crop, "
            "adjacent frames, OCR, or audio."
        )
        rationale = "One or more exact evidence questions remain unanswered."
    timestamp = now_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return MetadataSufficiencyDecision(
        decision_id=decision_id,
        image_ids=list(image_ids),
        visual_event_ids=list(visual_event_ids),
        script_block_ids=list(script_block_ids),
        metadata_revision_ids=list(metadata_revision_ids),
        questions=list(questions),
        answered_question_ids=answered,
        supporting_claim_ids=list(dict.fromkeys(supporting)),
        unanswered_question_ids=unanswered,
        exact_gaps=gaps,
        unattempted_evidence_actions=list(unattempted_evidence_actions),
        status=status,
        recommended_next_action=next_action,
        decided_by=decided_by,
        decided_at_utc=timestamp,
        rationale=rationale,
    )


def should_request_another_pass(
    decision: MetadataSufficiencyDecision,
    *,
    consecutive_passes_without_new_supported_information: int,
    stop_after_no_new_supported_information_passes: int = 2,
) -> bool:
    if decision.status != "insufficient":
        return False
    if (
        consecutive_passes_without_new_supported_information
        >= stop_after_no_new_supported_information_passes
    ):
        return False
    return bool(decision.unattempted_evidence_actions or decision.recommended_next_action)


__all__ = [
    "AppendOnlyViolationError",
    "ObservationIngestionResult",
    "ReconciliationError",
    "ReconciliationResult",
    "SemanticBatchContext",
    "StaleBaseRevisionError",
    "evaluate_sufficiency",
    "ingest_observation",
    "mark_revision_committed",
    "reconcile_observation",
    "should_request_another_pass",
]
