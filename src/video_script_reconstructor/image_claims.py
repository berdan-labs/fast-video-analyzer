"""Atomic image-claim helpers and evidence-quality rules."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from copy import deepcopy

from .ids import sequential_id
from .schemas import ClaimStatus, ImageClaim, ProposedImageClaim, VisualAnalysisObservation


def normalize_claim_text(text: str) -> str:
    """Normalize only for semantic deduplication; never replace preserved wording."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.rstrip(". ")


def claim_equivalence_key(claim: ImageClaim | ProposedImageClaim) -> tuple[str, str, str]:
    value = normalize_claim_text(claim.normalized_value or "")
    return claim.claim_class, normalize_claim_text(claim.statement), value


def find_equivalent_claim(
    proposed: ProposedImageClaim, existing_claims: Sequence[ImageClaim]
) -> ImageClaim | None:
    key = claim_equivalence_key(proposed)
    return next((claim for claim in existing_claims if claim_equivalence_key(claim) == key), None)


def next_claim_id(claims: Iterable[ImageClaim]) -> str:
    greatest = 0
    for claim in claims:
        match = re.fullmatch(r"IC([0-9]{6})", claim.claim_id)
        if match:
            greatest = max(greatest, int(match.group(1)))
    return sequential_id("image_claim", greatest + 1)


def proposed_to_claim(
    proposed: ProposedImageClaim,
    *,
    observation_id: str,
    image_ids: Sequence[str],
    revision_id: str,
    claim_id: str,
) -> ImageClaim:
    requires_independent = proposed.high_impact_token or proposed.importance == "high_impact"
    status: ClaimStatus = (
        "unresolved"
        if proposed.relationship == "unresolved" or proposed.claim_class == "unresolved"
        else "proposed"
        if requires_independent
        else "supported"
    )
    primary_region = (
        proposed.evidence_regions[0].region_xywh_normalized
        if len(proposed.evidence_regions) == 1
        else None
    )
    return ImageClaim(
        claim_id=claim_id,
        claim_class=proposed.claim_class,
        statement=proposed.statement,
        normalized_value=proposed.normalized_value,
        status=status,
        importance=proposed.importance,
        high_impact_token=proposed.high_impact_token,
        confidence=proposed.confidence,
        calibration_basis=proposed.calibration_basis,
        supporting_image_ids=list(dict.fromkeys(image_ids)),
        region_xywh_normalized=primary_region,
        evidence_regions=proposed.evidence_regions,
        ocr_observation_ids=proposed.ocr_observation_ids,
        # A proposed/high-impact claim still records the observation that
        # introduced it.  It is deliberately not consumable until a later
        # independent blind or attributable human check promotes it.
        supporting_observation_ids=[observation_id],
        first_seen_revision_id=revision_id,
        last_updated_revision_id=revision_id,
        uncertainty=proposed.uncertainty,
        alternatives=proposed.alternatives,
        supersedes_claim_ids=proposed.related_claim_ids
        if proposed.relationship == "narrow"
        else [],
    )


def observation_can_independently_check(observation: VisualAnalysisObservation) -> bool:
    """Whether one observation is eligible to promote a guarded claim."""

    return observation.actor_kind == "human" or (
        observation.analysis_depth == "blind" and not observation.prior_metadata_visible
    )


def independent_observation_pairs(
    claim: ImageClaim, observations: Sequence[VisualAnalysisObservation]
) -> list[tuple[VisualAnalysisObservation, VisualAnalysisObservation]]:
    """Return supporting pairs that are meaningfully independent.

    A blind pass and a cumulative pass count when the blind pass did not see prior
    metadata and the two observations do not share the same provider/model/prompt
    tuple. Repeated identical calls remain correlated.
    """

    by_id = {observation.observation_id: observation for observation in observations}
    supporters = [by_id[item] for item in claim.supporting_observation_ids if item in by_id]
    pairs: list[tuple[VisualAnalysisObservation, VisualAnalysisObservation]] = []
    for index, left in enumerate(supporters):
        for right in supporters[index + 1 :]:
            left_signature = (
                left.provider,
                left.model,
                left.model_version,
                left.prompt_template_hash,
            )
            right_signature = (
                right.provider,
                right.model,
                right.model_version,
                right.prompt_template_hash,
            )
            actor_independent = left.actor_kind == "human" or right.actor_kind == "human"
            blind_independent = (
                (left.analysis_depth == "blind" and not left.prior_metadata_visible)
                or (right.analysis_depth == "blind" and not right.prior_metadata_visible)
            ) and (
                left_signature != right_signature
                or left.prior_metadata_visible != right.prior_metadata_visible
                or left.analysis_depth != right.analysis_depth
            )
            if actor_independent or blind_independent:
                pairs.append((left, right))
    return pairs


def claim_has_independent_support(
    claim: ImageClaim, observations: Sequence[VisualAnalysisObservation]
) -> bool:
    by_id = {observation.observation_id: observation for observation in observations}
    supporters = [by_id[item] for item in claim.supporting_observation_ids if item in by_id]
    # An attributable human review is itself an independent evidentiary actor;
    # it does not need a second machine pass to become eligible.  Automated
    # high-impact claims still require a cumulative/blind comparison pair.
    if any(observation.actor_kind == "human" for observation in supporters):
        return True
    return bool(independent_observation_pairs(claim, observations))


def claim_requires_independent_check(claim: ImageClaim) -> bool:
    return (
        claim.high_impact_token or claim.importance == "high_impact" or claim.status == "disputed"
    )


def synthesize_factual_description(claims: Sequence[ImageClaim]) -> str | None:
    """Build current description only from supported claims, in stable order."""

    statements = [claim.statement.strip() for claim in claims if claim.status == "supported"]
    if not statements:
        return None
    return " ".join(
        statement if statement.endswith((".", "!", "?")) else statement + "."
        for statement in statements
    )


def clone_claims(claims: Sequence[ImageClaim]) -> list[ImageClaim]:
    return [ImageClaim.model_validate(deepcopy(claim.model_dump(mode="json"))) for claim in claims]


__all__ = [
    "claim_equivalence_key",
    "claim_has_independent_support",
    "claim_requires_independent_check",
    "clone_claims",
    "find_equivalent_claim",
    "independent_observation_pairs",
    "observation_can_independently_check",
    "next_claim_id",
    "normalize_claim_text",
    "proposed_to_claim",
    "synthesize_factual_description",
]
