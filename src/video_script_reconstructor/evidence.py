from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from .audit import audit_project
from .errors import InputError, ValidationFailure
from .image_claims import claim_has_independent_support, claim_requires_independent_check
from .image_metadata import (
    embed_metadata_with_file_hash,
    normalized_pixel_hash,
    read_embedded_metadata,
    verify_embedded_metadata,
)
from .metadata_reconcile import (
    SemanticBatchContext,
    evaluate_sufficiency,
    ingest_observation,
    mark_revision_committed,
)
from .render_markdown import render_to_path
from .review import load_project
from .schemas import EvidenceQuestion, ImageMetadataRevision, VisualAnalysisObservation
from .security import (
    JsonPatchState,
    atomic_update_json_fields,
    atomic_write_json,
    canonical_compact_for_payload,
    safe_relative_path,
)
from .validate_output import validate_project


def _frame_and_path(
    project_dir: Path,
    project: dict[str, Any],
    image_id: str,
    *,
    frame_index: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path]:
    frame = frame_index.get(image_id) if frame_index is not None else None
    if frame is None:
        frame = next(
            (
                item
                for item in project.get("frames", [])
                if (item.get("frame_id") or item.get("image_id")) == image_id
            ),
            None,
        )
    if frame is None:
        raise InputError(f"Unknown frame or crop ID: {image_id}")
    relative = frame.get("full_frame_path") or frame.get("path")
    if not relative:
        raise ValidationFailure(f"Image record {image_id} has no artifact path")
    return frame, safe_relative_path(project_dir.resolve(strict=True), str(relative))


def show_image_metadata(project_dir: Path, image_id: str) -> dict[str, Any]:
    project = load_project(project_dir)
    frame, path = _frame_and_path(project_dir, project, image_id)
    embedded = verify_embedded_metadata(path)
    canonical = next(
        (
            item
            for item in project.get("evidence_image_metadata", [])
            if item.get("image", {}).get("image_id") == image_id
        ),
        None,
    )
    if canonical is None:
        raise ValidationFailure(f"Canonical metadata mirror is missing for {image_id}")
    verify_embedded_metadata(path, canonical)
    revisions = [
        item for item in project.get("metadata_revisions", []) if item.get("image_id") == image_id
    ]
    observations = [
        item
        for item in project.get("visual_observations", [])
        if image_id in item.get("image_ids", [])
    ]
    return {
        "image_path": str(path),
        "current_factual_description": embedded.knowledge.current_factual_description,
        "claims": [item.model_dump(mode="json") for item in embedded.knowledge.claims],
        "explicit_unknowns": embedded.knowledge.explicit_unknowns,
        "unanswered_questions": embedded.analysis.sufficiency.unanswered_questions,
        "semantic_status": embedded.analysis.semantic_status,
        "latest_revision_id": embedded.analysis.latest_revision_id,
        "revision_number": embedded.analysis.revision_number,
        "observation_history": [
            item.model_dump(mode="json") for item in embedded.analysis.observation_history
        ],
        "revisions": revisions,
        "full_observations": observations,
        "integrity": embedded.integrity.model_dump(mode="json"),
        "canonical_match": True,
        "pixel_hash": normalized_pixel_hash(path),
    }


def verify_image_metadata(project_dir: Path, image_id: str | None = None) -> list[dict[str, Any]]:
    project = load_project(project_dir)
    targets = (
        [image_id]
        if image_id
        else [
            str(item.get("frame_id") or item.get("image_id")) for item in project.get("frames", [])
        ]
    )
    results: list[dict[str, Any]] = []
    for target in targets:
        frame, path = _frame_and_path(project_dir, project, target)
        canonical = next(
            (
                item
                for item in project.get("evidence_image_metadata", [])
                if item.get("image", {}).get("image_id") == target
            ),
            None,
        )
        if canonical is None:
            raise ValidationFailure(f"Canonical metadata mirror is missing for {target}")
        metadata = verify_embedded_metadata(path, canonical)
        if frame.get("latest_revision_id") != metadata.analysis.latest_revision_id:
            raise ValidationFailure(
                f"Frame {target} latest revision disagrees with embedded metadata"
            )
        if frame.get("metadata_payload_digest") != metadata.integrity.payload_digest:
            raise ValidationFailure(
                f"Frame {target} payload digest disagrees with embedded metadata"
            )
        results.append(
            {
                "image_id": target,
                "path": str(path),
                "verified": True,
                "latest_revision_id": metadata.analysis.latest_revision_id,
                "payload_digest": metadata.integrity.payload_digest,
                "pixel_hash": metadata.image.pixel_hash.value,
            }
        )
    return results


def show_packet(project_dir: Path, event_or_frame_id: str) -> dict[str, Any]:
    packet_path = project_dir / ".state" / "vision" / "packets" / f"{event_or_frame_id}.json"
    if packet_path.is_file():
        loaded = json.loads(packet_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValidationFailure(f"Evidence packet is not an object: {packet_path}")
        return cast(dict[str, Any], loaded)
    project = load_project(project_dir)
    frame = next(
        (item for item in project.get("frames", []) if item.get("frame_id") == event_or_frame_id),
        None,
    )
    if frame:
        event_ids = frame.get("linked_event_ids", [])
        if event_ids:
            return show_packet(project_dir, str(event_ids[0]))
    raise InputError(f"Unknown evidence packet event or frame ID: {event_or_frame_id}")


def ingest_project_observation(
    project_dir: Path,
    observation_path: Path,
    *,
    base_revision: str,
    finalize: bool = True,
    project_mutator: Callable[[dict[str, Any], str], None] | None = None,
    incremental_fields: Sequence[str] | None = None,
    project_state: dict[str, Any] | None = None,
    ledger_state: dict[str, Any] | None = None,
    canonical_patch_state: JsonPatchState | None = None,
    canonical_journal: Callable[[dict[str, Any]], None] | None = None,
    defer_ledger: bool = False,
    batch_context: SemanticBatchContext | None = None,
) -> dict[str, Any]:
    """Append one observation and reconcile its metadata transactionally.

    ``finalize`` controls only project-wide bookkeeping.  The image metadata,
    revision ledger, canonical frame mirror, and transaction journal are still
    committed for every observation when it is false.  Batch semantic workers
    use this mode to avoid rendering and validating the entire project once per
    frame, then perform one final global pass after all observations complete.
    The default remains the strict single-observation behavior used by the CLI.
    ``project_mutator`` is an optional in-memory hook invoked immediately
    before the canonical commit with the committed revision ID.  It is used by
    semantic batches to attach links without a second full-project write.
    Internal batch callers may provide ``incremental_fields`` to patch only
    known changed canonical root fields; the default remains a complete atomic
    write for public callers and unexpected mutations.
    ``project_state`` and ``ledger_state`` are internal sequential-batch caches;
    successful commits mutate them in place so the next observation avoids
    reparsing the same large envelopes. They are never required by public CLI
    callers and are safe to discard on restart.
    ``canonical_patch_state`` is a process-local offset cache for the same
    sequential batch; its stat-bound receipt check preserves restart safety.
    ``canonical_journal`` is an internal semantic-batch hook.  When supplied,
    the canonical and ledger deltas are durably journaled per observation and
    materialized by the batch owner once after all provider work completes.
    Public callers keep the historical per-observation atomic writes.
    ``batch_context`` is an internal process-local cache for semantic batches.
    It is populated from canonical state once and is never persisted.
    """
    project_dir = project_dir.resolve(strict=True)
    if project_state is not None and incremental_fields is None:
        raise ValueError("project_state requires incremental_fields for safe batch commits")
    project = project_state if project_state is not None else load_project(project_dir)
    try:
        raw = json.loads(observation_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"Observation file not found: {observation_path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"Observation is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise InputError("Observation root must be a JSON object")
    supplied = raw.get("base_revision_id")
    if supplied is not None and supplied != base_revision:
        raise InputError(
            f"Observation base_revision_id {supplied!r} disagrees with --base-revision {base_revision!r}"
        )
    raw["base_revision_id"] = base_revision
    raw.setdefault("validation_result", "accepted")
    observation = VisualAnalysisObservation.model_validate(raw)
    existing_claim_numbers = [
        int(value.removeprefix("IC"))
        for item in project.get("image_claims", [])
        if (value := str(item.get("claim_id", ""))).startswith("IC")
        and value.removeprefix("IC").isdigit()
    ] if batch_context is None else []
    next_claim_number = (
        max(existing_claim_numbers, default=0) + 1
        if batch_context is None
        else batch_context.max_claim_number + 1
    )
    allocated_claims = []
    for proposed in observation.proposed_claims:
        if proposed.claim_id is None:
            if batch_context is None:
                claim_id = f"IC{next_claim_number:06d}"
                next_claim_number += 1
            else:
                claim_id = batch_context.allocate_claim_id()
            proposed = proposed.model_copy(update={"claim_id": claim_id})
        allocated_claims.append(proposed)
    observation = observation.model_copy(update={"proposed_claims": allocated_claims})
    if not observation.image_ids:
        raise InputError("Observation must inspect at least one image")
    primary_id = observation.image_ids[0]
    frame, image_path = _frame_and_path(
        project_dir,
        project,
        primary_id,
        frame_index=batch_context.frames_by_id if batch_context is not None else None,
    )
    current_metadata = read_embedded_metadata(image_path)
    if batch_context is None:
        observations = [
            VisualAnalysisObservation.model_validate(item)
            for item in project.get("visual_observations", [])
        ]
        revisions = [
            ImageMetadataRevision.model_validate(item)
            for item in project.get("metadata_revisions", [])
        ]
    else:
        observations = batch_context.observations
        revisions = batch_context.revisions

    transaction_path = project_dir / ".state" / "checkpoints" / "metadata-transaction.json"
    atomic_write_json(
        transaction_path,
        {
            "phase": "observation_validated",
            "observation_id": observation.observation_id,
            "image_id": primary_id,
            "base_revision": base_revision,
            "previous_revision": current_metadata.analysis.latest_revision_id,
        },
    )
    ingestion = ingest_observation(
        current_metadata,
        observations,
        revisions,
        observation,
        allow_stale_reconcile=True,
        batch_context=batch_context,
    )
    # Re-evaluate the exact image/block/event use after every accepted semantic
    # observation.  A larger claim payload is not itself sufficient; the
    # question-scoped rule decides whether the new evidence can be consumed.
    from .ids import sequential_id
    from .image_metadata import prepare_metadata_payload

    if batch_context is None:
        existing_decisions = [
            item
            for item in project.get("sufficiency_decisions", [])
            if primary_id in item.get("image_ids", [])
        ]
        latest_decision = existing_decisions[-1] if existing_decisions else None
    else:
        latest_decision = batch_context.latest_decision_by_image.get(primary_id)
    if latest_decision is not None:
        questions = [
            EvidenceQuestion.model_validate(question)
            for question in latest_decision.get("questions", [])
        ]
    else:
        question_id = (
            observation.targeted_question_ids[0] if observation.targeted_question_ids else "Q000001"
        )
        questions = [
            EvidenceQuestion(
                question_id=question_id,
                question="What meaningful visible state or exact text supports the linked reconstruction?",
                importance="supporting",
                required_precision="observable visible state and consequential text",
                modality="visual",
                candidate_claim_ids=[],
            )
        ]
    if not questions:
        questions = [
            EvidenceQuestion(
                question_id="Q000001",
                question="What meaningful visible state or exact text supports the linked reconstruction?",
                importance="supporting",
                required_precision="observable visible state and consequential text",
                modality="visual",
                candidate_claim_ids=[],
            )
        ]
    claim_ids = [claim.claim_id for claim in ingestion.metadata.knowledge.claims]
    questions = [
        question.model_copy(
            update={
                "candidate_claim_ids": list(
                    dict.fromkeys([*question.candidate_claim_ids, *claim_ids])
                )
            }
        )
        for question in questions
    ]
    if batch_context is None:
        existing_ids = [
            str(item.get("decision_id")) for item in project.get("sufficiency_decisions", [])
        ]
        greatest_decision_number = max(
            (
                int(value.removeprefix("MS"))
                for value in existing_ids
                if value.startswith("MS") and value.removeprefix("MS").isdigit()
            ),
            default=0,
        )
        decision_id = sequential_id("metadata_sufficiency", greatest_decision_number + 1)
    else:
        decision_id = batch_context.allocate_decision_id()
    if batch_context is None:
        linked_block_ids = [
            str(block.get("block_id"))
            for block in project.get("script_blocks", [])
            if primary_id in block.get("frame_ids", [])
        ]
        linked_event_ids = [
            str(event.get("event_id"))
            for event in project.get("visual_events", [])
            if primary_id in event.get("evidence_frame_ids", [])
        ]
    else:
        linked_block_ids = [
            str(block.get("block_id"))
            for block in batch_context.blocks_by_frame_id.get(primary_id, [])
            if block.get("block_id") is not None
        ]
        linked_event_ids = [
            str(event.get("event_id"))
            for event in batch_context.events_by_frame_id.get(primary_id, [])
            if event.get("event_id") is not None
        ]
    sufficiency_decision = evaluate_sufficiency(
        decision_id=decision_id,
        questions=questions,
        claims=ingestion.metadata.knowledge.claims,
        observations=ingestion.observations,
        image_ids=[primary_id],
        metadata_revision_ids=[ingestion.revision.revision_id],
        visual_event_ids=linked_event_ids,
        script_block_ids=linked_block_ids,
        payload_current_and_valid=True,
        # This observation already inspected the packet's supplied pixels. If it
        # still leaves a gap, evaluate_sufficiency supplies the next targeted
        # action; a sufficient decision must never retain an unattempted action.
        unattempted_evidence_actions=[],
        semantic_observer_available=True,
        decided_by="metadata-sufficiency-rule-v1",
    )
    # Review identifiers must be present in the image envelope that is about to
    # be written.  The review records are appended to canonical state below,
    # but embedding first and linking later would leave the portable evidence
    # payload unable to navigate to a newly-created high-impact/dispute gate.
    review_items = project.setdefault("review_items", [])
    existing_review_claim_ids = (
        set(batch_context.guarded_review_claim_ids)
        if batch_context is not None
        else {
            str(claim_id)
            for review in review_items
            for claim_id in review.get("image_claim_ids", [])
            if review.get("category") in {"disputed_image_claim", "high_impact_image_claim"}
        }
    )
    if batch_context is None:
        review_numbers = [
            int(str(item.get("review_id", "R0")).removeprefix("R"))
            for item in review_items
            if str(item.get("review_id", "")).startswith("R")
            and str(item.get("review_id", "")).removeprefix("R").isdigit()
        ]
        next_review_number = max(review_numbers, default=0)
    else:
        next_review_number = batch_context.max_review_number
    planned_review_ids: list[str] = []
    relevant_review_ids = [
        str(item.get("review_id"))
        for item in (
            batch_context.reviews_by_frame_id.get(primary_id, [])
            if batch_context is not None
            else review_items
        )
        if (batch_context is not None or primary_id in [str(value) for value in item.get("frame_ids", [])])
        and item.get("review_id")
    ]
    for claim in ingestion.metadata.knowledge.claims:
        guarded = (
            claim.status == "disputed"
            or claim.high_impact_token
            or claim.importance == "high_impact"
        )
        if not guarded:
            continue
        matching = [
            str(item.get("review_id"))
            for item in (
                batch_context.reviews_by_claim_id.get(claim.claim_id, [])
                if batch_context is not None
                else review_items
            )
            if (batch_context is not None or claim.claim_id in item.get("image_claim_ids", []))
            and item.get("category") in {"disputed_image_claim", "high_impact_image_claim"}
            and item.get("review_id")
        ]
        if matching:
            relevant_review_ids.extend(matching)
        elif claim.claim_id not in existing_review_claim_ids:
            next_review_number += 1
            planned_review_ids.append(sequential_id("review", next_review_number))
    metadata_raw = ingestion.metadata.model_dump(mode="json")
    metadata_raw["links"]["review_item_ids"] = list(
        dict.fromkeys(
            [
                *metadata_raw.get("links", {}).get("review_item_ids", []),
                *relevant_review_ids,
                *planned_review_ids,
            ]
        )
    )
    metadata_raw["analysis"]["sufficiency"] = {
        "status": sufficiency_decision.status,
        "evaluated_question_ids": [question.question_id for question in questions],
        "answered_question_ids": sufficiency_decision.answered_question_ids,
        "unanswered_questions": [
            f"{question.question_id}: {question.question}"
            for question in questions
            if question.question_id in sufficiency_decision.unanswered_question_ids
        ],
        "recommended_next_action": sufficiency_decision.recommended_next_action,
    }
    prepared_ingestion_metadata = prepare_metadata_payload(metadata_raw)
    ingestion = replace(
        ingestion,
        metadata=prepared_ingestion_metadata,
        revision=ingestion.revision.model_copy(
            update={"new_payload_digest": prepared_ingestion_metadata.integrity.payload_digest}
        ),
    )
    atomic_write_json(
        transaction_path,
        {
            "phase": "revision_prepared",
            "observation_id": observation.observation_id,
            "image_id": primary_id,
            "base_revision": base_revision,
            "new_revision": ingestion.revision.revision_id,
        },
    )
    # Decode once before the transactional rewrite.  The creation envelope is
    # already canonical and its IDAT stream is copied byte-for-byte by the
    # internal fast path, so decoding the same large PNG again before/after the
    # write would add no new evidence.  Public metadata writes retain their
    # independent full-decode defaults.
    before_pixels = normalized_pixel_hash(image_path)
    if current_metadata.image.pixel_hash.value != before_pixels:
        raise ValidationFailure("Stored metadata pixel hash disagrees with decoded pixels")
    embedded, embedded_file_hash = embed_metadata_with_file_hash(
        image_path,
        ingestion.metadata,
        verify_source_pixels=False,
        verify_decoded_pixels=False,
    )
    verify_embedded_metadata(image_path, embedded, expected_pixel_hash=before_pixels)
    after_pixels = before_pixels
    committed_revision = mark_revision_committed(ingestion.revision)
    atomic_write_json(
        transaction_path,
        {
            "phase": "image_verified",
            "observation_id": observation.observation_id,
            "image_id": primary_id,
            "new_revision": committed_revision.revision_id,
            "pixel_hash": after_pixels,
        },
    )

    accepted = ingestion.observations[-1]
    if batch_context is None:
        project["visual_observations"] = [
            item.model_dump(mode="json") for item in ingestion.observations
        ]
    else:
        # The parsed history lives in the process-local context.  Persist only
        # the newly accepted append; rebuilding every prior model dump is an
        # avoidable O(N^2) hot-loop cost.
        batch_context.commit(accepted, committed_revision)
        project.setdefault("visual_observations", []).append(accepted.model_dump(mode="json"))
    project["metadata_revisions"].append(committed_revision.model_dump(mode="json"))
    payloads = project.setdefault("evidence_image_metadata", [])
    embedded_payload = embedded.model_dump(mode="json")
    payload_index = (
        batch_context.payload_index_by_image_id.get(primary_id)
        if batch_context is not None
        else None
    )
    if payload_index is not None and 0 <= payload_index < len(payloads):
        payloads[payload_index] = embedded_payload
    else:
        # Public callers retain the historical linear replacement behavior for
        # legacy projects whose payload index has not been built.
        replaced = False
        for index, payload in enumerate(payloads):
            if payload.get("image", {}).get("image_id") == primary_id:
                payloads[index] = embedded_payload
                payload_index = index
                replaced = True
                break
        if not replaced:
            payloads.append(embedded_payload)
            payload_index = len(payloads) - 1
        if batch_context is not None:
            assert payload_index is not None
            batch_context.record_payload(primary_id, payload_index)
    if batch_context is None:
        by_claim = {str(item.get("claim_id")): item for item in project.get("image_claims", [])}
        for claim in embedded.knowledge.claims:
            by_claim[claim.claim_id] = claim.model_dump(mode="json")
        project["image_claims"] = [by_claim[key] for key in sorted(by_claim)]
    else:
        # Claim IDs are allocated monotonically, so new IDs append after the
        # canonical sorted prefix. Existing IDs are replaced at their stable
        # index. This preserves ordering without rebuilding/sorting all claims
        # for every observation.
        image_claims = project.setdefault("image_claims", [])
        for claim in embedded.knowledge.claims:
            payload = claim.model_dump(mode="json")
            claim_index = batch_context.claim_index_by_id.get(claim.claim_id)
            if claim_index is None:
                claim_index = len(image_claims)
                last_claim_id = (
                    str(image_claims[-1].get("claim_id", ""))
                    if image_claims and isinstance(image_claims[-1], dict)
                    else ""
                )
                if last_claim_id and claim.claim_id < last_claim_id:
                    # Provider-generated IDs are normally monotonic. Preserve
                    # the canonical sort invariant for an explicitly supplied
                    # out-of-order ID without penalizing the common append path.
                    image_claims.append(payload)
                    image_claims.sort(key=lambda item: str(item.get("claim_id", "")))
                    batch_context.claim_index_by_id = {
                        str(item.get("claim_id")): index
                        for index, item in enumerate(image_claims)
                        if item.get("claim_id") is not None
                    }
                else:
                    image_claims.append(payload)
                    batch_context.claim_index_by_id[claim.claim_id] = claim_index
            else:
                image_claims[claim_index] = payload
    project.setdefault("sufficiency_decisions", []).append(
        sufficiency_decision.model_dump(mode="json")
    )
    frame.update(
        {
            "latest_revision_id": embedded.analysis.latest_revision_id,
            "metadata_payload_digest": embedded.integrity.payload_digest,
            "metadata_sufficiency_state": embedded.analysis.sufficiency.status,
            "file_hash": embedded_file_hash,
            "supported_claim_ids": embedded.knowledge.supported_claim_ids,
            "disputed_claim_ids": embedded.knowledge.disputed_claim_ids,
            "unresolved_claim_ids": embedded.knowledge.unresolved_claim_ids,
            "description": embedded.knowledge.current_factual_description
            or "Visual evidence retained; semantic description pending review.",
            "metadata": embedded.model_dump(mode="json"),
        }
    )
    decision_blocks = (
        batch_context.blocks_by_frame_id.get(primary_id, [])
        if batch_context is not None
        else project.get("script_blocks", [])
    )
    for block in decision_blocks:
        if primary_id not in block.get("frame_ids", []):
            continue
        decision_ids = block.setdefault("metadata_sufficiency_decision_ids", [])
        if decision_id not in decision_ids:
            decision_ids.append(decision_id)
    decision_reviews = (
        batch_context.reviews_by_frame_id.get(primary_id, [])
        if batch_context is not None
        else project.get("review_items", [])
    )
    for review in decision_reviews:
        if primary_id not in review.get("frame_ids", []):
            continue
        decision_ids = review.setdefault("sufficiency_decision_ids", [])
        if decision_id not in decision_ids:
            decision_ids.append(decision_id)
    # A disputed or high-impact claim is never silently left as metadata-only
    # state.  Surface a targeted review item with the competing statements and
    # the exact independent-check action required to consume it.
    review_items = project.setdefault("review_items", [])
    existing_review_claim_ids = (
        set(batch_context.guarded_review_claim_ids)
        if batch_context is not None
        else {
            str(claim_id)
            for review in review_items
            for claim_id in review.get("image_claim_ids", [])
            if review.get("category") in {"disputed_image_claim", "high_impact_image_claim"}
        }
    )
    if batch_context is None:
        review_numbers = [
            int(str(item.get("review_id", "R0")).removeprefix("R"))
            for item in review_items
            if str(item.get("review_id", "")).startswith("R")
            and str(item.get("review_id", "")).removeprefix("R").isdigit()
        ]
        next_review_number = max(review_numbers, default=0)
    else:
        next_review_number = batch_context.max_review_number
    for claim in embedded.knowledge.claims:
        guarded = (
            claim.status == "disputed"
            or claim.high_impact_token
            or claim.importance == "high_impact"
        )
        if not guarded or claim.claim_id in existing_review_claim_ids:
            continue
        next_review_number += 1
        review_id = sequential_id("review", next_review_number)
        # Keep alternatives supplied by the observer; do not repeat a disputed
        # statement as if it were current Markdown fact.
        alternatives = list(dict.fromkeys(claim.alternatives))
        review = {
                "review_id": review_id,
                "severity": "high"
                if claim.high_impact_token or claim.importance == "high_impact"
                else "medium",
                "category": (
                    "high_impact_image_claim"
                    if claim.high_impact_token or claim.importance == "high_impact"
                    else "disputed_image_claim"
                ),
                "start_ms": frame.get("actual_ms"),
                "end_ms": frame.get("actual_ms"),
                "block_ids": linked_block_ids,
                "segment_ids": [
                    str(segment_id)
                    for block in decision_blocks
                    if str(block.get("block_id")) in linked_block_ids
                    for segment_id in block.get("transcript_segment_ids", [])
                ],
                "event_ids": linked_event_ids,
                "frame_ids": [primary_id],
                "ocr_observation_ids": claim.ocr_observation_ids,
                "image_claim_ids": [claim.claim_id],
                "metadata_revision_ids": [ingestion.revision.revision_id],
                "sufficiency_decision_ids": [decision_id],
                "problem": (
                    "A high-impact image claim lacks the required independent blind check."
                    if claim.high_impact_token or claim.importance == "high_impact"
                    else "Credible image observations disagree about this visible claim."
                ),
                "alternatives": alternatives,
                "required_action": "Run an independent blind visual pass and reconcile the claim before consuming it.",
                "blocking": bool(
                    claim.high_impact_token or claim.importance in {"consequential", "high_impact"}
                ),
                "decision": None,
                "reviewer": None,
                "decision_timestamp_utc": None,
                "rationale": None,
            }
        review_items.append(review)
        if batch_context is not None:
            batch_context.record_reviews([review])
    if batch_context is not None:
        batch_context.record_claims(embedded.knowledge.claims)
        batch_context.record_decision(
            primary_id,
            sufficiency_decision.model_dump(mode="json"),
        )
        batch_context.max_review_number = max(
            batch_context.max_review_number,
            next_review_number,
        )
    all_observations = ingestion.observations
    consumable_claims = [
        claim
        for claim in embedded.knowledge.claims
        if claim.status == "supported"
        and (
            not claim_requires_independent_check(claim)
            or claim_has_independent_support(claim, all_observations)
        )
    ]
    for block in decision_blocks:
        if primary_id not in block.get("frame_ids", []):
            continue
        revision_ids = block.setdefault("metadata_revision_ids", [])
        if embedded.analysis.latest_revision_id not in revision_ids:
            revision_ids.append(embedded.analysis.latest_revision_id)
        current_image_claim_ids = {claim.claim_id for claim in embedded.knowledge.claims}
        claim_ids = [
            claim_id
            for claim_id in block.setdefault("image_claim_ids", [])
            if claim_id not in current_image_claim_ids
        ]
        for claim_id in [claim.claim_id for claim in consumable_claims]:
            if claim_id not in claim_ids:
                claim_ids.append(claim_id)
        block["image_claim_ids"] = claim_ids
        if consumable_claims:
            block["visual_description"] = " ".join(
                claim.statement.strip() for claim in consumable_claims
            )
        else:
            block["visual_description"] = (
                "[visual evidence retained; semantic description pending review]"
            )
    ledger_path = project_dir / ".state" / "vision" / "image-observations.json"
    if ledger_state is None:
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            ledger = {
                "schema_version": "1.0",
                "payloads": [],
                "payload_history": [],
                "observations": [],
                "claims": [],
                "revisions": [],
            }
        if not isinstance(ledger, dict):
            ledger = {
                "schema_version": "1.0",
                "payloads": [],
                "payload_history": [],
                "observations": [],
                "claims": [],
                "revisions": [],
            }
    else:
        ledger = ledger_state
    ledger.setdefault("payload_history", []).append(current_metadata.model_dump(mode="json"))
    ledger["payloads"] = payloads
    ledger["observations"] = project["visual_observations"]
    ledger["claims"] = project["image_claims"]
    ledger["revisions"] = project["metadata_revisions"]
    # Batch semantic callers can attach their event/block/review links while
    # this already-loaded project is in memory.  The callback runs before the
    # single canonical write below, so links remain part of the same
    # per-observation durable commit instead of forcing a second full JSON
    # read/serialize/write cycle.
    if project_mutator is not None:
        project_mutator(project, committed_revision.revision_id)
    if canonical_journal is not None:
        # Keep the per-observation durable record deliberately delta-shaped.
        # Large immutable arrays (transcript, frames, OCR) are never copied
        # wholesale; recovery upserts only the objects touched by this image.
        if batch_context is None:
            linked_blocks = {
                str(block.get("block_id")): block
                for block in project.get("script_blocks", [])
                if block.get("block_id") is not None
                and str(block.get("block_id")) in linked_block_ids
            }
            linked_events = {
                str(event.get("event_id")): event
                for event in project.get("visual_events", [])
                if event.get("event_id") is not None
                and str(event.get("event_id")) in linked_event_ids
            }
        else:
            linked_blocks = {
                block_id: batch_context.blocks_by_id[block_id]
                for block_id in linked_block_ids
                if block_id in batch_context.blocks_by_id
            }
            linked_events = {
                event_id: batch_context.events_by_id[event_id]
                for event_id in linked_event_ids
                if event_id in batch_context.events_by_id
            }
        linked_reviews: dict[str, dict[str, Any]] = {}
        claim_ids_for_reviews = {
            str(claim.claim_id) for claim in embedded.knowledge.claims
        }
        if batch_context is None:
            review_candidates = project.get("review_items", [])
            for review in review_candidates:
                if not isinstance(review, dict) or review.get("review_id") is None:
                    continue
                review_frame_ids = {str(value) for value in review.get("frame_ids", [])}
                review_event_ids = {str(value) for value in review.get("event_ids", [])}
                review_claim_ids = {str(value) for value in review.get("image_claim_ids", [])}
                if (
                    primary_id in review_frame_ids
                    or review_event_ids.intersection(linked_event_ids)
                    or review_claim_ids.intersection(claim_ids_for_reviews)
                    or str(decision_id)
                    in {str(value) for value in review.get("sufficiency_decision_ids", [])}
                ):
                    linked_reviews[str(review["review_id"])] = review
        else:
            related_review_ids: set[str] = set()
            for review in batch_context.reviews_by_frame_id.get(primary_id, []):
                if review.get("review_id") is not None:
                    related_review_ids.add(str(review["review_id"]))
            for event_id in linked_event_ids:
                for review in batch_context.reviews_by_event_id.get(event_id, []):
                    if review.get("review_id") is not None:
                        related_review_ids.add(str(review["review_id"]))
            for claim_id in claim_ids_for_reviews:
                for review in batch_context.reviews_by_claim_id.get(claim_id, []):
                    if review.get("review_id") is not None:
                        related_review_ids.add(str(review["review_id"]))
            for review in batch_context.reviews_by_decision_id.get(str(decision_id), []):
                if review.get("review_id") is not None:
                    related_review_ids.add(str(review["review_id"]))
            linked_reviews = {
                review_id: review
                for review_id, review in batch_context.reviews_by_id.items()
                if review_id in related_review_ids
            }
        metadata_item = (
            payloads[batch_context.payload_index_by_image_id[primary_id]]
            if batch_context is not None
            and primary_id in batch_context.payload_index_by_image_id
            and 0 <= batch_context.payload_index_by_image_id[primary_id] < len(payloads)
            else next(
                (
                    item
                    for item in payloads
                    if item.get("image", {}).get("image_id") == primary_id
                ),
                None,
            )
        )
        if metadata_item is None:
            raise ValidationFailure(f"Embedded metadata payload is absent: {primary_id}")
        canonical_journal(
            {
                "observation": accepted.model_dump(mode="json"),
                "metadata_revision": committed_revision.model_dump(mode="json"),
                "sufficiency_decision": sufficiency_decision.model_dump(mode="json"),
                "payload_history": current_metadata.model_dump(mode="json"),
                "payload": metadata_item,
                "claims": [claim.model_dump(mode="json") for claim in embedded.knowledge.claims],
                "frames": {str(frame.get("frame_id") or frame.get("image_id")): frame},
                "evidence_image_metadata": {primary_id: metadata_item},
                "script_blocks": linked_blocks,
                "visual_events": linked_events,
                "review_items": linked_reviews,
            }
        )
    # The image ledger is machine-consumed state and can contain thousands of
    # payload/revision records on long videos.  Compact JSON avoids repeating
    # indentation whitespace on every semantic commit; parsed content and
    # canonical revision digests are unchanged.
    if not defer_ledger:
        atomic_write_json(ledger_path, ledger, compact=True)
    if finalize:
        project["audit"] = audit_project(project)
        project["project_status"] = project["audit"]["final_project_status"]
    canonical_path = project_dir / ".state" / "canonical-project.json"
    if canonical_journal is not None:
        # The journal entry above is the durable commit for this observation;
        # the batch owner materializes the canonical project once at the end.
        pass
    elif incremental_fields is None:
        atomic_write_json(
            canonical_path,
            project,
            compact=canonical_compact_for_payload(canonical_path, project),
        )
    else:
        updates = {key: project[key] for key in incremental_fields if key in project}
        # Most semantic observations touch one frame and its linked event/block
        # records.  Patch those objects in place instead of re-encoding the
        # complete multi-megabyte root arrays.  Appended/reconciled arrays stay
        # root-field updates because their ordering and membership changed.
        array_updates: dict[str, tuple[str, dict[str, Any]]] = {}
        frame_identity_path = "frame_id" if frame.get("frame_id") is not None else "image_id"
        frame_identity = frame.get(frame_identity_path)
        if frame_identity is not None:
            array_updates["frames"] = (frame_identity_path, {str(frame_identity): frame})
        metadata_item = (
            payloads[batch_context.payload_index_by_image_id[primary_id]]
            if batch_context is not None
            and primary_id in batch_context.payload_index_by_image_id
            and 0 <= batch_context.payload_index_by_image_id[primary_id] < len(payloads)
            else next(
                (
                    item
                    for item in payloads
                    if item.get("image", {}).get("image_id") == primary_id
                ),
                None,
            )
        )
        if metadata_item is not None:
            array_updates["evidence_image_metadata"] = (
                "image.image_id",
                {primary_id: metadata_item},
            )
        linked_blocks = (
            {
                block_id: batch_context.blocks_by_id[block_id]
                for block_id in linked_block_ids
                if block_id in batch_context.blocks_by_id
            }
            if batch_context is not None
            else {
                str(block.get("block_id")): block
                for block in project.get("script_blocks", [])
                if block.get("block_id") is not None
                and str(block.get("block_id")) in linked_block_ids
            }
        )
        if linked_blocks:
            array_updates["script_blocks"] = ("block_id", linked_blocks)
        linked_events = (
            {
                event_id: batch_context.events_by_id[event_id]
                for event_id in linked_event_ids
                if event_id in batch_context.events_by_id
            }
            if batch_context is not None
            else {
                str(event.get("event_id")): event
                for event in project.get("visual_events", [])
                if event.get("event_id") is not None
                and str(event.get("event_id")) in linked_event_ids
            }
        )
        if linked_events:
            array_updates["visual_events"] = ("event_id", linked_events)
        for key in array_updates:
            updates.pop(key, None)
        atomic_update_json_fields(
            canonical_path,
            updates,
            fallback_payload=project,
            array_item_updates=array_updates,
            patch_state=canonical_patch_state,
        )
    markdown = next(project_dir.glob("*.md"), project_dir / f"{project_dir.name}.md")
    if finalize:
        atomic_write_json(project_dir / ".state" / "audit.json", project["audit"])
        render_to_path(project, markdown)
        validation = validate_project(project_dir, use_cached_file_hash=True)
        if not validation.valid:
            raise ValidationFailure(
                "Post-ingestion validation failed: " + "; ".join(validation.errors)
            )
    atomic_write_json(
        transaction_path,
        {
            "phase": "committed",
            "observation_id": accepted.observation_id,
            "image_id": primary_id,
            "new_revision": committed_revision.revision_id,
        },
    )
    transaction_path.unlink(missing_ok=True)
    return {
        "observation_id": accepted.observation_id,
        "image_id": primary_id,
        "previous_revision_id": current_metadata.analysis.latest_revision_id,
        "new_revision_id": committed_revision.revision_id,
        "stale_base_reconciled": committed_revision.stale_base_reconciled,
        "supported_claim_ids": embedded.knowledge.supported_claim_ids,
        "disputed_claim_ids": embedded.knowledge.disputed_claim_ids,
        "sufficiency_decision_id": decision_id,
        "sufficiency_status": sufficiency_decision.status,
        "pixel_invariance_verified": before_pixels == after_pixels,
        "markdown_regenerated": str(markdown),
    }
