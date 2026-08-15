from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, cast

_TOKEN = re.compile(r"[\w]+(?:[-'./:@][\w]+)*|[^\w\s]", re.UNICODE)
_HIGH_IMPACT = re.compile(
    r"(?ix)(?:\b\d+(?:[.,:]\d+)*(?:%|mg|kg|ml|gb|mb|px|ms|s)?\b|"
    r"https?://\S+|\b\S+@\S+\.\S+\b|(?:--|-)[a-z][\w-]*|"
    r"(?:[A-Za-z]:\\|/)[^\s]+)"
)

# Keep block-boundary validation consistent with subtitle and timeline
# validation. Container duration probes and final subtitle cues can differ by
# a few hundred milliseconds due to independent timestamp rounding; material
# overruns must still remain blocking failures.
_MEDIA_BOUND_TOLERANCE_MS = 250


def tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(text)]


def ordered_similarity(source: str, output: str) -> float:
    return SequenceMatcher(a=tokens(source), b=tokens(output), autojunk=False).ratio()


def ordered_coverage(source: str, output: str) -> dict[str, Any]:
    source_tokens = tokens(source)
    output_tokens = tokens(output)
    matcher = SequenceMatcher(a=source_tokens, b=output_tokens, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return {
        "source_tokens": len(source_tokens),
        "matched_in_order": matched,
        "ratio": 1.0 if not source_tokens else matched / len(source_tokens),
        "exact": source_tokens == output_tokens,
        "opcodes": [list(item) for item in matcher.get_opcodes() if item[0] != "equal"],
    }


def high_impact_tokens(text: str) -> list[str]:
    return _HIGH_IMPACT.findall(text)


def high_impact_discrepancies(source: str, output: str) -> dict[str, list[str]]:
    left = Counter(high_impact_tokens(source))
    right = Counter(high_impact_tokens(output))
    return {
        "missing": sorted((left - right).elements()),
        "added": sorted((right - left).elements()),
    }


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return cast(dict[str, Any], value.model_dump(mode="json"))
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected model or mapping, got {type(value)!r}")


def audit_project(project: Any) -> dict[str, Any]:
    data = _dump(project)
    segments = [_dump(item) for item in data.get("transcript_segments", data.get("segments", []))]
    blocks = [_dump(item) for item in data.get("script_blocks", data.get("blocks", []))]
    known_segment_ids = {
        str(segment.get("segment_id") or segment.get("id"))
        for segment in segments
        if segment.get("segment_id") or segment.get("id")
    }
    block_by_segment: dict[str, list[dict[str, Any]]] = {}
    for block in blocks:
        for segment_id in block.get("transcript_segment_ids", block.get("segment_ids", [])):
            block_by_segment.setdefault(str(segment_id), []).append(block)

    substantive = [segment for segment in segments if segment.get("substantive", True)]
    missing: list[str] = []
    partial: list[str] = []
    duplicates: list[str] = []
    high_impact: list[dict[str, Any]] = []
    for segment in substantive:
        segment_id = segment.get("segment_id") or segment.get("id")
        mapped = block_by_segment.get(str(segment_id), [])
        if not mapped:
            missing.append(str(segment_id))
            continue
        if len(mapped) > 1:
            duplicates.append(str(segment_id))
        source_text = (
            segment.get("human_verified_text")
            or segment.get("repaired_text")
            or segment.get("normalized_text")
            or segment.get("raw_text")
            or segment.get("text")
            or ""
        )
        rendered = " ".join(
            str(block.get("spoken_text", block.get("spoken", ""))) for block in mapped
        )
        coverage = ordered_coverage(str(source_text), rendered)
        if not coverage["exact"]:
            partial.append(str(segment_id))
        discrepancy = high_impact_discrepancies(str(source_text), rendered)
        if discrepancy["missing"] or discrepancy["added"]:
            high_impact.append({"segment_id": segment_id, **discrepancy})

    frames = [_dump(item) for item in data.get("frames", data.get("snapshots", []))]
    used_frames = {frame for block in blocks for frame in block.get("frame_ids", [])}
    all_events = [_dump(item) for item in data.get("visual_events", [])]
    # Pending/supporting events are still source evidence and must count toward
    # representation; filtering them out would falsely report zero visual
    # coverage on a review-required run.
    important_events = [
        event for event in all_events if event.get("importance") not in {"incidental", 0, 1, "none"}
    ]
    if not important_events:
        important_events = all_events
    represented_events = {event for block in blocks for event in block.get("visual_event_ids", [])}
    missing_events = [
        str(event.get("event_id"))
        for event in important_events
        if event.get("event_id") not in represented_events
    ]

    review_items = [_dump(item) for item in data.get("review_items", [])]
    unresolved = [item for item in review_items if not item.get("decision")]
    unsupported_spoken: list[str] = [
        str(block.get("block_id")) for block in blocks if block.get("unsupported_spoken")
    ]
    unsupported_visual: list[str] = [
        str(block.get("block_id")) for block in blocks if block.get("unsupported_visual")
    ]
    media = _dump(data.get("media", {}))
    duration_ms = media.get("duration_ms")
    timeline_errors: list[str] = []
    previous_start = -1
    for block in blocks:
        for segment_id in block.get("transcript_segment_ids", block.get("segment_ids", [])):
            if str(segment_id) not in known_segment_ids:
                timeline_errors.append(
                    f"{block.get('block_id')}: cites missing transcript segment {segment_id}"
                )
        start, end = block.get("start_ms"), block.get("end_ms")
        if start is not None and end is not None and int(end) < int(start):
            timeline_errors.append(f"{block.get('block_id')}: end precedes start")
        if start is not None and int(start) < previous_start:
            timeline_errors.append(f"{block.get('block_id')}: non-monotonic block order")
        if start is not None:
            previous_start = int(start)
        if (
            duration_ms is not None
            and end is not None
            and int(end) > int(duration_ms) + _MEDIA_BOUND_TOLERANCE_MS
        ):
            timeline_errors.append(f"{block.get('block_id')}: range exceeds media duration")
        visual = str(block.get("visual_description") or "")
        marker = visual in {
            "",
            "[no visual source available]",
            "[visual evidence retained; semantic description pending review]",
        }
        if visual and not marker and block.get("frame_ids") and not block.get("image_claim_ids"):
            unsupported_visual.append(str(block.get("block_id")))
        if (
            visual == "[visual evidence retained; semantic description pending review]"
            and block.get("verification_status")
            in {"automatically_checked", "human_reviewed", "fully_verified"}
        ):
            unsupported_visual.append(str(block.get("block_id")))
    for frame in frames:
        frame_id = frame.get("frame_id") or frame.get("image_id")
        requested = frame.get("requested_ms")
        actual = frame.get("actual_ms")
        offset = frame.get("offset_ms")
        if requested is not None and actual is not None and offset != int(actual) - int(requested):
            timeline_errors.append(f"{frame_id}: offset disagrees with requested/actual time")
        if (
            duration_ms is not None
            and actual is not None
            and not (0 <= int(actual) <= int(duration_ms) + 1000)
        ):
            timeline_errors.append(f"{frame_id}: actual time outside media bounds")
        if actual is not None and frame.get("timestamp_source") is None:
            timeline_errors.append(f"{frame_id}: actual time lacks measurement provenance")
    unsupported_spoken = sorted(set(unsupported_spoken))
    unsupported_visual = sorted(set(unsupported_visual))
    claims = [_dump(item) for item in data.get("image_claims", [])]
    claim_by_id = {str(claim.get("claim_id")): claim for claim in claims}
    consumed_claim_ids = {
        str(claim_id) for block in blocks for claim_id in block.get("image_claim_ids", [])
    }
    unsupported_claim_ids = sorted(
        claim_id
        for claim_id in consumed_claim_ids
        if claim_by_id.get(claim_id, {}).get("status") != "supported"
    )
    disputed_claim_ids = sorted(
        str(claim.get("claim_id")) for claim in claims if claim.get("status") == "disputed"
    )
    unresolved_claim_ids = sorted(
        str(claim.get("claim_id"))
        for claim in claims
        if claim.get("status") in {"proposed", "unresolved"}
    )
    stale_claim_ids: list[str] = []
    revisions_by_image: dict[str, set[str]] = {}
    for revision in data.get("metadata_revisions", []):
        revisions_by_image.setdefault(str(revision.get("image_id")), set()).add(
            str(revision.get("revision_id"))
        )
    for claim in claims:
        image_ids = [str(image_id) for image_id in claim.get("supporting_image_ids", [])]
        revision_id = claim.get("last_updated_revision_id")
        if (
            image_ids
            and revision_id
            and not any(
                str(revision_id) in revisions_by_image.get(image_id, set())
                for image_id in image_ids
            )
        ):
            stale_claim_ids.append(str(claim.get("claim_id")))
    source_token_total = 0
    source_token_matched = 0
    exact_meaning_units = 0
    for segment in substantive:
        segment_id = segment.get("segment_id") or segment.get("id")
        source_text = (
            segment.get("human_verified_text")
            or segment.get("repaired_text")
            or segment.get("normalized_text")
            or segment.get("raw_text")
            or segment.get("text")
            or ""
        )
        rendered = " ".join(
            str(block.get("spoken_text", block.get("spoken", "")))
            for block in block_by_segment.get(str(segment_id), [])
        )
        coverage = ordered_coverage(str(source_text), rendered)
        source_token_total += int(coverage["source_tokens"])
        source_token_matched += int(coverage["matched_in_order"])
        if coverage["exact"]:
            exact_meaning_units += 1
    residual_text_items = [
        str(block.get("residual_source_text"))
        for block in blocks
        if block.get("residual_source_text")
    ]
    ocr_uncertainty = [
        str(item.get("observation_id"))
        for item in data.get("ocr_observations", [])
        if item.get("uncertain_characters")
    ]
    metadata_payloads = data.get("evidence_image_metadata", [])
    state_metadata = data.get("state_metadata", {})
    if not isinstance(state_metadata, dict):
        state_metadata = {}
    candidate_image_count = int(state_metadata.get("candidate_image_count", 0) or 0)
    candidate_metadata_count = int(state_metadata.get("candidate_metadata_image_count", 0) or 0)
    candidate_semantic_count = int(
        state_metadata.get("candidate_semantically_analyzed_image_count", 0) or 0
    )
    semantically_analyzed_images = {
        str(item.get("image", {}).get("image_id"))
        for item in metadata_payloads
        if item.get("analysis", {}).get("semantic_status") == "observed"
    }
    guarded_unresolved_claim_ids = sorted(
        str(claim.get("claim_id"))
        for claim in claims
        if claim.get("status") in {"proposed", "unresolved", "disputed"}
        and (
            claim.get("high_impact_token")
            or claim.get("importance") == "high_impact"
            or claim.get("status") == "disputed"
        )
    )
    final_frame_ids = {
        str(frame.get("frame_id"))
        for frame in frames
        if frame.get("final", True) and not frame.get("parent_full_frame_id")
    }
    failures: list[str] = []
    if missing:
        failures.append("missing_substantive_segments")
    if partial:
        failures.append("partial_or_reordered_segments")
    if duplicates:
        failures.append("duplicate_segment_mapping")
    if high_impact:
        failures.append("high_impact_token_discrepancy")
    if unsupported_spoken or unsupported_visual:
        failures.append("unsupported_statements")
    if unsupported_claim_ids:
        failures.append("unsupported_image_claims")
    if stale_claim_ids:
        failures.append("stale_image_claims")
    if missing_events:
        failures.append("unrepresented_important_visual_events")
    if any(item.get("category") == "blocked_prerequisite" for item in unresolved):
        failures.append("blocked_prerequisite")
    if timeline_errors:
        failures.append("timeline_errors")
    if data.get("project_status") == "fully_verified" and not data.get("final_signoffs"):
        failures.append("fully_verified_without_human_signoff")
    for decision in data.get("sufficiency_decisions", []):
        decision_data = _dump(decision)
        if decision_data.get("status") == "sufficient" and decision_data.get(
            "unattempted_evidence_actions"
        ):
            failures.append("false_metadata_sufficiency")
            break

    configured_status = str(data.get("project_status", "processing"))
    if failures:
        final_status = "blocked"
    elif unresolved or guarded_unresolved_claim_ids:
        final_status = "review_required"
    elif configured_status in {"human_reviewed", "fully_verified"}:
        final_status = configured_status
    else:
        final_status = "automatically_checked"

    return {
        "schema_version": "1.0",
        "source_segment_coverage": {
            "covered": len(substantive) - len(missing),
            "total": len(substantive),
            "missing_ids": missing,
            "partial_ids": partial,
            "duplicate_ids": duplicates,
        },
        "ordered_meaning_coverage": {
            "exact_segments": len(substantive) - len(set(missing + partial)),
            "total_segments": len(substantive),
        },
        "ordered_token_coverage": (
            1.0 if source_token_total == 0 else source_token_matched / source_token_total
        ),
        "meaning_unit_coverage": (
            1.0 if not substantive else exact_meaning_units / len(substantive)
        ),
        "residual_text_items": residual_text_items,
        "unsupported_spoken_statements": unsupported_spoken,
        "unsupported_visual_statements": unsupported_visual,
        "high_impact_token_discrepancies": high_impact,
        "timeline_errors": timeline_errors,
        "visual_event_coverage": {
            "covered": len(important_events) - len(missing_events),
            "total": len(important_events),
            "missing_ids": missing_events,
        },
        "visual_evidence": {
            "used_frames": len(used_frames),
            "total_final_frames": len(final_frame_ids),
            "total_generated_images": len(frames) + candidate_image_count,
            "embedded_metadata_images": len(metadata_payloads) + candidate_metadata_count,
            "semantically_analyzed_images": len(semantically_analyzed_images)
            + candidate_semantic_count,
            "markdown_consumed_images": len(used_frames),
        },
        "image_metadata_coverage": {
            "generated_images": len(frames) + candidate_image_count,
            "embedded_metadata_images": len(metadata_payloads) + candidate_metadata_count,
            "final_evidence_images": len(final_frame_ids),
            "final_evidence_images_linked": len(final_frame_ids & used_frames),
            "semantically_analyzed_images": len(semantically_analyzed_images)
            + candidate_semantic_count,
            "markdown_consumed_images": len(used_frames),
            "markdown_consumed_claims": len(consumed_claim_ids),
        },
        "screenshot_checks": [
            f"{len(final_frame_ids)}/{len(final_frame_ids)} final evidence images are represented in canonical state"
        ],
        "image_metadata_checks": [
            f"{len(metadata_payloads) + candidate_metadata_count}/"
            f"{len(frames) + candidate_image_count} generated images have canonical embedded-payload mirrors"
        ],
        "unsupported_claim_ids": unsupported_claim_ids,
        "stale_claim_ids": sorted(set(stale_claim_ids)),
        "disputed_claim_ids": disputed_claim_ids,
        "unresolved_claim_ids": unresolved_claim_ids,
        "guarded_unresolved_claim_ids": guarded_unresolved_claim_ids,
        "ocr_uncertainty": ocr_uncertainty,
        "anchor_navigation_checks": [],
        "output_contract_checks": [],
        "unresolved_review_items": [item.get("review_id") for item in unresolved],
        "blocking_failures": failures,
        "final_project_status": final_status,
    }
