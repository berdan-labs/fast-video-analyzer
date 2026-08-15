from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import audit_project
from .errors import InputError, ValidationFailure
from .render_markdown import render_to_path
from .schemas import CanonicalProject
from .security import atomic_write_json, canonical_compact_for_payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_path(project_dir: Path) -> Path:
    return project_dir / ".state" / "canonical-project.json"


def load_project(project_dir: Path) -> dict[str, Any]:
    path = canonical_path(project_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"Canonical project not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationFailure(f"Invalid canonical project: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationFailure("Canonical project must be a JSON object")
    return data


def save_project(project_dir: Path, project: dict[str, Any]) -> None:
    canonical = canonical_path(project_dir)
    atomic_write_json(
        canonical,
        project,
        compact=canonical_compact_for_payload(canonical, project),
    )
    markdown = next(project_dir.glob("*.md"), None)
    if markdown is None:
        slug = project_dir.name
        markdown = project_dir / f"{slug}.md"
    render_to_path(project, markdown)


def list_review_items(project_dir: Path) -> list[dict[str, Any]]:
    return list(load_project(project_dir).get("review_items", []))


def show_review_item(project_dir: Path, review_id: str) -> dict[str, Any]:
    project = load_project(project_dir)
    frames = {
        str(frame.get("frame_id") or frame.get("image_id")): frame
        for frame in project.get("frames", [])
    }
    claims = {str(claim.get("claim_id")): claim for claim in project.get("image_claims", [])}
    for item in project.get("review_items", []):
        if item.get("review_id") == review_id:
            result = dict(item)
            frame_ids = [str(value) for value in item.get("frame_ids", [])]
            result["time_range_ms"] = {
                "start": item.get("start_ms"),
                "end": item.get("end_ms"),
            }
            result["image_paths"] = [
                str(frame.get("full_frame_path") or frame.get("path"))
                for frame_id in frame_ids
                if (frame := frames.get(frame_id))
                and (frame.get("full_frame_path") or frame.get("path"))
            ]
            result["source_ids"] = {
                "blocks": list(item.get("block_ids", [])),
                "segments": list(item.get("segment_ids", [])),
                "events": list(item.get("event_ids", [])),
                "frames": frame_ids,
                "ocr_observations": list(item.get("ocr_observation_ids", [])),
                "image_claims": list(item.get("image_claim_ids", [])),
                "metadata_revisions": list(item.get("metadata_revision_ids", [])),
                "sufficiency_decisions": list(item.get("sufficiency_decision_ids", [])),
            }
            competing_evidence: list[dict[str, Any]] = []
            for raw_claim_id in item.get("image_claim_ids", []):
                claim_id = str(raw_claim_id)
                claim = claims.get(claim_id)
                if claim is None:
                    continue
                competing_evidence.append(
                    {
                        "claim_id": claim_id,
                        "statement": claim.get("statement"),
                        "status": claim.get("status"),
                        "alternatives": claim.get("alternatives", []),
                    }
                )
            result["competing_evidence"] = competing_evidence
            return result
    raise InputError(f"Unknown review item: {review_id}")


def apply_review(
    project_dir: Path,
    review_id: str,
    *,
    reviewer: str,
    decision: str,
    replacement: str | None,
    rationale: str,
) -> dict[str, Any]:
    if not reviewer.strip() or not rationale.strip():
        raise InputError("Reviewer and rationale are required")
    project = load_project(project_dir)
    item = next(
        (value for value in project.get("review_items", []) if value.get("review_id") == review_id),
        None,
    )
    if item is None:
        raise InputError(f"Unknown review item: {review_id}")
    category = str(item.get("category") or "")
    textual_correction = replacement is not None and category in {
        "transcript",
        "wording",
        "spoken",
    }
    segment_ids = [str(value) for value in item.get("segment_ids", [])]
    if textual_correction and len(segment_ids) != 1:
        raise InputError(
            "A transcript replacement must target exactly one preserved transcript segment"
        )
    old = dict(item)
    item.update(
        {
            "decision": decision,
            "reviewer": reviewer,
            "decision_timestamp_utc": _now(),
            "rationale": rationale,
        }
    )
    if replacement is not None:
        item["replacement"] = replacement
        if textual_correction:
            target_segment_id = segment_ids[0]
            segment = next(
                (
                    value
                    for value in project.get("transcript_segments", [])
                    if str(value.get("segment_id")) == target_segment_id
                ),
                None,
            )
            if segment is None:
                raise ValidationFailure(
                    f"Review item {review_id} cites missing transcript segment {target_segment_id}"
                )
            segment["human_verified_text"] = replacement
            segment["verification_status"] = "human_reviewed"
        for block in project.get("script_blocks", project.get("blocks", [])):
            block_ids = item.get(
                "block_ids", ([item.get("block_id")] if item.get("block_id") else [])
            )
            if block.get("block_id") in block_ids and textual_correction:
                block["spoken_text"] = replacement
                block["verification_status"] = "human_reviewed"
    correction = {
        "target_id": review_id,
        "reviewer": reviewer,
        "decision": decision,
        "replacement": replacement,
        "rationale": rationale,
        "old_value": old,
        "new_value": dict(item),
        "timestamp": item["decision_timestamp_utc"],
    }
    project.setdefault("corrections", []).append(correction)
    project["project_status"] = "human_reviewed"
    project["audit"] = audit_project(project)
    project["project_status"] = project["audit"]["final_project_status"]
    CanonicalProject.model_validate(project)
    atomic_write_json(project_dir / ".state" / "corrections.json", project["corrections"])
    atomic_write_json(project_dir / ".state" / "review-queue.json", project.get("review_items", []))
    atomic_write_json(project_dir / ".state" / "audit.json", project["audit"])
    save_project(project_dir, project)
    from .validate_output import validate_project

    validation = validate_project(project_dir)
    if not validation.valid:
        raise ValidationFailure(
            "Post-correction validation failed: " + "; ".join(validation.errors)
        )
    return correction


def finalize_project(project_dir: Path, *, reviewer: str, rationale: str) -> dict[str, Any]:
    if not reviewer.strip() or not rationale.strip():
        raise InputError("Reviewer and rationale are required")
    project = load_project(project_dir)
    audit = audit_project(project)
    unresolved = [item for item in project.get("review_items", []) if not item.get("decision")]
    guarded_claims = audit.get("guarded_unresolved_claim_ids", [])
    if audit.get("blocking_failures") or unresolved or guarded_claims:
        raise ValidationFailure("Finalization gates are not satisfied")
    signoff = {"reviewer": reviewer, "rationale": rationale, "timestamp": _now()}
    project.setdefault("final_signoffs", []).append(signoff)
    project["project_status"] = "fully_verified"
    audit["final_project_status"] = "fully_verified"
    project["audit"] = audit
    CanonicalProject.model_validate(project)
    atomic_write_json(project_dir / ".state" / "audit.json", audit)
    save_project(project_dir, project)
    from .validate_output import validate_project

    validation = validate_project(project_dir)
    if not validation.valid:
        raise ValidationFailure(
            "Post-finalization validation failed: " + "; ".join(validation.errors)
        )
    return signoff
