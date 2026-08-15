from __future__ import annotations

from video_script_reconstructor.audit import (
    audit_project,
    high_impact_discrepancies,
    ordered_coverage,
)


def _project(source: str, output: str) -> dict:
    return {
        "media": {"duration_ms": 5_000},
        "transcript_segments": [
            {
                "segment_id": "T000001",
                "raw_text": source,
                "normalized_text": source,
                "substantive": True,
            }
        ],
        "script_blocks": [
            {
                "block_id": "B000001",
                "start_ms": 0,
                "end_ms": 1_000,
                "spoken_text": output,
                "transcript_segment_ids": ["T000001"],
                "frame_ids": [],
                "visual_event_ids": [],
            }
        ],
        "frames": [],
        "review_items": [],
        "visual_events": [],
        "project_status": "processing",
    }


def test_ordered_alignment_rejects_bag_of_words_reordering() -> None:
    coverage = ordered_coverage("Dog bites man.", "Man bites dog.")
    assert coverage["exact"] is False
    assert coverage["ratio"] < 1.0
    audit = audit_project(_project("Dog bites man.", "Man bites dog."))
    assert "partial_or_reordered_segments" in audit["blocking_failures"]


def test_high_impact_number_and_flag_are_exact() -> None:
    discrepancy = high_impact_discrepancies(
        "Use tool --strict with 42.", "Use tool --fast with 43."
    )
    assert discrepancy == {"missing": ["--strict", "42"], "added": ["--fast", "43"]}


def test_full_verification_requires_signoff() -> None:
    project = _project("Exact words.", "Exact words.")
    project["project_status"] = "fully_verified"
    audit = audit_project(project)
    assert "fully_verified_without_human_signoff" in audit["blocking_failures"]


def test_unconsumed_high_impact_image_claim_requires_review_without_blocking_output() -> None:
    project = _project("Exact words.", "Exact words.")
    project["image_claims"] = [
        {
            "claim_id": "IC000001",
            "claim_class": "exact_text",
            "statement": "The displayed value is 42.",
            "status": "proposed",
            "importance": "high_impact",
            "high_impact_token": True,
            "supporting_image_ids": ["F000001"],
            "supporting_observation_ids": ["VA000001"],
        }
    ]
    project["review_items"] = [
        {
            "review_id": "R000001",
            "category": "high_impact_image_claim",
            "blocking": True,
            "decision": None,
        }
    ]
    audit = audit_project(project)
    assert audit["guarded_unresolved_claim_ids"] == ["IC000001"]
    assert audit["blocking_failures"] == []
    assert audit["final_project_status"] == "review_required"


def test_audit_accepts_small_media_boundary_rounding() -> None:
    project = _project("Exact words.", "Exact words.")
    project["script_blocks"][0]["end_ms"] = 5_200

    audit = audit_project(project)

    assert audit["timeline_errors"] == []


def test_audit_rejects_material_media_overrun() -> None:
    project = _project("Exact words.", "Exact words.")
    project["script_blocks"][0]["end_ms"] = 5_251

    audit = audit_project(project)

    assert any("range exceeds media duration" in item for item in audit["timeline_errors"])
