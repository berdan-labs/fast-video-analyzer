from __future__ import annotations

import pytest

from video_script_reconstructor.diarization import (
    DiarizationError,
    apply_diarization,
    apply_explicit_identity_evidence,
    extract_explicit_self_identifications,
    neutralize_speaker_labels,
    repair_automatic_identity_labels,
)
from video_script_reconstructor.subtitle_parse import ParsedTranscriptSegment


def test_backend_labels_become_neutral_and_overlap_is_preserved() -> None:
    turns, mapping = neutralize_speaker_labels(
        [
            {"turn_id": "t1", "start_ms": 0, "end_ms": 1000, "speaker": "SPEAKER_09"},
            {"turn_id": "t2", "start_ms": 800, "end_ms": 1500, "speaker": "SPEAKER_02"},
        ]
    )
    assert mapping == {"SPEAKER_09": "Speaker 1", "SPEAKER_02": "Speaker 2"}
    assert turns[0].overlaps_turn_ids == ["t2"]
    assert turns[1].overlaps_turn_ids == ["t1"]


def test_assignment_marks_ambiguous_boundary_and_manual_names_need_evidence() -> None:
    segment = ParsedTranscriptSegment("s", 700, 1100, "fixture", "hello", "hello")
    assigned, _, _ = apply_diarization(
        [segment],
        [
            {"start_ms": 0, "end_ms": 900, "speaker": "a"},
            {"start_ms": 900, "end_ms": 1500, "speaker": "b"},
        ],
        boundary_tolerance_ms=10,
    )
    assert assigned[0].speaker_label == "Speaker 1"
    assert "uncertain_speaker_boundary" in assigned[0].uncertainty_items
    with pytest.raises(DiarizationError):
        neutralize_speaker_labels(
            [{"start_ms": 0, "end_ms": 1, "speaker": "a"}], manual_mapping={"a": "Alice"}
        )


def test_explicit_self_identification_is_exact_and_identity_safe() -> None:
    segments = [
        {
            "segment_id": "T1",
            "start_ms": 0,
            "end_ms": 1000,
            "raw_text": "Hi, I'm Coach Princess, isang coach ng Freight Course 101.",
        },
        {
            "segment_id": "T2",
            "start_ms": 1000,
            "end_ms": 2000,
            "raw_text": "I am a coach and I am from Manila.",
        },
        {
            "segment_id": "T3",
            "start_ms": 2000,
            "end_ms": 3000,
            "raw_text": "She said, I'm Princess.",
        },
    ]
    claims = extract_explicit_self_identifications(segments)
    assert len(claims) == 1
    assert claims[0].name == "Coach Princess"
    assert claims[0].segment_id == "T1"
    assert claims[0].pattern == "i_am"
    assert claims[0].quote == "I'm Coach Princess"

    labeled, applied = apply_explicit_identity_evidence(segments, claims)
    assert len(applied) == 1
    assert labeled[0]["speaker_label"] == "Coach Princess"
    assert "speaker_label" not in labeled[1]
    assert "speaker_label" not in labeled[2]


def test_filipino_explicit_name_and_introduction_opt_in() -> None:
    segment = {"segment_id": "T1", "start_ms": 0, "end_ms": 1, "raw_text": "Ako po si Maria Santos."}
    claims = extract_explicit_self_identifications([segment])
    assert claims[0].name == "Maria Santos"
    assert claims[0].pattern == "ako_si"
    variants = extract_explicit_self_identifications(
        [
            {"segment_id": "T3", "raw_text": "Ang pangalan ko ay si Maria Santos."},
            {"segment_id": "T4", "raw_text": "Ako po ay Maria Santos."},
        ]
    )
    assert [(claim.segment_id, claim.name, claim.pattern) for claim in variants] == [
        ("T3", "Maria Santos", "ang_pangalan_ko_ay"),
        ("T4", "Maria Santos", "ako_ay"),
    ]
    assert extract_explicit_self_identifications(
        [{"segment_id": "T2", "raw_text": "This is Maria Santos."}]
    ) == []
    assert extract_explicit_self_identifications(
        [{"segment_id": "T2", "raw_text": "This is Maria Santos."}],
        include_introduction_pattern=True,
    )[0].name == "Maria Santos"


def test_conflicting_explicit_names_remain_uncertain() -> None:
    segment = {
        "segment_id": "T1",
        "raw_text": "I'm Ana and my name is Bea.",
        "uncertainty_items": [],
    }
    labeled, _claims = apply_explicit_identity_evidence([segment])
    assert "speaker_label" not in labeled[0]
    assert "uncertain_speaker_identity" in labeled[0]["uncertainty_items"]


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, si broker.",
        "I'm calling about the load.",
        "I'm having issues with the truck.",
        "I'm using my microphone.",
        "I'm expecting everyone to join.",
        "I am requesting for two hours detention.",
        "I'm just giving you a tip.",
        "I'm showing here a load.",
        "I'm really proud of me.",
    ],
)
def test_conversational_i_am_phrases_are_not_identity_claims(text: str) -> None:
    assert extract_explicit_self_identifications([{"segment_id": "T1", "raw_text": text}]) == []


def test_hypothetical_name_example_is_not_identity_claim() -> None:
    segment = {
        "segment_id": "T1",
        "raw_text": "Use an American name, for example, hi my name is America.",
    }
    assert extract_explicit_self_identifications([segment]) == []


def test_repair_removes_stale_automatic_label_but_keeps_exact_claim() -> None:
    segments = [
        {
            "segment_id": "T1",
            "raw_text": "My name is Princess.",
            "speaker_label": "Princess",
            "verification_status": "automatically_transcribed",
        },
        {
            "segment_id": "T2",
            "raw_text": "I'm calling about the load.",
            "speaker_label": "calling about",
            "verification_status": "automatically_transcribed",
        },
        {
            "segment_id": "T3",
            "raw_text": "Manual mapping.",
            "speaker_label": "Alice",
            "verification_status": "human_reviewed",
        },
    ]
    repaired, corrections = repair_automatic_identity_labels(segments)
    assert repaired[0]["speaker_label"] == "Princess"
    assert repaired[1]["speaker_label"] is None
    assert repaired[2]["speaker_label"] == "Alice"
    assert corrections == [
        {
            "segment_id": "T2",
            "old_label": "calling about",
            "reason": "automatic_identity_label_lacks_current_exact_self_identification",
        }
    ]
