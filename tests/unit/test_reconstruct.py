from __future__ import annotations

from video_script_reconstructor.reconstruct import (
    audit_block_coverage,
    audit_high_impact_tokens,
    audit_ordered_tokens,
    build_lossless_blocks,
)
from video_script_reconstructor.subtitle_parse import ParsedTranscriptSegment


def seg(identifier: str, start: int, end: int, text: str) -> ParsedTranscriptSegment:
    return ParsedTranscriptSegment(
        identifier, start, end, "fixture", text, text, speaker_label="Speaker 1"
    )


def test_lossless_grouping_retains_exact_ids_texts_and_coverage() -> None:
    segments = [
        seg("s1", 0, 1000, "First sentence."),
        seg("s2", 1100, 2000, "Second sentence with 42."),
    ]
    blocks = build_lossless_blocks(segments, group_adjacent=True)
    assert len(blocks) == 1
    assert blocks[0].source_segment_ids == ["s1", "s2"]
    assert blocks[0].source_texts == ["First sentence.", "Second sentence with 42."]
    audit = audit_block_coverage(segments, blocks)
    assert audit.valid
    assert audit.covered_segments == 2


def test_ordered_token_audit_rejects_same_tokens_in_wrong_order() -> None:
    audit = audit_ordered_tokens("Dog bites man.", "Man bites dog.")
    assert not audit.faithful
    assert audit.reordered


def test_ordered_token_and_high_impact_audits_find_mutations() -> None:
    token_audit = audit_ordered_tokens(
        "Run tool --strict with value 42", "Run tool --strict with value 43"
    )
    entity_audit = audit_high_impact_tokens(
        "Run tool --strict with value 42", "Run tool --safe with value 43"
    )
    assert not token_audit.faithful
    assert token_audit.substitutions == (("42", "43"),)
    assert not entity_audit.valid
    assert set(entity_audit.missing) == {"--strict", "42"}
    assert set(entity_audit.added) == {"--safe", "43"}


def test_visual_only_event_outside_speech_becomes_a_block() -> None:
    blocks = build_lossless_blocks(
        [seg("s1", 1000, 2000, "Speech")],
        visual_events=[
            {
                "event_id": "v1",
                "start_ms": 3000,
                "end_ms": 3100,
                "important": True,
                "description": "A supported visual state.",
            }
        ],
    )
    assert [block.block_kind for block in blocks] == ["speech", "visual_only"]
    assert blocks[1].visual_event_ids == ["v1"]
