from __future__ import annotations

import json

from video_script_reconstructor.subtitle_parse import (
    decode_subtitle_bytes,
    parse_ass,
    parse_json_transcript,
    parse_plain_text,
    parse_srt,
    parse_timestamp,
    parse_vtt,
)


def test_srt_parses_multiline_and_preserves_distinct_states() -> None:
    segments = parse_srt("1\n00:00:01,250 --> 00:00:03.500\nFirst line\nsecond line\n")
    assert len(segments) == 1
    assert segments[0].start_ms == 1250
    assert segments[0].end_ms == 3500
    assert segments[0].raw_text == "First line\nsecond line"
    assert segments[0].human_verified_text is None
    assert segments[0].repaired_text is None


def test_vtt_and_ass_remove_only_container_markup() -> None:
    vtt = parse_vtt(
        "WEBVTT\n\nc1\n00:01.000 --> 00:02.000 align:start\n"
        "<v Narrator><b>Hello</b> &amp; welcome\n"
    )
    ass = parse_ass(
        "[Events]\nFormat: Layer, Start, End, Style, Name, Text\n"
        "Dialogue: 0,0:00:02.00,0:00:03.50,Default,Host,{\\i1}Value, with comma\\Nnext\n"
    )
    assert vtt[0].normalized_text == "Hello & welcome"
    assert vtt[0].speaker_label == "Narrator"
    assert ass[0].normalized_text == "Value, with comma\nnext"
    assert ass[0].speaker_label == "Host"


def test_json_seconds_and_milliseconds_and_word_times() -> None:
    seconds = parse_json_transcript(
        json.dumps(
            {
                "language": "en",
                "segments": [
                    {
                        "start": 1.25,
                        "end": 2,
                        "text": "hello",
                        "words": [{"start": 1.25, "end": 1.5, "word": "hello"}],
                    }
                ],
            }
        )
    )
    milliseconds = parse_json_transcript(
        json.dumps(
            {
                "time_unit": "milliseconds",
                "segments": [{"start": 1250, "end": 2000, "text": "hello"}],
            }
        )
    )
    assert (seconds[0].start_ms, seconds[0].end_ms) == (1250, 2000)
    assert seconds[0].words[0].start_ms == 1250
    assert milliseconds[0].start_ms == 1250


def test_plain_text_supports_timed_lines_and_untimed_paragraphs() -> None:
    timed = parse_plain_text("[00:01] First\n00:02 - Second")
    untimed = parse_plain_text("Paragraph one.\n\nParagraph two.")
    assert [item.start_ms for item in timed] == [1000, 2000]
    assert [item.normalized_text for item in untimed] == ["Paragraph one.", "Paragraph two."]
    assert all(item.start_ms is None for item in untimed)


def test_encoding_and_timestamp_edges_are_explicit() -> None:
    text, encoding = decode_subtitle_bytes("café".encode("cp1252"))
    assert text == "café"
    assert encoding == "cp1252"
    assert parse_timestamp("10:02:03,004") == 36_123_004
