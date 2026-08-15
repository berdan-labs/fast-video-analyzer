from __future__ import annotations

from video_script_reconstructor.subtitle_parse import ParsedTranscriptSegment
from video_script_reconstructor.subtitle_validate import validate_segments


def segment(identifier: str, start: int, end: int, text: str) -> ParsedTranscriptSegment:
    return ParsedTranscriptSegment(identifier, start, end, "fixture", text, text)


def test_validation_detects_range_order_overlap_duplicate_gap_and_rate() -> None:
    segments = [
        segment(
            "a",
            0,
            1000,
            "an implausibly dense caption with far too many words for one second total",
        ),
        segment("b", 900, 1800, "repeat"),
        segment("c", 1700, 2000, "repeat"),
        segment("d", 20_000, 19_000, "bad range"),
        segment("e", 1500, 1600, "non monotonic"),
        segment("f", 10_000, 11_000, "after a long gap"),
    ]
    report = validate_segments(segments, media_duration_ms=25_000, max_gap_ms=5_000)
    codes = {issue.code for issue in report.issues}
    assert {
        "reading_rate",
        "overlap",
        "duplicate",
        "invalid_range",
        "non_monotonic",
        "long_gap",
    } <= codes
    assert report.unreliable_intervals


def test_validation_detects_speech_coverage_placeholder_and_encoding() -> None:
    segments = [segment("a", 0, 500, "..."), segment("b", 600, 1000, "bad � text")]
    report = validate_segments(segments, speech_intervals=[(0, 1000), (5000, 9000)])
    codes = {issue.code for issue in report.issues}
    assert {
        "placeholder",
        "encoding_corruption",
        "missing_speech",
        "speech_activity_mismatch",
    } <= codes
    assert report.metrics["speech_coverage"] < 0.65


def test_validation_detects_progressive_timing_drift() -> None:
    segments = [
        segment("a", 0, 1000, "one"),
        segment("b", 3000, 4000, "two"),
        segment("c", 7000, 8000, "three"),
    ]
    report = validate_segments(
        segments,
        speech_intervals=[(0, 1000), (2000, 3000), (4000, 5000)],
    )
    assert "drift" in {issue.code for issue in report.issues}


def test_validation_treats_whisper_filipino_alias_as_expected_language() -> None:
    item = segment("filipino", 0, 1200, "Kumusta sa inyong lahat")
    item.language = "tl"  # faster-whisper's Filipino language code
    report = validate_segments([item], expected_language="fil")
    assert "language_mismatch" not in {issue.code for issue in report.issues}
