from __future__ import annotations

from video_script_reconstructor.subtitle_parse import ParsedTranscriptSegment
from video_script_reconstructor.timeline import build_timeline, validate_timeline


def test_timeline_orders_modalities_by_time_and_retains_untimed_records() -> None:
    timed = ParsedTranscriptSegment("timed", 1000, 2000, "source", "spoken", "spoken")
    untimed = ParsedTranscriptSegment(
        "untimed", None, None, "untimed_source", "appendix", "appendix"
    )
    timeline = build_timeline(
        [timed, untimed],
        chapters=[{"chapter_id": "c", "start_ms": 0, "end_ms": 3000}],
        visual_events=[{"event_id": "v", "start_ms": 500, "end_ms": 500}],
    )
    assert [item.kind for item in timeline] == [
        "chapter",
        "visual_event",
        "transcript_segment",
        "transcript_segment",
    ]
    assert timeline[-1].source_id == "untimed"
    report = validate_timeline(timeline, media_duration_ms=3000)
    assert report.valid
    assert (report.timed_count, report.untimed_count) == (3, 1)


def test_timeline_validation_rejects_unsorted_and_out_of_bounds() -> None:
    timeline = build_timeline(
        [],
        visual_events=[
            {"event_id": "first", "start_ms": 100, "end_ms": 100},
            {"event_id": "second", "start_ms": 200, "end_ms": 200},
        ],
    )
    reversed_timeline = list(reversed(timeline))
    report = validate_timeline(reversed_timeline, media_duration_ms=150)
    assert not report.valid
    assert any("chronological" in error for error in report.errors)
    assert any("duration" in error for error in report.errors)


def test_timeline_clips_small_media_boundary_rounding_without_mutating_payload() -> None:
    segment = ParsedTranscriptSegment("tail", 900, 1_200, "source_srt", "Bye", "Bye")
    timeline = build_timeline([segment], media_duration_ms=1_000)

    assert timeline[0].end_ms == 1_000
    assert timeline[0].payload.end_ms == 1_200
    assert validate_timeline(timeline, media_duration_ms=1_000).valid


def test_timeline_keeps_material_media_overrun_invalid() -> None:
    timeline = build_timeline(
        [ParsedTranscriptSegment("tail", 900, 1_251, "source_srt", "Bye", "Bye")],
        media_duration_ms=1_000,
    )

    report = validate_timeline(timeline, media_duration_ms=1_000)
    assert not report.valid
    assert any("duration" in error for error in report.errors)
