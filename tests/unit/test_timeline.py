from __future__ import annotations

from typing import Any

import pytest

from video_script_reconstructor.subtitle_parse import ParsedTranscriptSegment
from video_script_reconstructor.timeline import (
    TimelineItem,
    build_timeline,
    make_timeline_item,
    validate_timeline,
)


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


def _item(
    timeline_id: str,
    start_ms: int | None,
    end_ms: int | None = None,
    *,
    kind: str = "transcript_segment",
    source_id: str = "src",
    timing_provenance: str = "source_timing",
    payload: Any = None,
    source_order: int = 0,
) -> TimelineItem:
    return TimelineItem(
        timeline_id=timeline_id,
        kind=kind,
        source_id=source_id,
        start_ms=start_ms,
        end_ms=end_ms,
        timing_provenance=timing_provenance,
        payload=payload if payload is not None else {},
        source_order=source_order,
    )


def test_validate_timeline_accepts_sorted_valid_timeline() -> None:
    timeline = [
        _item("a", 0, 10),
        _item("b", 10, 20),
        _item("c", None, None, timing_provenance="untimed"),
    ]
    report = validate_timeline(timeline)
    assert report.valid
    assert report.errors == []
    assert report.warnings == []
    assert (report.timed_count, report.untimed_count) == (2, 1)


@pytest.mark.parametrize("position", ["beginning", "middle", "end"])
def test_validate_timeline_flags_adjacent_inversion_at_each_position(position: str) -> None:
    base = [
        _item("a", 0, 5),
        _item("b", 10, 15),
        _item("c", 20, 25),
        _item("d", 30, 35),
        _item("e", 40, 45),
    ]
    index = {"beginning": 0, "middle": 2, "end": 3}[position]
    swapped = list(base)
    swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
    report = validate_timeline(swapped)
    assert not report.valid
    assert report.errors[-1] == "timeline items are not in canonical chronological order"


def test_validate_timeline_reports_duplicate_ids_per_repeat() -> None:
    duplicate = _item("dup", 10, 20)
    timeline = [_item("first", 0, 5), duplicate, duplicate]
    report = validate_timeline(timeline)
    assert not report.valid
    # One error per occurrence after the first.
    assert report.errors == ["duplicate timeline ID: dup"]
    assert (report.timed_count, report.untimed_count) == (3, 0)


def test_validate_timeline_untimed_provenance_and_skip_semantics() -> None:
    sorted_with_untimed_tail = [
        _item("timed", 0, 10),
        _item("appendix", None, None, timing_provenance="untimed"),
    ]
    report = validate_timeline(sorted_with_untimed_tail)
    assert report.valid
    assert report.warnings == []

    bad_provenance = validate_timeline([_item("late", None, None, timing_provenance="deferred")])
    assert bad_provenance.errors == []
    assert bad_provenance.warnings == ["late: null timing has non-untimed provenance"]

    # Untimed records skip the range/media/snapshot checks entirely, so an
    # untimed snapshot must NOT raise the snapshot warning.
    untimed_snapshot = _item(
        "snap",
        None,
        None,
        kind="snapshot",
        timing_provenance="untimed",
        payload={"actual_time_ms": None},
    )
    snapshot_report = validate_timeline([untimed_snapshot])
    assert snapshot_report.valid
    assert snapshot_report.warnings == []

    timed_snapshot = _item("snap2", 5, 5, kind="snapshot")
    warned = validate_timeline([timed_snapshot])
    assert warned.valid
    assert warned.warnings == ["snap2: snapshot lacks measured actual frame time"]


def test_validate_timeline_untimed_record_in_timed_region_breaks_canonical_order() -> None:
    timeline = [
        _item("first", 0, 10),
        _item("appendix", None, None, timing_provenance="untimed"),
        _item("second", 20, 30),
    ]
    report = validate_timeline(timeline)
    assert not report.valid
    assert report.errors[-1] == "timeline items are not in canonical chronological order"
    assert (report.timed_count, report.untimed_count) == (2, 1)


def test_validate_timeline_emits_range_errors_first_and_ordering_error_last() -> None:
    timeline = [
        _item("bad_end", 100, 50),
        _item("negative", -10, -5),
        _item("partial", 30, None),
        _item("later", 40, 60),
        _item("early_out_of_order", 10, 20),
    ]
    report = validate_timeline(timeline)
    assert not report.valid
    assert report.errors == [
        "bad_end: invalid timing range",
        "negative: invalid timing range",
        "partial: partial timing range",
        "partial: invalid timing range",
        "timeline items are not in canonical chronological order",
    ]


def test_validate_timeline_media_bounds() -> None:
    assert validate_timeline([_item("a", 0, 90)], media_duration_ms=100).valid

    over_start = validate_timeline([_item("b", 150, 200)], media_duration_ms=100)
    assert not over_start.valid
    assert over_start.errors == ["b: timing exceeds media duration"]

    over_end = validate_timeline([_item("c", 50, 150)], media_duration_ms=100)
    assert not over_end.valid
    assert over_end.errors == ["c: timing exceeds media duration"]


def test_validate_timeline_require_sorted_false_skips_order_check() -> None:
    timeline = [_item("b", 100, 110), _item("a", 0, 10)]
    relaxed = validate_timeline(timeline, require_sorted=False)
    assert relaxed.valid
    assert relaxed.errors == []

    strict = validate_timeline(timeline)
    assert not strict.valid
    assert strict.errors[-1] == "timeline items are not in canonical chronological order"


def test_validate_timeline_empty_input_is_valid() -> None:
    report = validate_timeline([])
    assert report.valid
    assert report.errors == []
    assert report.warnings == []
    assert (report.timed_count, report.untimed_count) == (0, 0)


def test_validate_timeline_converts_raw_mappings_like_make_timeline_item() -> None:
    raw_records: list[dict[str, Any]] = [
        {"kind": "chapter", "chapter_id": "c1", "start_ms": 0, "end_ms": 100},
        {
            "kind": "snapshot",
            "snapshot_id": "s1",
            "timestamp_ms": 200,
            "actual_time_ms": 205,
        },
        {
            "kind": "transcript_segment",
            "segment_id": "t1",
            "start_ms": 300,
            "end_ms": 400,
        },
    ]
    converted = [
        make_timeline_item(record, str(record["kind"]), index)
        for index, record in enumerate(raw_records)
    ]
    from_raw = validate_timeline(raw_records)
    from_items = validate_timeline(converted)
    assert from_raw == from_items
    assert from_raw.valid
    assert (from_raw.timed_count, from_raw.untimed_count) == (3, 0)

    unsorted_raw = validate_timeline([raw_records[1], raw_records[0], raw_records[2]])
    assert not unsorted_raw.valid
    assert unsorted_raw.errors[-1] == "timeline items are not in canonical chronological order"

    untimed_raw = validate_timeline([{"kind": "review_decision", "decision_id": "r1"}])
    assert untimed_raw.valid
    assert (untimed_raw.timed_count, untimed_raw.untimed_count) == (0, 1)
