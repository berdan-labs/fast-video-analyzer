"""Deterministic construction and validation of the canonical timeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any

LOGGER = logging.getLogger(__name__)


class TimelineError(ValueError):
    """Raised when timeline records cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class TimelineItem:
    timeline_id: str
    kind: str
    source_id: str
    start_ms: int | None
    end_ms: int | None
    timing_provenance: str
    payload: Any
    source_order: int

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TimelineValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timed_count: int = 0
    untimed_count: int = 0


_KIND_PRIORITY = {
    "chapter": 0,
    "scene": 1,
    "speaker_turn": 2,
    "transcript_segment": 3,
    "transcript_word": 4,
    "ocr": 5,
    "visual_event": 6,
    "snapshot": 7,
    "non_speech_audio": 8,
    "uncertainty": 9,
    "review_decision": 10,
}

# Subtitle containers commonly round the final cue independently from the
# media probe (for example, a cue ending 199 ms after the probed duration).
# Keep this aligned with subtitle_validate.validate_segments so the canonical
# timeline remains playable without hiding materially bad source timing.
_DEFAULT_MEDIA_BOUND_TOLERANCE_MS = 250


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _first(record: Any, keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = _get(record, key)
        if value is not None:
            return value
    return default


def _source_id(record: Any, kind: str, index: int) -> str:
    id_fields = {
        "transcript_segment": ("segment_id", "id"),
        "transcript_word": ("word_id", "id"),
        "speaker_turn": ("turn_id", "id"),
        "chapter": ("chapter_id", "id"),
        "scene": ("scene_id", "id"),
        "visual_event": ("event_id", "visual_event_id", "id"),
        "ocr": ("ocr_id", "observation_id", "id"),
        "snapshot": ("snapshot_id", "frame_id", "id"),
        "non_speech_audio": ("event_id", "id"),
        "uncertainty": ("uncertainty_id", "id"),
        "review_decision": ("decision_id", "id"),
    }
    existing = _first(record, id_fields.get(kind, ("id",)))
    if existing is not None:
        return str(existing)
    payload = json.dumps(
        [kind, index, _get(record, "start_ms"), _get(record, "end_ms")], separators=(",", ":")
    ).encode("utf-8")
    return f"{kind}_{sha256(payload).hexdigest()[:16]}"


def _times(record: Any, kind: str) -> tuple[int | None, int | None]:
    start = _first(
        record,
        ("start_ms", "actual_time_ms", "timestamp_ms", "time_ms", "requested_time_ms"),
    )
    end = _first(record, ("end_ms",))
    if (
        end is None
        and start is not None
        and kind in {"visual_event", "ocr", "snapshot", "uncertainty", "review_decision"}
    ):
        end = start
    return (int(start) if start is not None else None, int(end) if end is not None else None)


def _provenance(record: Any, kind: str, start: int | None) -> str:
    explicit = _first(record, ("timing_provenance", "time_provenance", "provenance"))
    if explicit is not None:
        return str(explicit)
    if start is None:
        return "untimed"
    if kind == "snapshot" and _get(record, "actual_time_ms") is not None:
        return "measured_frame_time"
    return "source_timing"


def make_timeline_item(
    record: Any,
    kind: str,
    source_order: int,
    *,
    media_duration_ms: int | None = None,
    media_bound_tolerance_ms: int = _DEFAULT_MEDIA_BOUND_TOLERANCE_MS,
) -> TimelineItem:
    source_id = _source_id(record, kind, source_order)
    start, end = _times(record, kind)
    if (
        media_duration_ms is not None
        and media_duration_ms >= 0
        and media_bound_tolerance_ms >= 0
        and start is not None
        and end is not None
        and start < media_duration_ms
        and media_duration_ms < end <= media_duration_ms + media_bound_tolerance_ms
    ):
        # Retain the source record unchanged in ``payload`` while ensuring the
        # derived timeline range never points beyond the playable media.
        end = media_duration_ms
    canonical = json.dumps(
        [kind, source_id, start, end], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return TimelineItem(
        timeline_id=f"timeline_{sha256(canonical).hexdigest()[:20]}",
        kind=kind,
        source_id=source_id,
        start_ms=start,
        end_ms=end,
        timing_provenance=_provenance(record, kind, start),
        payload=record,
        source_order=source_order,
    )


def timeline_sort_key(item: TimelineItem) -> tuple[int, int, int, int, int, str]:
    """Sort timed items chronologically and untimed records deterministically last."""

    untimed = item.start_ms is None
    return (
        1 if untimed else 0,
        item.start_ms or 0,
        _KIND_PRIORITY.get(item.kind, 100),
        item.end_ms if item.end_ms is not None else item.start_ms or 0,
        item.source_order,
        item.source_id,
    )


def build_timeline(
    transcript_segments: Sequence[Any] = (),
    *,
    speaker_turns: Sequence[Any] = (),
    chapters: Sequence[Any] = (),
    scenes: Sequence[Any] = (),
    visual_events: Sequence[Any] = (),
    ocr_observations: Sequence[Any] = (),
    snapshots: Sequence[Any] = (),
    non_speech_events: Sequence[Any] = (),
    uncertainty_items: Sequence[Any] = (),
    review_decisions: Sequence[Any] = (),
    include_words: bool = True,
    media_duration_ms: int | None = None,
    media_bound_tolerance_ms: int = _DEFAULT_MEDIA_BOUND_TOLERANCE_MS,
) -> list[TimelineItem]:
    """Build one ordered timeline; no language model participates in chronology.

    A small end-boundary overrun is clipped when ``media_duration_ms`` is
    supplied. This handles container/subtitle rounding while preserving the
    original record in each item's payload; larger overruns remain invalid.
    """

    items: list[TimelineItem] = []
    order = 0

    def add(records: Sequence[Any], kind: str) -> None:
        nonlocal order
        for record in records:
            items.append(
                make_timeline_item(
                    record,
                    kind,
                    order,
                    media_duration_ms=media_duration_ms,
                    media_bound_tolerance_ms=media_bound_tolerance_ms,
                )
            )
            order += 1

    add(chapters, "chapter")
    add(scenes, "scene")
    add(speaker_turns, "speaker_turn")
    for segment in transcript_segments:
        items.append(
            make_timeline_item(
                segment,
                "transcript_segment",
                order,
                media_duration_ms=media_duration_ms,
                media_bound_tolerance_ms=media_bound_tolerance_ms,
            )
        )
        order += 1
        if include_words:
            add(list(_get(segment, "words", []) or []), "transcript_word")
    add(ocr_observations, "ocr")
    add(visual_events, "visual_event")
    add(snapshots, "snapshot")
    add(non_speech_events, "non_speech_audio")
    add(uncertainty_items, "uncertainty")
    add(review_decisions, "review_decision")
    items.sort(key=timeline_sort_key)
    LOGGER.info(
        "built_timeline",
        extra={
            "item_count": len(items),
            "timed_count": sum(item.start_ms is not None for item in items),
        },
    )
    return items


def validate_timeline(
    timeline: Sequence[TimelineItem | Any],
    *,
    media_duration_ms: int | None = None,
    require_sorted: bool = True,
) -> TimelineValidationReport:
    """Validate ranges, ordering, media bounds, and untimed provenance."""

    errors: list[str] = []
    warnings: list[str] = []
    converted: list[TimelineItem] = []
    ids: set[str] = set()
    for index, raw in enumerate(timeline):
        item = (
            raw
            if isinstance(raw, TimelineItem)
            else make_timeline_item(raw, str(_get(raw, "kind", "unknown")), index)
        )
        converted.append(item)
        if item.timeline_id in ids:
            errors.append(f"duplicate timeline ID: {item.timeline_id}")
        ids.add(item.timeline_id)
        if (item.start_ms is None) != (item.end_ms is None):
            errors.append(f"{item.timeline_id}: partial timing range")
        if item.start_ms is None:
            if item.timing_provenance != "untimed" and "untimed" not in item.timing_provenance:
                warnings.append(f"{item.timeline_id}: null timing has non-untimed provenance")
            continue
        if item.start_ms < 0 or item.end_ms is None or item.end_ms < item.start_ms:
            errors.append(f"{item.timeline_id}: invalid timing range")
        if media_duration_ms is not None and (
            item.start_ms > media_duration_ms or (item.end_ms or 0) > media_duration_ms
        ):
            errors.append(f"{item.timeline_id}: timing exceeds media duration")
        if item.kind == "snapshot" and _get(item.payload, "actual_time_ms") is None:
            warnings.append(f"{item.timeline_id}: snapshot lacks measured actual frame time")
    if require_sorted and converted != sorted(converted, key=timeline_sort_key):
        errors.append("timeline items are not in canonical chronological order")
    return TimelineValidationReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        timed_count=sum(item.start_ms is not None for item in converted),
        untimed_count=sum(item.start_ms is None for item in converted),
    )


__all__ = [
    "TimelineError",
    "TimelineItem",
    "TimelineValidationReport",
    "build_timeline",
    "make_timeline_item",
    "timeline_sort_key",
    "validate_timeline",
]
