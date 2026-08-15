"""Deterministic transcript-candidate quality diagnostics."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    severity: str
    segment_ids: tuple[str, ...] = ()
    start_ms: int | None = None
    end_ms: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IntervalDiagnostic:
    start_ms: int
    end_ms: int
    reliable: bool
    issue_codes: tuple[str, ...] = ()
    segment_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue]
    interval_diagnostics: list[IntervalDiagnostic]
    reliable_intervals: list[tuple[int, int]]
    unreliable_intervals: list[tuple[int, int]]
    metrics: dict[str, float | int | None]
    usable: bool
    quality_score: float

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return asdict(self)


_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:[.·…_-]{2,}|\[(?:blank|placeholder)\]|(?:n/?a|null|undefined))\s*$",
    re.I,
)
_WORD_RE = re.compile(r"\b[^\W_]+(?:['’][^\W_]+)?\b", re.UNICODE)
_HIGH_IMPACT_RE = re.compile(
    r"(?:\b\d+(?:[.,:]\d+)*(?:%|mg|kg|cm|mm|ml|gb|mb|kb)?\b|"
    r"https?://\S+|\b\S+@\S+\.\S+\b|(?:^|\s)--?[A-Za-z][\w-]*|"
    r"(?:[A-Za-z]:[\\/]|/)[^\s]+)",
    re.I,
)
_MOJIBAKE_RE = re.compile(r"(?:\ufffd|Ã.|Â.|â€|ðŸ)")

# Some ASR runtimes expose legacy/alternate ISO language codes. Treat aliases
# as equivalent for validation while preserving the original detected code in
# the evidence record. Whisper, for example, exposes Filipino as ``tl`` while
# the public CLI and config use the standards-based ``fil`` spelling.
_LANGUAGE_ALIASES: dict[str, str] = {
    "fil": "fil",
    "tl": "fil",
    "tagalog": "fil",
    "filipino": "fil",
    "iw": "he",
    "heb": "he",
    "in": "id",
    "ind": "id",
    "ji": "yi",
    "yid": "yi",
}


def _language_family(value: str | None) -> str | None:
    """Return a comparison key without rewriting provenance values."""

    if not value:
        return None
    root = str(value).strip().casefold().replace("_", "-").split("-")[0]
    return _LANGUAGE_ALIASES.get(root, root)


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _segment_id(segment: Any, index: int) -> str:
    return str(_get(segment, "segment_id", _get(segment, "id", f"segment-{index + 1}")))


def _text(segment: Any) -> str:
    for key in ("human_verified_text", "repaired_text", "normalized_text", "raw_text", "text"):
        value = _get(segment, key)
        if value is not None:
            return str(value)
    return ""


def _merge_intervals(
    intervals: Iterable[tuple[int, int]], *, touching: bool = True
) -> list[tuple[int, int]]:
    ordered = sorted(
        (max(0, int(start)), max(0, int(end))) for start, end in intervals if end > start
    )
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + (1 if touching else 0):
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intersection_ms(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _speech_coverage(
    timed_segments: Sequence[tuple[int, int]], speech_intervals: Sequence[tuple[int, int]]
) -> tuple[float | None, list[tuple[int, int]]]:
    speech = _merge_intervals(speech_intervals)
    if not speech:
        return None, []
    total = sum(end - start for start, end in speech)
    covered = 0
    missing: list[tuple[int, int]] = []
    transcript = _merge_intervals(timed_segments)
    for interval in speech:
        interval_covered = sum(_intersection_ms(interval, cue) for cue in transcript)
        covered += min(interval[1] - interval[0], interval_covered)
        if interval_covered < (interval[1] - interval[0]) * 0.35:
            missing.append(interval)
    return covered / total if total else None, missing


def validate_segments(
    segments: Sequence[Any],
    *,
    media_duration_ms: int | None = None,
    speech_intervals: Sequence[tuple[int, int]] | None = None,
    expected_language: str | None = None,
    max_gap_ms: int = 15_000,
    max_words_per_second: float = 6.0,
    max_chars_per_second: float = 30.0,
    overlap_tolerance_ms: int = 80,
    drift_tolerance_ms: int = 1_000,
) -> ValidationReport:
    """Validate transcript segments and identify reliable/suspect intervals.

    Input order is retained for non-monotonic checks; sorting is used only when
    calculating temporal gaps and coverage.
    """

    issues: list[ValidationIssue] = []
    timed: list[tuple[int, int, str, str, int]] = []
    issue_intervals: list[tuple[int, int]] = []
    placeholder_count = duplicate_count = overlap_count = 0
    prior_start: int | None = None
    normalized_texts: list[str] = []

    def add_issue(
        code: str,
        message: str,
        severity: str,
        ids: Sequence[str] = (),
        start: int | None = None,
        end: int | None = None,
        **details: Any,
    ) -> None:
        issues.append(ValidationIssue(code, message, severity, tuple(ids), start, end, details))
        if (
            start is not None
            and end is not None
            and end > start
            and severity in {"error", "warning"}
        ):
            issue_intervals.append((start, end))

    for index, segment in enumerate(segments):
        segment_id = _segment_id(segment, index)
        text = _text(segment)
        normalized = " ".join(text.casefold().split())
        normalized_texts.append(normalized)
        start = _get(segment, "start_ms")
        end = _get(segment, "end_ms")
        if _MOJIBAKE_RE.search(text) or any(
            ord(char) < 32 and char not in "\n\r\t" for char in text
        ):
            add_issue(
                "encoding_corruption",
                "Caption contains replacement, mojibake, or control characters",
                "error",
                [segment_id],
                start,
                end,
            )
        if not text.strip() or _PLACEHOLDER_RE.match(text):
            placeholder_count += 1
            add_issue(
                "placeholder",
                "Caption is empty or a placeholder",
                "warning",
                [segment_id],
                start,
                end,
            )
        if (start is None) != (end is None):
            add_issue(
                "broken_boundary",
                "Only one caption boundary is present",
                "error",
                [segment_id],
                start,
                end,
            )
            continue
        if start is None:
            continue
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
        ):
            add_issue(
                "invalid_range", "Caption times must be integer milliseconds", "error", [segment_id]
            )
            continue
        if start < 0 or end <= start:
            add_issue(
                "invalid_range",
                "Caption has a negative or non-positive time range",
                "error",
                [segment_id],
                max(0, start),
                max(max(0, start) + 1, end),
            )
            continue
        if prior_start is not None and start < prior_start:
            add_issue(
                "non_monotonic",
                "Caption starts before a preceding source cue",
                "error",
                [segment_id],
                start,
                end,
            )
        prior_start = start
        if media_duration_ms is not None and (
            start >= media_duration_ms or end > media_duration_ms + 250
        ):
            add_issue(
                "duration_mismatch",
                "Caption lies outside the media duration",
                "error",
                [segment_id],
                start,
                end,
                media_duration_ms=media_duration_ms,
            )
        duration_seconds = (end - start) / 1000
        words_per_second = len(_WORD_RE.findall(text)) / duration_seconds
        chars_per_second = len(re.sub(r"\s", "", text)) / duration_seconds
        if words_per_second > max_words_per_second or chars_per_second > max_chars_per_second:
            add_issue(
                "reading_rate",
                "Caption reading rate is unreasonable",
                "warning",
                [segment_id],
                start,
                end,
                words_per_second=round(words_per_second, 3),
                chars_per_second=round(chars_per_second, 3),
            )
        language = _get(segment, "language")
        if (
            expected_language
            and language
            and _language_family(str(language)) != _language_family(expected_language)
        ):
            add_issue(
                "language_mismatch",
                "Caption language differs from the expected language",
                "warning",
                [segment_id],
                start,
                end,
                actual=str(language),
                expected=expected_language,
            )
        if _HIGH_IMPACT_RE.search(text) and re.search(
            r"(?:\?{2,}|\ufffd|\[uncertain)", text, flags=re.I
        ):
            add_issue(
                "suspicious_high_impact_token",
                "A high-impact token appears corrupted or uncertain",
                "error",
                [segment_id],
                start,
                end,
            )
        timed.append((start, end, segment_id, normalized, index))

    by_time = sorted(timed, key=lambda item: (item[0], item[1], item[4]))
    for position, current in enumerate(by_time):
        start, end, segment_id, normalized, _ = current
        if position:
            previous = by_time[position - 1]
            if start < previous[1] - overlap_tolerance_ms:
                overlap_count += 1
                add_issue(
                    "overlap",
                    "Caption overlaps the preceding timed cue",
                    "warning",
                    [previous[2], segment_id],
                    start,
                    min(end, previous[1]),
                    overlap_ms=previous[1] - start,
                )
            gap = start - previous[1]
            if gap > max_gap_ms:
                add_issue(
                    "long_gap",
                    "Long unexplained gap between timed captions",
                    "warning",
                    [previous[2], segment_id],
                    previous[1],
                    start,
                    gap_ms=gap,
                )
        if normalized:
            for previous in by_time[max(0, position - 3) : position]:
                if normalized == previous[3]:
                    duplicate_count += 1
                    add_issue(
                        "duplicate",
                        "Nearby cues repeat the same caption text",
                        "warning",
                        [previous[2], segment_id],
                        min(start, previous[0]),
                        max(end, previous[1]),
                    )
                    break

    # Detect loops of two or more consecutive captions repeated immediately.
    for width in range(2, min(6, len(normalized_texts) // 2 + 1)):
        for offset in range(0, len(normalized_texts) - width * 2 + 1):
            first = normalized_texts[offset : offset + width]
            if any(not value for value in first):
                continue
            if first == normalized_texts[offset + width : offset + width * 2]:
                members = [_segment_id(segments[i], i) for i in range(offset, offset + width * 2)]
                starts = [_get(segments[i], "start_ms") for i in range(offset, offset + width * 2)]
                ends = [_get(segments[i], "end_ms") for i in range(offset, offset + width * 2)]
                valid_starts = [value for value in starts if isinstance(value, int)]
                valid_ends = [value for value in ends if isinstance(value, int)]
                add_issue(
                    "hallucination_loop",
                    "A sequence of captions repeats immediately",
                    "error",
                    members,
                    min(valid_starts) if valid_starts else None,
                    max(valid_ends) if valid_ends else None,
                    loop_width=width,
                )

    coverage, missing_speech = _speech_coverage(
        [(start, end) for start, end, *_ in timed], speech_intervals or []
    )
    ordered_speech = sorted(speech_intervals or [])
    if len(by_time) == len(ordered_speech) and len(by_time) >= 2:
        offsets = [
            cue[0] - int(speech[0]) for cue, speech in zip(by_time, ordered_speech, strict=True)
        ]
        drift_span = max(offsets) - min(offsets)
        if drift_span > drift_tolerance_ms:
            add_issue(
                "drift",
                "Caption timing offset changes progressively against speech activity",
                "error",
                [item[2] for item in by_time],
                by_time[0][0],
                by_time[-1][1],
                first_offset_ms=offsets[0],
                last_offset_ms=offsets[-1],
                drift_span_ms=drift_span,
            )
    for start, end in missing_speech:
        add_issue(
            "missing_speech",
            "Speech activity is weakly covered by captions",
            "error",
            (),
            start,
            end,
        )
    if coverage is not None and coverage < 0.65:
        add_issue(
            "speech_activity_mismatch",
            "Overall speech coverage is too low",
            "error",
            coverage=round(coverage, 4),
        )

    unreliable = _merge_intervals(issue_intervals)
    outer_start = min((item[0] for item in timed), default=0)
    outer_end = max((item[1] for item in timed), default=media_duration_ms or 0)
    if media_duration_ms is not None:
        outer_start, outer_end = 0, media_duration_ms
    reliable: list[tuple[int, int]] = []
    cursor = outer_start
    for start, end in unreliable:
        if start > cursor:
            reliable.append((cursor, start))
        cursor = max(cursor, end)
    if outer_end > cursor:
        reliable.append((cursor, outer_end))

    diagnostics: list[IntervalDiagnostic] = []
    for start, end in unreliable:
        relevant = [
            issue
            for issue in issues
            if issue.start_ms is not None
            and issue.end_ms is not None
            and _intersection_ms((start, end), (issue.start_ms, issue.end_ms)) > 0
        ]
        diagnostics.append(
            IntervalDiagnostic(
                start,
                end,
                False,
                tuple(dict.fromkeys(issue.code for issue in relevant)),
                tuple(dict.fromkeys(item for issue in relevant for item in issue.segment_ids)),
            )
        )
    diagnostics.extend(IntervalDiagnostic(start, end, True) for start, end in reliable)
    diagnostics.sort(key=lambda item: (item.start_ms, not item.reliable, item.end_ms))

    severity_penalty = {"info": 0.01, "warning": 0.08, "error": 0.22}
    denominator = max(1, len(segments))
    penalty = sum(severity_penalty.get(issue.severity, 0.1) for issue in issues) / math.sqrt(
        denominator
    )
    quality_score = max(0.0, min(1.0, 1.0 - penalty))
    error_codes = {issue.code for issue in issues if issue.severity == "error"}
    has_substantive_caption = placeholder_count < len(segments)
    usable = (
        bool(segments)
        and has_substantive_caption
        and quality_score >= 0.35
        and not (
            {"encoding_corruption", "invalid_range", "speech_activity_mismatch"} & error_codes
            and quality_score < 0.6
        )
    )
    report = ValidationReport(
        issues=issues,
        interval_diagnostics=diagnostics,
        reliable_intervals=reliable,
        unreliable_intervals=unreliable,
        metrics={
            "segment_count": len(segments),
            "timed_segment_count": len(timed),
            "placeholder_count": placeholder_count,
            "duplicate_count": duplicate_count,
            "overlap_count": overlap_count,
            "speech_coverage": round(coverage, 6) if coverage is not None else None,
            "issue_count": len(issues),
        },
        usable=usable,
        quality_score=round(quality_score, 6),
    )
    LOGGER.info(
        "validated_transcript",
        extra={
            "segment_count": len(segments),
            "issue_count": len(issues),
            "quality_score": report.quality_score,
            "usable": usable,
        },
    )
    return report


def validate_candidate(candidate: Any, **kwargs: Any) -> ValidationReport:
    """Validate the segments associated with a transcript-source candidate."""

    return validate_segments(list(_get(candidate, "segments", [])), **kwargs)


__all__ = [
    "IntervalDiagnostic",
    "ValidationIssue",
    "ValidationReport",
    "validate_candidate",
    "validate_segments",
]
