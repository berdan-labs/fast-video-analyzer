"""Subtitle and transcript parsing with lossless raw-text preservation.

The parsers intentionally do not clean or summarize spoken text.  Container
markup is removed where it is not part of the visible caption, while both the
raw and normalized representations are retained on every segment.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from html import unescape
from pathlib import Path
from typing import Any

from .ids import deterministic_id, sequential_id
from .schemas import TranscriptSegment

LOGGER = logging.getLogger(__name__)


class SubtitleParseError(ValueError):
    """Raised when transcript input cannot be parsed safely."""


@dataclass(slots=True)
class ParsedTranscriptSegment:
    """Portable parsed-segment representation used before canonical loading."""

    segment_id: str
    start_ms: int | None
    end_ms: int | None
    timing_provenance: str
    raw_text: str
    normalized_text: str
    repaired_text: str | None = None
    human_verified_text: str | None = None
    speaker_label: str | None = None
    language: str | None = None
    words: list[Any] = field(default_factory=list)
    source_candidate_id: str | None = None
    source_track: str | None = None
    confidence: float | None = None
    verification_status: str = "unverified"
    repair_record_ids: list[str] = field(default_factory=list)
    uncertainty_items: list[str] = field(default_factory=list)
    substantive: bool = True

    def model_dump(self, **_: Any) -> dict[str, Any]:
        """Match the small portion of the Pydantic API used by the pipeline."""

        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


_TIME_RE = re.compile(
    r"^\s*(?:(?P<hours>\d{1,3}):)?(?P<minutes>\d{1,2}):"
    r"(?P<seconds>\d{1,2})(?:[,.](?P<fraction>\d{1,6}))?\s*$"
)
_TIMING_LINE_RE = re.compile(
    r"^\s*(?P<start>(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[,.]\d+)?)\s*"
    r"-->\s*(?P<end>(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[,.]\d+)?)"
    r"(?:\s+.*)?$"
)
_PLAIN_RANGE_RE = re.compile(
    r"^\s*[\[(]?(?P<start>(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[,.]\d+)?)\s*"
    r"(?:-->|[-–—])\s*(?P<end>(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[,.]\d+)?)"
    r"[\])]?\s*[:\-]?\s*(?P<text>.*)$"
)
_PLAIN_POINT_RE = re.compile(
    r"^\s*[\[(]?(?P<time>(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[,.]\d+)?)"
    r"(?:[\])]\s*|\s*[:\-]\s*)(?P<text>.+)$"
)
_VTT_TAG_RE = re.compile(
    r"</?(?:c(?:\.[^ >]+)?|v(?:\s+[^>]*)?|lang(?:\s+[^>]*)?|ruby|rt|b|i|u)>", re.I
)
_ASS_OVERRIDE_RE = re.compile(r"\{[^{}]*\}")


def parse_timestamp(value: str) -> int:
    """Parse an SRT/VTT/ASS-style timestamp into integer milliseconds."""

    value = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return int(round(float(value) * 1000))
    match = _TIME_RE.match(value)
    if not match:
        raise SubtitleParseError(f"Invalid timestamp: {value!r}")
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise SubtitleParseError(f"Out-of-range timestamp: {value!r}")
    fraction = match.group("fraction") or ""
    milliseconds = int((fraction + "000")[:3])
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def _stable_segment_id(
    source: str, index: int, start_ms: int | None, end_ms: int | None, raw_text: str
) -> str:
    # Transcript IDs are ordered, portable evidence references.  The source
    # and cue contents still determine parsing and candidate ranking; the
    # canonical visible ID follows the contract's T000001 form and remains
    # stable while that candidate's ordering is unchanged.
    del source, start_ms, end_ms, raw_text
    return sequential_id("transcript", index + 1)


def _normalize_visible_text(text: str) -> str:
    return "\n".join(
        line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip()


def _segment(
    *,
    source: str,
    index: int,
    start_ms: int | None,
    end_ms: int | None,
    raw_text: str,
    normalized_text: str | None = None,
    timing_provenance: str,
    speaker_label: str | None = None,
    language: str | None = None,
    confidence: float | None = None,
    source_candidate_id: str | None = None,
) -> Any:
    normalized = _normalize_visible_text(
        normalized_text if normalized_text is not None else raw_text
    )
    values: dict[str, Any] = dict(
        segment_id=_stable_segment_id(source, index, start_ms, end_ms, raw_text),
        start_ms=start_ms,
        end_ms=end_ms,
        timing_provenance=timing_provenance,
        raw_text=raw_text,
        normalized_text=normalized,
        speaker_label=speaker_label or None,
        language=language,
        confidence=confidence,
        source_candidate_id=source_candidate_id,
        substantive=bool(normalized.strip()),
    )
    return TranscriptSegment(**values)


def decode_subtitle_bytes(data: bytes, encoding: str | None = None) -> tuple[str, str]:
    """Decode caption bytes without silently replacing undecodable content."""

    if encoding:
        try:
            return data.decode(encoding), encoding
        except (LookupError, UnicodeDecodeError) as exc:
            raise SubtitleParseError(f"Unable to decode subtitles as {encoding}: {exc}") from exc
    candidates = ["utf-8-sig"]
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates.append("utf-16")
    elif data and data.count(b"\x00") / len(data) > 0.15:
        candidates.extend(("utf-16-le", "utf-16-be"))
    candidates.append("cp1252")
    for candidate in candidates:
        try:
            text = data.decode(candidate)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text:
            return text, candidate
    raise SubtitleParseError("Subtitle encoding could not be determined without data loss")


def parse_srt(
    text: str, *, source: str = "srt", source_candidate_id: str | None = None
) -> list[ParsedTranscriptSegment]:
    """Parse SubRip captions, including multiline cues and omitted cue numbers."""

    lines = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    segments: list[ParsedTranscriptSegment] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        cue_start = index
        if lines[index].strip().isdigit() and index + 1 < len(lines):
            index += 1
        timing = _TIMING_LINE_RE.match(lines[index]) if index < len(lines) else None
        if timing is None:
            preview = lines[cue_start][:80]
            raise SubtitleParseError(f"Expected SRT timing line near {preview!r}")
        start_ms = parse_timestamp(timing.group("start"))
        end_ms = parse_timestamp(timing.group("end"))
        index += 1
        body: list[str] = []
        while index < len(lines) and lines[index].strip():
            body.append(lines[index])
            index += 1
        raw_text = "\n".join(body)
        segments.append(
            _segment(
                source=source,
                index=len(segments),
                start_ms=start_ms,
                end_ms=end_ms,
                raw_text=raw_text,
                timing_provenance="source_srt",
                source_candidate_id=source_candidate_id,
            )
        )
    return segments


def _clean_vtt_text(text: str) -> tuple[str, str | None]:
    speaker: str | None = None
    voice = re.match(r"\s*<v(?:\s+([^>]+))?>", text, flags=re.I)
    if voice and voice.group(1):
        speaker = unescape(voice.group(1).strip())
    cleaned = _VTT_TAG_RE.sub("", text)
    return unescape(cleaned), speaker


def parse_vtt(
    text: str, *, source: str = "vtt", source_candidate_id: str | None = None
) -> list[ParsedTranscriptSegment]:
    """Parse WebVTT while ignoring NOTE/STYLE/REGION metadata blocks."""

    lines = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    index = 0
    if lines and lines[0].strip().startswith("WEBVTT"):
        index = 1
    segments: list[ParsedTranscriptSegment] = []
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if re.match(r"^(NOTE|STYLE|REGION)(?:\s|$)", lines[index].strip(), flags=re.I):
            index += 1
            while index < len(lines) and lines[index].strip():
                index += 1
            continue
        if _TIMING_LINE_RE.match(lines[index]) is None and index + 1 < len(lines):
            index += 1  # cue identifier
        timing = _TIMING_LINE_RE.match(lines[index]) if index < len(lines) else None
        if timing is None:
            raise SubtitleParseError(f"Expected WebVTT timing line near {lines[index][:80]!r}")
        start_ms = parse_timestamp(timing.group("start"))
        end_ms = parse_timestamp(timing.group("end"))
        index += 1
        body: list[str] = []
        while index < len(lines) and lines[index].strip():
            body.append(lines[index])
            index += 1
        raw_text = "\n".join(body)
        cleaned, speaker = _clean_vtt_text(raw_text)
        segments.append(
            _segment(
                source=source,
                index=len(segments),
                start_ms=start_ms,
                end_ms=end_ms,
                raw_text=raw_text,
                normalized_text=cleaned,
                timing_provenance="source_vtt",
                speaker_label=speaker,
                source_candidate_id=source_candidate_id,
            )
        )
    return segments


def parse_ass(
    text: str, *, source: str = "ass", source_candidate_id: str | None = None
) -> list[ParsedTranscriptSegment]:
    """Parse ASS/SSA dialogue events according to the declared Events format."""

    in_events = False
    fields = [
        "Layer",
        "Start",
        "End",
        "Style",
        "Name",
        "MarginL",
        "MarginR",
        "MarginV",
        "Effect",
        "Text",
    ]
    segments: list[ParsedTranscriptSegment] = []
    for line_number, line in enumerate(text.lstrip("\ufeff").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_events = stripped.casefold() == "[events]"
            continue
        if not in_events or not stripped or stripped.startswith(";"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.strip().casefold() == "format":
            fields = [part.strip() for part in value.split(",")]
            continue
        if key.strip().casefold() != "dialogue":
            continue
        parts = value.lstrip().split(",", max(0, len(fields) - 1))
        if len(parts) != len(fields):
            raise SubtitleParseError(f"Malformed ASS Dialogue event on line {line_number}")
        record = {name.casefold(): part for name, part in zip(fields, parts, strict=True)}
        if not {"start", "end", "text"}.issubset(record):
            raise SubtitleParseError("ASS Events Format must include Start, End, and Text")
        raw_text = record["text"]
        visible = (
            _ASS_OVERRIDE_RE.sub("", raw_text)
            .replace(r"\N", "\n")
            .replace(r"\n", "\n")
            .replace(r"\h", " ")
        )
        segments.append(
            _segment(
                source=source,
                index=len(segments),
                start_ms=parse_timestamp(record["start"]),
                end_ms=parse_timestamp(record["end"]),
                raw_text=raw_text,
                normalized_text=visible,
                timing_provenance="source_ass",
                speaker_label=record.get("name", "").strip() or None,
                source_candidate_id=source_candidate_id,
            )
        )
    return segments


def _json_records(payload: Any) -> tuple[list[Mapping[str, Any]], str | None]:
    language: str | None = None
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        language_value = payload.get("language") or payload.get("lang")
        language = str(language_value) if language_value is not None else None
        records = []
        for key in ("segments", "cues", "captions", "transcript", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if not records and any(key in payload for key in ("text", "raw_text", "transcript")):
            records = [payload]
    else:
        raise SubtitleParseError("Transcript JSON must be an object or array")
    if not all(isinstance(item, Mapping) for item in records):
        raise SubtitleParseError("Every transcript JSON segment must be an object")
    return list(records), language


def _json_time(record: Mapping[str, Any], base: str, unit_hint: str | None) -> int | None:
    ms_key = f"{base}_ms"
    if ms_key in record and record[ms_key] is not None:
        return int(round(float(record[ms_key])))
    value = record.get(base)
    if value is None and base == "start":
        value = record.get("start_time")
    if value is None and base == "end":
        value = record.get("end_time")
    if value is None:
        return None
    if isinstance(value, str) and ":" in value:
        return parse_timestamp(value)
    if not isinstance(value, (int, float, str)):
        raise SubtitleParseError(f"Invalid JSON {base} timestamp: {value!r}")
    numeric = float(value)
    return int(
        round(numeric if unit_hint in {"ms", "millisecond", "milliseconds"} else numeric * 1000)
    )


def parse_json_transcript(
    text: str, *, source: str = "json", source_candidate_id: str | None = None
) -> list[ParsedTranscriptSegment]:
    """Parse common timestamped JSON transcript shapes without losing untimed text."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SubtitleParseError(f"Invalid transcript JSON: {exc}") from exc
    records, document_language = _json_records(payload)
    document_unit = (
        str(payload.get("time_unit", "seconds")).casefold()
        if isinstance(payload, Mapping)
        else "seconds"
    )
    segments: list[ParsedTranscriptSegment] = []
    for index, record in enumerate(records):
        raw_value = record.get("raw_text", record.get("text", record.get("transcript", "")))
        if isinstance(raw_value, list):
            raw_value = " ".join(str(value) for value in raw_value)
        raw_text = str(raw_value)
        unit = str(record.get("time_unit", document_unit)).casefold()
        start_ms = _json_time(record, "start", unit)
        end_ms = _json_time(record, "end", unit)
        if end_ms is None and start_ms is not None and record.get("duration") is not None:
            multiplier = 1 if unit.startswith("ms") or unit.startswith("milli") else 1000
            end_ms = start_ms + int(round(float(record["duration"]) * multiplier))
        confidence = record.get("confidence")
        segments.append(
            _segment(
                source=source,
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                raw_text=raw_text,
                normalized_text=str(record.get("normalized_text", raw_text)),
                timing_provenance="source_json" if start_ms is not None else "untimed_source",
                speaker_label=str(record["speaker"]) if record.get("speaker") is not None else None,
                language=str(record.get("language", document_language))
                if record.get("language", document_language) is not None
                else None,
                confidence=float(confidence) if confidence is not None else None,
                source_candidate_id=source_candidate_id,
            )
        )
        words = record.get("words")
        if isinstance(words, list):
            normalized_words: list[dict[str, Any]] = []
            for word_index, word in enumerate(words):
                if not isinstance(word, Mapping):
                    continue
                word_text = str(word.get("text", word.get("word", "")))
                word_unit = str(word.get("time_unit", unit)).casefold()
                word_start = _json_time(word, "start", word_unit)
                word_end = _json_time(word, "end", word_unit)
                normalized_words.append(
                    {
                        "word_id": str(
                            word.get(
                                "word_id",
                                deterministic_id(
                                    "word",
                                    segments[-1].segment_id,
                                    word_index,
                                    word_start,
                                    word_end,
                                    word_text,
                                ),
                            )
                        ),
                        "text": word_text,
                        "start_ms": word_start,
                        "end_ms": word_end,
                        "confidence": float(word["confidence"])
                        if word.get("confidence") is not None
                        else None,
                        "source": str(word.get("source", source)),
                        "language": str(word.get("language", segments[-1].language))
                        if word.get("language", segments[-1].language) is not None
                        else None,
                        "uncertainty_flags": list(word.get("uncertainty_flags", [])),
                    }
                )
            segments[-1].words = normalized_words
    return segments


def parse_plain_text(
    text: str, *, source: str = "text", source_candidate_id: str | None = None
) -> list[ParsedTranscriptSegment]:
    """Parse untimed paragraphs and common timestamp-prefixed plain text."""

    normalized_input = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized_input.split("\n")
    has_timestamp_lines = any(
        _PLAIN_RANGE_RE.match(line) or _PLAIN_POINT_RE.match(line) for line in lines if line.strip()
    )
    records: list[tuple[int | None, int | None, str, str]] = []
    if has_timestamp_lines:
        for line in lines:
            if not line.strip():
                continue
            range_match = _PLAIN_RANGE_RE.match(line)
            if range_match:
                records.append(
                    (
                        parse_timestamp(range_match.group("start")),
                        parse_timestamp(range_match.group("end")),
                        range_match.group("text"),
                        "source_text_range",
                    )
                )
                continue
            point_match = _PLAIN_POINT_RE.match(line)
            if point_match:
                records.append(
                    (
                        parse_timestamp(point_match.group("time")),
                        None,
                        point_match.group("text"),
                        "source_text_point",
                    )
                )
            elif records:
                start, end, prior, provenance = records[-1]
                records[-1] = (start, end, f"{prior}\n{line}", provenance)
            else:
                records.append((None, None, line, "untimed_source"))
    else:
        paragraphs = [
            part.strip("\n") for part in re.split(r"\n\s*\n", normalized_input) if part.strip()
        ]
        records = [(None, None, paragraph, "untimed_source") for paragraph in paragraphs]
    return [
        _segment(
            source=source,
            index=index,
            start_ms=start,
            end_ms=end,
            raw_text=raw,
            timing_provenance=provenance,
            source_candidate_id=source_candidate_id,
        )
        for index, (start, end, raw, provenance) in enumerate(records)
    ]


def infer_subtitle_format(path: str | Path | None, text: str = "") -> str:
    """Infer a supported transcript format from suffix, then conservative content cues."""

    suffix = Path(path).suffix.casefold() if path else ""
    by_suffix = {
        ".srt": "srt",
        ".vtt": "vtt",
        ".ass": "ass",
        ".ssa": "ass",
        ".json": "json",
        ".txt": "text",
        ".md": "text",
    }
    if suffix in by_suffix:
        return by_suffix[suffix]
    stripped = text.lstrip("\ufeff \t\r\n")
    if stripped.startswith("WEBVTT"):
        return "vtt"
    if re.search(r"(?im)^\s*\[events\]\s*$", stripped):
        return "ass"
    if stripped.startswith(("{", "[")):
        parsed_as_json = True
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            parsed_as_json = False
        if parsed_as_json:
            return "json"
    if any(_TIMING_LINE_RE.match(line) for line in stripped.splitlines()):
        return "srt"
    return "text"


def parse_subtitle_text(
    text: str,
    format: str | None = None,
    *,
    source: str = "inline",
    source_candidate_id: str | None = None,
) -> list[ParsedTranscriptSegment]:
    """Parse transcript text in a specified or inferred supported format."""

    selected = (format or infer_subtitle_format(None, text)).casefold().lstrip(".")
    parsers = {
        "srt": parse_srt,
        "vtt": parse_vtt,
        "ass": parse_ass,
        "ssa": parse_ass,
        "json": parse_json_transcript,
        "text": parse_plain_text,
        "txt": parse_plain_text,
    }
    try:
        parser = parsers[selected]
    except KeyError as exc:
        raise SubtitleParseError(f"Unsupported transcript format: {selected!r}") from exc
    result = parser(text, source=source, source_candidate_id=source_candidate_id)
    LOGGER.info(
        "parsed_transcript",
        extra={"format": selected, "segment_count": len(result), "source": source},
    )
    return result


def parse_transcript(
    path: str | Path,
    *,
    format: str | None = None,
    encoding: str | None = None,
    source_candidate_id: str | None = None,
) -> list[ParsedTranscriptSegment]:
    """Read and parse a transcript file using its suffix and/or content."""

    transcript_path = Path(path)
    if not transcript_path.is_file():
        raise SubtitleParseError(f"Transcript file does not exist: {transcript_path}")
    text, detected_encoding = decode_subtitle_bytes(transcript_path.read_bytes(), encoding)
    selected = format or infer_subtitle_format(transcript_path, text)
    LOGGER.debug(
        "decoded_transcript", extra={"path": str(transcript_path), "encoding": detected_encoding}
    )
    return parse_subtitle_text(
        text,
        selected,
        source=transcript_path.name,
        source_candidate_id=source_candidate_id,
    )


# Backwards-friendly aliases with explicit names.
parse_json = parse_json_transcript
parse_text = parse_plain_text


__all__ = [
    "ParsedTranscriptSegment",
    "SubtitleParseError",
    "decode_subtitle_bytes",
    "infer_subtitle_format",
    "parse_ass",
    "parse_json",
    "parse_json_transcript",
    "parse_plain_text",
    "parse_srt",
    "parse_subtitle_text",
    "parse_text",
    "parse_timestamp",
    "parse_transcript",
    "parse_vtt",
]
