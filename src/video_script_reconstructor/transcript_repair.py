"""Bounded, evidence-recorded repair of suspect transcript intervals."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .ids import deterministic_id
from .whisper_adapter import (
    ASRAdapter,
    normalize_asr_result,
    offset_transcript_timestamps,
)

LOGGER = logging.getLogger(__name__)


class TranscriptRepairError(RuntimeError):
    """Raised when a bounded transcript repair cannot be performed safely."""


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    output_path: Path
    requested_start_ms: int
    requested_end_ms: int
    actual_start_ms: int
    actual_end_ms: int
    context_padding_ms: int
    command: tuple[str, ...]


@dataclass(slots=True)
class RepairRecordData:
    record_id: str
    source_segment_ids: list[str]
    before_text: str
    after_text: str
    action: str
    audio_start_ms: int
    audio_end_ms: int
    context_padding_ms: int
    asr_model: str | None
    asr_settings: dict[str, Any]
    alignment_evidence: dict[str, Any]
    confidence: float | None
    rationale: str
    alternatives: list[str]
    actor: str = "automatic_repair"
    timestamp: str | None = None

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_segment_ids": list(self.source_segment_ids),
            "before_text": self.before_text,
            "candidate_after_text": self.after_text or None,
            "action": self.action,
            "audio_interval": {
                "start_ms": self.audio_start_ms,
                "end_ms": self.audio_end_ms,
            },
            "context_padding_ms": self.context_padding_ms,
            "asr_model": self.asr_model,
            "asr_settings": dict(self.asr_settings),
            "alignment_evidence": [
                json.dumps(
                    self.alignment_evidence,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ],
            "confidence": self.confidence,
            "rationale": self.rationale,
            "alternatives": list(self.alternatives),
            "actor": self.actor,
            "created_at_utc": self.timestamp,
        }

    @property
    def candidate_after_text(self) -> str:
        return self.after_text

    @property
    def created_at_utc(self) -> str | None:
        return self.timestamp


@dataclass(slots=True)
class RepairOutcome:
    segments: list[Any]
    records: list[RepairRecordData]
    extracted_intervals: list[ExtractionResult]
    unresolved_intervals: list[tuple[int, int]]


def _run_checked(command: Sequence[str]) -> None:
    # The command is an argument vector assembled by this module; no shell is used.
    completed = subprocess.run(  # noqa: S603
        command, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise TranscriptRepairError(
            f"ffmpeg interval extraction failed ({completed.returncode}): {stderr}"
        )


def extract_interval_audio(
    media_path: str | Path,
    output_path: str | Path,
    start_ms: int,
    end_ms: int,
    *,
    context_padding_ms: int = 750,
    media_duration_ms: int | None = None,
    ffmpeg_path: str = "ffmpeg",
    runner: Callable[[Sequence[str]], None] | None = None,
) -> ExtractionResult:
    """Extract exactly one bounded audio clip, honoring configurable padding."""

    if start_ms < 0 or end_ms <= start_ms:
        raise TranscriptRepairError(
            "Repair interval must have a non-negative start and positive duration"
        )
    if context_padding_ms < 0:
        raise TranscriptRepairError("context_padding_ms cannot be negative")
    actual_start = max(0, start_ms - context_padding_ms)
    actual_end = end_ms + context_padding_ms
    if media_duration_ms is not None:
        actual_end = min(media_duration_ms, actual_end)
    if actual_end <= actual_start:
        raise TranscriptRepairError("Padded repair interval is empty")
    source = Path(media_path)
    target = Path(output_path)
    if not source.is_file():
        raise TranscriptRepairError(f"Media input does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = (
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{actual_start / 1000:.3f}",
        "-i",
        str(source),
        "-t",
        f"{(actual_end - actual_start) / 1000:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    )
    (runner or _run_checked)(command)
    return ExtractionResult(
        target, start_ms, end_ms, actual_start, actual_end, context_padding_ms, command
    )


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _copy_with(record: Any, **updates: Any) -> Any:
    if isinstance(record, Mapping):
        copied: MutableMapping[str, Any] = dict(record)
        copied.update(updates)
        return copied
    if hasattr(record, "model_copy"):
        return record.model_copy(update=updates)
    if is_dataclass(record):
        return replace(record, **updates)  # type: ignore[type-var]
    copied = record.__class__.__new__(record.__class__)
    copied.__dict__.update(record.__dict__)
    copied.__dict__.update(updates)
    return copied


def _text(record: Any) -> str:
    for key in ("human_verified_text", "repaired_text", "normalized_text", "raw_text", "text"):
        value = _get(record, key)
        if value is not None:
            return str(value)
    return ""


def _id(record: Any, index: int) -> str:
    return str(_get(record, "segment_id", _get(record, "id", f"segment-{index + 1}")))


def _overlaps(record: Any, interval: tuple[int, int]) -> bool:
    start, end = _get(record, "start_ms"), _get(record, "end_ms")
    return (
        start is not None
        and end is not None
        and int(start) < interval[1]
        and int(end) > interval[0]
    )


def merge_repair_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Coalesce suspect ranges so the media is never transcribed once per cue."""

    merged: list[tuple[int, int]] = []
    for start, end in sorted((int(start), int(end)) for start, end in intervals):
        if start < 0 or end <= start:
            raise TranscriptRepairError(f"Invalid suspect interval: {(start, end)!r}")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def sequence_alignment(before: str, after: str) -> dict[str, Any]:
    """Return ordered-token alignment evidence for a proposed repair."""

    before_tokens = before.split()
    after_tokens = after.split()
    matcher = SequenceMatcher(
        a=[token.casefold() for token in before_tokens],
        b=[token.casefold() for token in after_tokens],
        autojunk=False,
    )
    return {
        "ratio": round(matcher.ratio(), 6),
        "before_token_count": len(before_tokens),
        "after_token_count": len(after_tokens),
        "opcodes": [list(opcode) for opcode in matcher.get_opcodes()],
    }


def _confidence(segments: Sequence[Any]) -> float | None:
    values = [
        float(value)
        for value in (_get(segment, "confidence") for segment in segments)
        if value is not None
    ]
    return sum(values) / len(values) if values else None


def _recognized_text_for_segment(segment: Any, recognized: Sequence[Any]) -> str:
    start, end = _get(segment, "start_ms"), _get(segment, "end_ms")
    if start is None or end is None:
        return ""
    matching = [item for item in recognized if _overlaps(item, (int(start), int(end)))]
    return " ".join(_text(item).strip() for item in matching if _text(item).strip()).strip()


def _record_id(
    segment_ids: Sequence[str], interval: tuple[int, int], before: str, after: str
) -> str:
    return deterministic_id("repair", list(segment_ids), list(interval), before, after)


def repair_suspect_intervals(
    media_path: str | Path,
    segments: Sequence[Any],
    suspect_intervals: Sequence[tuple[int, int]],
    adapter: ASRAdapter,
    *,
    context_padding_ms: int = 750,
    media_duration_ms: int | None = None,
    minimum_confidence: float = 0.50,
    language: str | None = None,
    extractor: Callable[..., ExtractionResult] = extract_interval_audio,
    work_dir: str | Path | None = None,
) -> RepairOutcome:
    """Repair only suspect intervals and leave all reliable segment objects intact."""

    intervals = merge_repair_intervals(suspect_intervals)
    output = list(segments)
    records: list[RepairRecordData] = []
    extractions: list[ExtractionResult] = []
    unresolved: list[tuple[int, int]] = []

    def process(directory: Path) -> None:
        nonlocal output
        for interval_index, interval in enumerate(intervals):
            clip_path = directory / f"repair-{interval_index:04d}.wav"
            extraction = extractor(
                media_path,
                clip_path,
                interval[0],
                interval[1],
                context_padding_ms=context_padding_ms,
                media_duration_ms=media_duration_ms,
            )
            extractions.append(extraction)
            local = adapter.transcribe(
                extraction.output_path,
                interval_start_ms=None,
                interval_end_ms=None,
                context_padding_ms=0,
                language=language,
            )
            result = normalize_asr_result(local, source=adapter.backend_name)
            recognized = offset_transcript_timestamps(result.segments, extraction.actual_start_ms)
            affected_indices = [
                index for index, segment in enumerate(output) if _overlaps(segment, interval)
            ]
            if not affected_indices:
                unresolved.append(interval)
                continue
            interval_changed = False
            for segment_index in affected_indices:
                current = output[segment_index]
                segment_id = _id(current, segment_index)
                before = _text(current)
                after = _recognized_text_for_segment(current, recognized)
                matching = [
                    item
                    for item in recognized
                    if _overlaps(
                        item, (int(_get(current, "start_ms")), int(_get(current, "end_ms")))
                    )
                ]
                confidence = _confidence(matching)
                evidence = sequence_alignment(before, after)
                supported = bool(after) and (confidence is None or confidence >= minimum_confidence)
                if supported and after != before:
                    repair_ids = list(_get(current, "repair_record_ids", []) or [])
                    record_id = _record_id([segment_id], interval, before, after)
                    repair_ids.append(record_id)
                    output[segment_index] = _copy_with(
                        current,
                        repaired_text=after,
                        repair_record_ids=repair_ids,
                        verification_status="automatically_repaired",
                    )
                    action = "replace"
                    rationale = (
                        "Bounded ASR produced non-empty wording at or above the configured "
                        "confidence threshold"
                    )
                    interval_changed = True
                elif after == before and after:
                    record_id = _record_id([segment_id], interval, before, after)
                    action = "retain"
                    rationale = "Bounded ASR agrees with the preserved source wording"
                else:
                    record_id = _record_id([segment_id], interval, before, after)
                    action = "unresolved"
                    rationale = (
                        "Bounded ASR did not provide sufficiently supported replacement wording"
                    )
                records.append(
                    RepairRecordData(
                        record_id=record_id,
                        source_segment_ids=[segment_id],
                        before_text=before,
                        after_text=after,
                        action=action,
                        audio_start_ms=extraction.actual_start_ms,
                        audio_end_ms=extraction.actual_end_ms,
                        context_padding_ms=context_padding_ms,
                        asr_model=str(result.metadata.get("model"))
                        if result.metadata.get("model") is not None
                        else None,
                        asr_settings=dict(result.metadata.get("decoding_settings", {})),
                        alignment_evidence=evidence,
                        confidence=confidence,
                        rationale=rationale,
                        alternatives=[before, after] if after and after != before else [before],
                        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    )
                )
            if not interval_changed and all(
                record.action == "unresolved"
                for record in records
                if record.audio_start_ms == extraction.actual_start_ms
                and record.audio_end_ms == extraction.actual_end_ms
            ):
                unresolved.append(interval)

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="vsr-repair-") as temporary:
            process(Path(temporary))
    else:
        directory = Path(work_dir)
        directory.mkdir(parents=True, exist_ok=True)
        process(directory)
    LOGGER.info(
        "selective_transcript_repair",
        extra={
            "interval_count": len(intervals),
            "repair_record_count": len(records),
            "unresolved_count": len(unresolved),
        },
    )
    return RepairOutcome(output, records, extractions, unresolved)


selective_repair = repair_suspect_intervals


__all__ = [
    "ExtractionResult",
    "RepairOutcome",
    "RepairRecordData",
    "TranscriptRepairError",
    "extract_interval_audio",
    "merge_repair_intervals",
    "repair_suspect_intervals",
    "selective_repair",
    "sequence_alignment",
]
