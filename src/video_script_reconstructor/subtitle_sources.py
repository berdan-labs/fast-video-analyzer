"""Transcript-source ranking and interval-level selection."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from .subtitle_validate import ValidationReport, validate_segments

LOGGER = logging.getLogger(__name__)


class EmbeddedSubtitleError(RuntimeError):
    """Raised when a supported embedded subtitle track cannot be extracted safely."""


@dataclass(frozen=True, slots=True)
class EmbeddedSubtitleTrack:
    stream_index: int
    codec_name: str
    output_format: str | None
    output_suffix: str | None
    language: str | None
    title: str | None
    disposition: dict[str, int]
    tags: dict[str, str]
    source_type: str
    authorship: str
    supported: bool
    unsupported_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedEmbeddedSubtitle:
    track: EmbeddedSubtitleTrack
    path: Path
    command: tuple[str, ...]


SOURCE_PRIORITY: dict[str, int] = {
    "user_human_transcript": 7,
    "user_subtitle": 6,
    "embedded_human_subtitle": 5,
    "official_caption": 4,
    "other_human_caption": 3,
    "automatic_caption": 2,
    "local_asr": 1,
}

_ALIASES = {
    "human_transcript": "user_human_transcript",
    "provided_transcript": "user_human_transcript",
    "provided_subtitle": "user_subtitle",
    "embedded_subtitle": "embedded_human_subtitle",
    "creator_caption": "official_caption",
    "auto_caption": "automatic_caption",
    "asr": "local_asr",
    "whisper": "local_asr",
}

_TEXT_SUBTITLE_CODECS: dict[str, tuple[str, str, str]] = {
    "ass": ("ass", ".ass", "ass"),
    "ssa": ("ass", ".ass", "ass"),
    "subrip": ("srt", ".srt", "srt"),
    "srt": ("srt", ".srt", "srt"),
    "text": ("srt", ".srt", "srt"),
    "mov_text": ("srt", ".srt", "srt"),
    "webvtt": ("webvtt", ".vtt", "webvtt"),
}
_AUTO_AUTHORSHIP_RE = re.compile(r"\b(?:auto(?:matic)?|generated|speech[- ]to[- ]text)\b", re.I)
_HUMAN_AUTHORSHIP_RE = re.compile(r"\b(?:human|manual|official|creator)\b", re.I)


def _track_authorship(tags: Mapping[str, str]) -> tuple[str, str]:
    description = " ".join(str(tags.get(key, "")) for key in ("title", "handler_name", "comment"))
    if _AUTO_AUTHORSHIP_RE.search(description):
        return "automatic_caption", "auto_generated"
    if _HUMAN_AUTHORSHIP_RE.search(description):
        return "embedded_human_subtitle", "human"
    return "embedded_human_subtitle", "unknown"


def discover_embedded_subtitle_tracks(
    media_path: str | Path,
    *,
    probe: Any | None = None,
    ffprobe_bin: str = "ffprobe",
) -> list[EmbeddedSubtitleTrack]:
    """Discover embedded subtitle streams through the production FFprobe path."""

    if probe is None:
        from .media_probe import probe_media

        probe = probe_media(media_path, ffprobe_bin=ffprobe_bin)
    tracks: list[EmbeddedSubtitleTrack] = []
    for stream in probe.subtitle_streams:
        codec = str(stream.codec_name or "unknown").casefold()
        conversion = _TEXT_SUBTITLE_CODECS.get(codec)
        tags = {str(key): str(value) for key, value in dict(stream.tags).items()}
        disposition = {str(key): int(value) for key, value in dict(stream.disposition).items()}
        source_type, authorship = _track_authorship(tags)
        tracks.append(
            EmbeddedSubtitleTrack(
                stream_index=int(stream.index),
                codec_name=codec,
                output_format=conversion[0] if conversion else None,
                output_suffix=conversion[1] if conversion else None,
                language=stream.language or tags.get("language"),
                title=tags.get("title") or tags.get("handler_name"),
                disposition=disposition,
                tags=tags,
                source_type=source_type,
                authorship=authorship,
                supported=conversion is not None,
                unsupported_reason=(
                    None
                    if conversion is not None
                    else f"Subtitle codec {codec!r} is not a supported text subtitle codec"
                ),
            )
        )
    return tracks


def build_embedded_subtitle_extract_command(
    media_path: str | Path,
    track: EmbeddedSubtitleTrack,
    output_path: str | Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Build a shell-free FFmpeg argv for one absolute stream index."""

    if not track.supported or track.output_format is None:
        raise EmbeddedSubtitleError(track.unsupported_reason or "Unsupported subtitle track")
    conversion = _TEXT_SUBTITLE_CODECS[track.codec_name]
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(Path(media_path)),
        "-map",
        f"0:{track.stream_index}",
        "-c:s",
        conversion[2],
        "-f",
        track.output_format,
        str(Path(output_path)),
    ]


def _run_embedded_extract(command: Sequence[str]) -> None:
    try:
        completed = subprocess.run(  # noqa: S603
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise EmbeddedSubtitleError(f"FFmpeg executable was not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise EmbeddedSubtitleError("FFmpeg subtitle extraction exceeded 60 seconds") from exc
    if completed.returncode != 0:
        detail = re.sub(r"\s+", " ", completed.stderr).strip()[-1000:]
        raise EmbeddedSubtitleError(
            f"FFmpeg embedded subtitle extraction failed ({completed.returncode}): {detail}"
        )


def extract_embedded_subtitle_track(
    media_path: str | Path,
    track: EmbeddedSubtitleTrack,
    output_dir: str | Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    runner: Callable[[Sequence[str]], None] | None = None,
) -> ExtractedEmbeddedSubtitle:
    """Extract one supported text track atomically and preserve its raw output."""

    source = Path(media_path)
    if not source.is_file():
        raise EmbeddedSubtitleError(f"Media input does not exist: {source}")
    if not track.supported or track.output_suffix is None:
        raise EmbeddedSubtitleError(track.unsupported_reason or "Unsupported subtitle track")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    title = safe_track_label(track.title or track.language or "und")
    target = directory / f"embedded-stream-{track.stream_index}-{title}{track.output_suffix}"
    temporary = (
        directory / f".embedded-stream-{track.stream_index}-{title}.tmp{track.output_suffix}"
    )
    command = build_embedded_subtitle_extract_command(
        source, track, temporary, ffmpeg_bin=ffmpeg_bin
    )
    try:
        (runner or _run_embedded_extract)(command)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise EmbeddedSubtitleError(
                f"FFmpeg produced no subtitle data for stream {track.stream_index}"
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return ExtractedEmbeddedSubtitle(track, target, tuple(command))


def extract_embedded_subtitle_candidates(
    media_path: str | Path,
    output_dir: str | Path,
    *,
    probe: Any | None = None,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[list[ExtractedEmbeddedSubtitle], list[EmbeddedSubtitleTrack]]:
    """Discover all tracks, extract supported text tracks, and retain diagnostics."""

    tracks = discover_embedded_subtitle_tracks(media_path, probe=probe, ffprobe_bin=ffprobe_bin)
    extracted = [
        extract_embedded_subtitle_track(media_path, track, output_dir, ffmpeg_bin=ffmpeg_bin)
        for track in tracks
        if track.supported
    ]
    return extracted, tracks


def safe_track_label(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized[:40] or "track"


@dataclass(slots=True)
class RankedTranscriptCandidate:
    candidate_id: str
    source_type: str
    origin: str
    segments: list[Any]
    language: str | None = None
    human_authored: bool | None = None
    raw_preservation_path: str | None = None
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    reliable_intervals: list[tuple[int, int]] = field(default_factory=list)
    unreliable_intervals: list[tuple[int, int]] = field(default_factory=list)
    issues: list[Any] = field(default_factory=list)
    selection_score: float = 0.0
    selected_intervals: list[tuple[int, int]] = field(default_factory=list)
    decision_rationale: str = ""

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True, slots=True)
class IntervalSelection:
    start_ms: int
    end_ms: int
    candidate_id: str
    score: float
    rationale: str


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _candidate_id(candidate: Any, index: int) -> str:
    existing = _get(candidate, "candidate_id", _get(candidate, "id"))
    if existing:
        return str(existing)
    payload = json.dumps(
        [_get(candidate, "source_type", "unknown"), _get(candidate, "origin", ""), index],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"candidate_{sha256(payload).hexdigest()[:16]}"


def normalize_source_type(source_type: str) -> str:
    normalized = source_type.strip().casefold().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(normalized, normalized)


def source_priority(source_type: str, human_authored: bool | None = None) -> int:
    normalized = normalize_source_type(source_type)
    priority = SOURCE_PRIORITY.get(normalized, 0)
    if human_authored is True and normalized in {"automatic_caption", "local_asr"}:
        priority = max(priority, SOURCE_PRIORITY["other_human_caption"])
    return priority


def score_candidate(candidate: Any, report: ValidationReport) -> float:
    """Combine quality with the provenance preference without blindly trusting it."""

    provenance = source_priority(
        str(_get(candidate, "source_type", "unknown")),
        _get(candidate, "human_authored"),
    )
    provenance_component = provenance / max(SOURCE_PRIORITY.values())
    timed_segments = int(report.metrics.get("timed_segment_count") or 0)
    segment_count = int(report.metrics.get("segment_count") or 0)
    timing_component = timed_segments / segment_count if segment_count else 0.0
    speech_coverage = report.metrics.get("speech_coverage")
    coverage_component = float(speech_coverage) if speech_coverage is not None else 0.75
    score = (
        report.quality_score * 0.62
        + provenance_component * 0.20
        + timing_component * 0.08
        + max(0.0, min(1.0, coverage_component)) * 0.10
    )
    if not report.usable:
        score *= 0.35
    return round(max(0.0, min(1.0, score)), 6)


def rank_candidates(
    candidates: Sequence[Any],
    *,
    media_duration_ms: int | None = None,
    speech_intervals: Sequence[tuple[int, int]] | None = None,
    expected_language: str | None = None,
) -> list[RankedTranscriptCandidate]:
    """Evaluate every candidate and return a deterministic best-first ranking."""

    ranked: list[RankedTranscriptCandidate] = []
    for index, candidate in enumerate(candidates):
        segments = list(_get(candidate, "segments", []))
        report = validate_segments(
            segments,
            media_duration_ms=media_duration_ms,
            speech_intervals=speech_intervals,
            expected_language=expected_language,
        )
        candidate_id = _candidate_id(candidate, index)
        score = score_candidate(candidate, report)
        priority = source_priority(
            str(_get(candidate, "source_type", "unknown")),
            _get(candidate, "human_authored"),
        )
        ranked.append(
            RankedTranscriptCandidate(
                candidate_id=candidate_id,
                source_type=normalize_source_type(str(_get(candidate, "source_type", "unknown"))),
                origin=str(_get(candidate, "origin", "")),
                language=_get(candidate, "language"),
                human_authored=_get(candidate, "human_authored"),
                raw_preservation_path=_get(candidate, "raw_preservation_path"),
                segments=segments,
                quality_metrics=dict(report.metrics)
                | {"quality_score": report.quality_score, "usable": report.usable},
                reliable_intervals=list(report.reliable_intervals),
                unreliable_intervals=list(report.unreliable_intervals),
                issues=list(report.issues),
                selection_score=score,
                decision_rationale=(
                    f"quality={report.quality_score:.3f}; provenance_priority="
                    f"{priority}; "
                    f"usable={report.usable}"
                ),
            )
        )
    ranked.sort(
        key=lambda item: (
            -item.selection_score,
            -source_priority(item.source_type, item.human_authored),
            item.candidate_id,
        )
    )
    LOGGER.info(
        "ranked_transcript_candidates",
        extra={
            "candidate_count": len(ranked),
            "best_candidate_id": ranked[0].candidate_id if ranked else None,
        },
    )
    return ranked


def _covers(intervals: Sequence[tuple[int, int]], start: int, end: int) -> bool:
    return any(left <= start and right >= end for left, right in intervals)


def select_candidate_intervals(
    candidates: Sequence[Any],
    *,
    media_duration_ms: int | None = None,
    speech_intervals: Sequence[tuple[int, int]] | None = None,
    expected_language: str | None = None,
) -> list[IntervalSelection]:
    """Choose the strongest reliable candidate independently for each interval.

    This function never concatenates complete competing tracks.  Boundaries are
    derived from all candidates' reliable ranges, and each atomic interval is
    assigned to at most one source.
    """

    if candidates and isinstance(candidates[0], RankedTranscriptCandidate):
        ranked = list(candidates)
    else:
        ranked = rank_candidates(
            candidates,
            media_duration_ms=media_duration_ms,
            speech_intervals=speech_intervals,
            expected_language=expected_language,
        )
    boundaries: set[int] = set()
    for candidate in ranked:
        for start, end in candidate.reliable_intervals:
            boundaries.update((start, end))
    if media_duration_ms is not None:
        boundaries.update((0, media_duration_ms))
    ordered = sorted(boundaries)
    selections: list[IntervalSelection] = []
    for start, end in zip(ordered, ordered[1:], strict=False):
        if end <= start:
            continue
        eligible = [
            candidate for candidate in ranked if _covers(candidate.reliable_intervals, start, end)
        ]
        if not eligible:
            continue
        winner = eligible[0]
        rationale = (
            f"highest validated interval score ({winner.selection_score:.3f}) "
            f"among {len(eligible)} reliable source(s)"
        )
        if (
            selections
            and selections[-1].candidate_id == winner.candidate_id
            and selections[-1].end_ms == start
        ):
            previous = selections[-1]
            selections[-1] = IntervalSelection(
                previous.start_ms, end, winner.candidate_id, winner.selection_score, rationale
            )
        else:
            selections.append(
                IntervalSelection(
                    start, end, winner.candidate_id, winner.selection_score, rationale
                )
            )
    by_id = {candidate.candidate_id: candidate for candidate in ranked}
    for selection in selections:
        by_id[selection.candidate_id].selected_intervals.append(
            (selection.start_ms, selection.end_ms)
        )
    return selections


def select_candidate(candidates: Sequence[Any], **kwargs: Any) -> RankedTranscriptCandidate | None:
    """Select the highest-ranked usable whole-track candidate, if one exists."""

    ranked = rank_candidates(candidates, **kwargs)
    return next(
        (candidate for candidate in ranked if candidate.quality_metrics.get("usable")), None
    )


__all__ = [
    "EmbeddedSubtitleError",
    "EmbeddedSubtitleTrack",
    "ExtractedEmbeddedSubtitle",
    "IntervalSelection",
    "RankedTranscriptCandidate",
    "SOURCE_PRIORITY",
    "build_embedded_subtitle_extract_command",
    "discover_embedded_subtitle_tracks",
    "extract_embedded_subtitle_candidates",
    "extract_embedded_subtitle_track",
    "normalize_source_type",
    "rank_candidates",
    "score_candidate",
    "select_candidate",
    "select_candidate_intervals",
    "source_priority",
]
