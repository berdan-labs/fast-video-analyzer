from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Literal

from .errors import BlockedError, InputError, ValidationFailure

_SHOWINFO = re.compile(
    r"\bn:\s*(?P<n>\d+)\s+pts:\s*(?P<pts>-?\d+)\s+pts_time:\s*(?P<time>-?[0-9.eE+]+)"
    r"(?:.*?\bs:\s*(?P<width>\d+)x(?P<height>\d+))?"
)
_TIME_BASE = re.compile(r"config in time_base:\s*(?P<base>\d+/\d+)")
# Container duration can include a small audio/encoder tail after the final
# decodable video frame.  A periodic request inside that tail is not an
# approximate frame: it is an invalid request, and the exact extractor is
# correct to refuse it.  Keep the safety sample one quarter-second inside the
# declared duration so the strict cadence remains intact without guessing a
# timestamp.  Structural/contextual candidates are measured independently.
_PERIODIC_TAIL_GUARD_MS = 250


@dataclass(frozen=True)
class SurveyCandidate:
    candidate_id: str
    requested_ms: int
    actual_ms: int | None
    raw_pts: int | None
    time_base: str | None
    reasons: tuple[str, ...]
    score: float
    timestamp_source: str

    @property
    def importance(self) -> float:
        """Importance derived from measured survey reasons (not semantics)."""

        return survey_candidate_importance(self)

    @property
    def importance_tier(self) -> str:
        """Stable audit band used when a survey candidate becomes evidence."""

        return survey_candidate_importance_tier(self)


_SURVEY_IMPORTANCE_BY_REASON = {
    # Hard cuts and explicit text changes are the strongest measured state
    # transitions.  They remain protected even when their visual footprint is
    # small or a periodic sample lands nearby.
    "scene_cut": 0.98,
    "ocr_change": 0.98,
    "perceptual_change": 0.90,
    "adaptive_frame_difference": 0.88,
    "motion_change": 0.72,
    "post_motion_stable": 0.70,
    "deictic_speech_reference": 0.76,
    "chapter_boundary": 0.74,
    # Periodic safety samples are important for temporal coverage but are not
    # themselves evidence of a state change.  Keeping this distinction lets
    # downstream selection collapse repetitive low-value context safely while
    # preserving the strict no-gap schedule where requested.
    "periodic_safety": 0.25,
}


def _normalized_survey_reason(reason: str) -> str:
    return str(reason).strip().casefold().replace("-", "_")


def survey_candidate_importance(candidate: SurveyCandidate) -> float:
    """Map structural survey signals to a deterministic [0, 1] importance.

    This is deliberately reason-based rather than a semantic claim: survey
    detection has not yet interpreted what is visible.  The strongest measured
    reason wins, with the detector score providing a bounded fallback for
    custom signal producers.
    """

    measured = [
        _SURVEY_IMPORTANCE_BY_REASON.get(_normalized_survey_reason(reason), 0.0)
        for reason in candidate.reasons
    ]
    return max(0.0, min(1.0, max([*measured, float(candidate.score)])))


def survey_candidate_importance_tier(candidate: SurveyCandidate) -> str:
    importance = survey_candidate_importance(candidate)
    if importance >= 0.90:
        return "very_high"
    if importance >= 0.75:
        return "high"
    if importance >= 0.45:
        return "supporting"
    return "low"


def survey_candidate_is_protected(candidate: SurveyCandidate) -> bool:
    """Whether measured survey evidence should survive visual deduplication."""

    reasons = {_normalized_survey_reason(reason) for reason in candidate.reasons}
    return bool(
        survey_candidate_importance(candidate) >= 0.75
        or reasons.intersection(
            {
                "scene_cut",
                "ocr_change",
                "deictic_speech_reference",
                "chapter_boundary",
            }
        )
    )


@dataclass(frozen=True)
class SurveySignal:
    actual_ms: int
    raw_pts: int | None = None
    time_base: str | None = None
    frame_difference: float = 0.0
    perceptual_change: float = 0.0
    motion: float = 0.0
    blur: float = 0.0
    ocr_text: str | None = None


@dataclass(frozen=True)
class SurveyFrameTiming:
    output_index: int
    raw_pts: int
    actual_ms: int
    time_base: str | None
    width: int
    height: int
    timestamp_source: str = "ffmpeg-showinfo"


@dataclass(frozen=True)
class SurveyFrame:
    """A lossless frame emitted by a shared contextual survey pass."""

    branch: Literal["hard", "periodic"]
    requested_ms: int
    path: Path
    timing: SurveyFrameTiming


def _normalized_ocr(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def adaptive_signal_candidates(
    signals: Iterable[SurveySignal],
    *,
    minimum_difference: float = 0.04,
    motion_threshold: float = 0.35,
    perceptual_threshold: float = 0.10,
) -> tuple[SurveyCandidate, ...]:
    """Turn decoded survey measurements into adaptive, evidence-preserving candidates."""
    samples = sorted(signals, key=lambda item: item.actual_ms)
    if not samples:
        return ()
    if any(item.actual_ms < 0 for item in samples):
        raise InputError("survey signal times cannot be negative")
    differences = [max(0.0, item.frame_difference) for item in samples]
    baseline = median(differences)
    deviations = [abs(value - baseline) for value in differences]
    threshold = max(minimum_difference, baseline + 2.5 * median(deviations))
    candidates: list[SurveyCandidate] = []
    previous: SurveySignal | None = None
    for sample in samples:
        reasons: list[str] = []
        strengths: list[float] = []
        if sample.frame_difference >= threshold:
            reasons.append("adaptive_frame_difference")
            strengths.append(min(1.0, sample.frame_difference))
        if sample.perceptual_change >= perceptual_threshold:
            reasons.append("perceptual_change")
            strengths.append(min(1.0, sample.perceptual_change))
        if sample.motion >= motion_threshold:
            reasons.append("motion_change")
            strengths.append(min(1.0, sample.motion))
        if previous is not None and _normalized_ocr(previous.ocr_text) != _normalized_ocr(
            sample.ocr_text
        ):
            if _normalized_ocr(previous.ocr_text) or _normalized_ocr(sample.ocr_text):
                reasons.append("ocr_change")
                strengths.append(1.0)
        # A sharp frame immediately following blur is often the readable settled state.
        if previous is not None and previous.blur >= 0.5 and sample.blur <= 0.2:
            reasons.append("post_motion_stable")
            strengths.append(0.7)
        if reasons:
            candidates.append(
                SurveyCandidate(
                    candidate_id=f"VC{len(candidates) + 1:06d}",
                    requested_ms=sample.actual_ms,
                    actual_ms=sample.actual_ms,
                    raw_pts=sample.raw_pts,
                    time_base=sample.time_base,
                    reasons=tuple(reasons),
                    score=max(strengths),
                    timestamp_source="decoded-survey-signal",
                )
            )
        previous = sample
    return tuple(candidates)


def contextual_candidates(
    *,
    chapter_times_ms: Iterable[int] = (),
    speech_reference_times_ms: Iterable[int] = (),
) -> tuple[SurveyCandidate, ...]:
    tagged = [(time_ms, "chapter_boundary", 0.65) for time_ms in chapter_times_ms]
    tagged.extend(
        (time_ms, "deictic_speech_reference", 0.75) for time_ms in speech_reference_times_ms
    )
    result: list[SurveyCandidate] = []
    for index, (time_ms, reason, score) in enumerate(sorted(tagged), 1):
        if time_ms < 0:
            raise InputError("context candidate times cannot be negative")
        result.append(
            SurveyCandidate(
                candidate_id=f"VC{index:06d}",
                requested_ms=time_ms,
                actual_ms=None,
                raw_pts=None,
                time_base=None,
                reasons=(reason,),
                score=score,
                timestamp_source="context-aligned-candidate",
            )
        )
    return tuple(result)


def periodic_candidate_times(
    duration_ms: int,
    *,
    interval_seconds: float = 30.0,
    strict: bool = True,
    tail_guard_ms: int = _PERIODIC_TAIL_GUARD_MS,
) -> tuple[int, ...]:
    """Return safety-sampling times while enforcing the strict 30-second ceiling.

    The declared container duration is not always the timestamp of a
    decodable video frame (audio padding and muxer tails are common).  When
    the final periodic point would fall inside ``tail_guard_ms`` of that
    boundary, move only that point inward.  This preserves the temporal
    coverage contract and never fabricates a measured frame timestamp.
    """
    if duration_ms < 0:
        raise InputError("duration_ms cannot be negative")
    if interval_seconds <= 0:
        raise InputError("periodic safety interval must be positive")
    if tail_guard_ms < 0:
        raise InputError("periodic tail guard cannot be negative")
    if strict and interval_seconds > 30:
        raise InputError(
            "strict mode requires a periodic safety interval no greater than 30 seconds"
        )
    interval_ms = max(1, round(interval_seconds * 1000))
    if duration_ms == 0:
        return (0,)
    times = list(range(0, duration_ms, interval_ms))
    if not times:
        times.append(0)
    # Add a near-tail sample only when the final uncovered span would exceed the interval.
    if duration_ms - times[-1] > interval_ms:
        times.append(max(0, duration_ms - 1))
    if tail_guard_ms and duration_ms - times[-1] < tail_guard_ms:
        guarded_tail = max(0, duration_ms - tail_guard_ms)
        # Remove any prior samples that the inward move passes, then append
        # the guarded point. This keeps the schedule sorted/unique even when
        # a caller deliberately chooses an interval shorter than the guard.
        times = [time_ms for time_ms in times if time_ms <= guarded_tail]
        if not times or times[-1] != guarded_tail:
            times.append(guarded_tail)
    return tuple(times)


def periodic_candidates(
    duration_ms: int,
    *,
    interval_seconds: float = 30.0,
    strict: bool = True,
    tail_guard_ms: int = _PERIODIC_TAIL_GUARD_MS,
) -> tuple[SurveyCandidate, ...]:
    return tuple(
        SurveyCandidate(
            candidate_id=f"VC{index:06d}",
            requested_ms=time_ms,
            actual_ms=None,
            raw_pts=None,
            time_base=None,
            reasons=("periodic_safety",),
            score=0.25,
            timestamp_source="requested-candidate",
        )
        for index, time_ms in enumerate(
            periodic_candidate_times(
                duration_ms,
                interval_seconds=interval_seconds,
                strict=strict,
                tail_guard_ms=tail_guard_ms,
            ),
            1,
        )
    )


def parse_scene_showinfo(stderr: str) -> tuple[tuple[int, int], ...]:
    """Parse measured raw PTS and presentation milliseconds from FFmpeg showinfo."""
    parsed: list[tuple[int, int]] = []
    for match in _SHOWINFO.finditer(stderr):
        try:
            actual_ms = int((Decimal(match.group("time")) * 1000).to_integral_value())
        except InvalidOperation as exc:
            raise ValidationFailure("Invalid pts_time in FFmpeg scene output") from exc
        parsed.append((int(match.group("pts")), actual_ms))
    return tuple(parsed)


def parse_survey_frame_timings(stderr: str) -> tuple[SurveyFrameTiming, ...]:
    """Parse full measured frame records, including dimensions, from showinfo."""

    time_base = _showinfo_time_base(stderr)
    parsed: list[SurveyFrameTiming] = []
    for match in _SHOWINFO.finditer(stderr):
        try:
            actual_ms = int((Decimal(match.group("time")) * 1000).to_integral_value())
        except InvalidOperation as exc:
            raise ValidationFailure("Invalid pts_time in FFmpeg survey frame output") from exc
        width = match.group("width")
        height = match.group("height")
        if width is None or height is None:
            raise ValidationFailure("FFmpeg survey frame output omitted frame dimensions")
        parsed.append(
            SurveyFrameTiming(
                output_index=int(match.group("n")),
                raw_pts=int(match.group("pts")),
                actual_ms=actual_ms,
                time_base=time_base,
                width=int(width),
                height=int(height),
            )
        )
    return tuple(parsed)


def _showinfo_time_base(stderr: str) -> str | None:
    match = _TIME_BASE.search(stderr)
    return match.group("base") if match else None


def build_scene_detection_command(
    media_path: Path,
    *,
    threshold: float = 0.3,
    video_stream_index: int = 0,
    ffmpeg_threads: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    if not 0 <= threshold <= 1:
        raise InputError("scene threshold must be between 0 and 1")
    if video_stream_index < 0:
        raise InputError("video_stream_index cannot be negative")
    if ffmpeg_threads is not None and ffmpeg_threads <= 0:
        raise InputError("ffmpeg_threads must be positive when provided")
    # The expression is generated only from a validated float and passed as one argv item.
    filter_graph = f"select=gt(scene\\,{threshold:.8g}),showinfo"
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        *( ["-threads", str(ffmpeg_threads)] if ffmpeg_threads is not None else [] ),
        "-i",
        str(media_path),
        "-map",
        f"0:v:{video_stream_index}",
        "-vf",
        filter_graph,
        "-an",
        "-sn",
        "-f",
        "null",
        "-",
    ]


def build_adaptive_detection_command(
    media_path: Path,
    *,
    threshold: float = 0.003,
    sample_fps: float = 2.0,
    video_stream_index: int = 0,
    ffmpeg_threads: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    if not 0 <= threshold <= 1:
        raise InputError("adaptive difference threshold must be between 0 and 1")
    if not 0 < sample_fps <= 30:
        raise InputError("adaptive survey sample_fps must be greater than 0 and no more than 30")
    if ffmpeg_threads is not None and ffmpeg_threads <= 0:
        raise InputError("ffmpeg_threads must be positive when provided")
    filter_graph = f"fps={sample_fps:.8g},select=gt(scene\\,{threshold:.8g}),showinfo"
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        *( ["-threads", str(ffmpeg_threads)] if ffmpeg_threads is not None else [] ),
        "-i",
        str(media_path),
        "-map",
        f"0:v:{video_stream_index}",
        "-vf",
        filter_graph,
        "-an",
        "-sn",
        "-f",
        "null",
        "-",
    ]


def build_combined_survey_command(
    media_path: Path,
    *,
    scene_threshold: float = 0.3,
    adaptive_threshold: float = 0.003,
    adaptive_sample_fps: float = 2.0,
    video_stream_index: int = 0,
    ffmpeg_threads: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Build one decode pass for hard-cut and adaptive survey branches.

    Named ``showinfo`` instances keep the two candidate families attributable
    in stderr even though FFmpeg shares the decoded input through ``split``.
    Both detector branches terminate in ``nullsink`` so an empty branch cannot
    make the null muxer fail and trigger two expensive fallback decodes.  A
    pass-through copy of the decoded input is the sole mapped output.  Mapping
    the complete input, rather than a short synthetic keepalive, is essential:
    FFmpeg may otherwise stop the graph before late detector measurements have
    been produced.
    """

    if not 0 <= scene_threshold <= 1:
        raise InputError("scene threshold must be between 0 and 1")
    if not 0 <= adaptive_threshold <= 1:
        raise InputError("adaptive difference threshold must be between 0 and 1")
    if not 0 < adaptive_sample_fps <= 30:
        raise InputError("adaptive survey sample_fps must be greater than 0 and no more than 30")
    if video_stream_index < 0:
        raise InputError("video_stream_index cannot be negative")
    if ffmpeg_threads is not None and ffmpeg_threads <= 0:
        raise InputError("ffmpeg_threads must be positive when provided")
    filter_graph = (
        f"[0:v:{video_stream_index}]split=3[hard_input][adaptive_input][survey_output];"
        f"[hard_input]select=gt(scene\\,{scene_threshold:.8g}),showinfo@hard,nullsink;"
        f"[adaptive_input]fps={adaptive_sample_fps:.8g},"
        f"select=gt(scene\\,{adaptive_threshold:.8g}),showinfo@adaptive,nullsink;"
        "[survey_output]null[keepalive]"
    )
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        *( ["-threads", str(ffmpeg_threads)] if ffmpeg_threads is not None else [] ),
        "-i",
        str(media_path),
        "-filter_complex",
        filter_graph,
        "-map",
        "[keepalive]",
        "-an",
        "-sn",
        "-f",
        "null",
        "-",
    ]


def _survey_window_select_filter(requested_times_ms: Sequence[int]) -> str:
    """Select one measured frame per bounded contextual request window."""

    if not requested_times_ms:
        raise InputError("shared survey frame emission requires at least one request")
    clauses: list[str] = []
    for index, requested_ms in enumerate(requested_times_ms):
        if requested_ms < 0:
            raise InputError("shared survey frame request times cannot be negative")
        start = Decimal(requested_ms) / Decimal(1000)
        end = start + Decimal("0.250")
        guard = f"lt(prev_selected_t\\,{start:f})"
        if index == 0:
            guard = f"(isnan(prev_selected_pts)+{guard})"
        clauses.append(f"(gte(t\\,{start:f})*lt(t\\,{end:f})*{guard})")
    return f"select='{'+'.join(clauses)}',showinfo@periodic[periodic_output]"


def build_combined_survey_frame_command(
    media_path: Path,
    output_dir: Path,
    periodic_times_ms: Sequence[int],
    *,
    scene_threshold: float = 0.3,
    adaptive_threshold: float = 0.003,
    adaptive_sample_fps: float = 2.0,
    video_stream_index: int = 0,
    ffmpeg_threads: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Build one survey decode that emits safe periodic/contextual frames.

    Hard-cut frames are emitted through a sentinel-prefixed PNG stream so an
    empty hard-cut branch remains muxable; the detector discards that sentinel
    and retains only measured hard-cut frames. Adaptive samples remain
    measurement-only and terminate through ``nullsink`` because exact
    per-request extraction can land on a different source frame after a
    sample decision. The periodic/contextual branch always has requests and
    emits PNGs equivalent to exact extraction.
    """

    if not 0 <= scene_threshold <= 1:
        raise InputError("scene threshold must be between 0 and 1")
    if not 0 <= adaptive_threshold <= 1:
        raise InputError("adaptive difference threshold must be between 0 and 1")
    if not 0 < adaptive_sample_fps <= 30:
        raise InputError("adaptive survey sample_fps must be greater than 0 and no more than 30")
    if video_stream_index < 0:
        raise InputError("video_stream_index cannot be negative")
    if ffmpeg_threads is not None and ffmpeg_threads <= 0:
        raise InputError("ffmpeg_threads must be positive when provided")
    ordered_times = tuple(sorted(set(int(value) for value in periodic_times_ms)))
    periodic_filter = _survey_window_select_filter(ordered_times)
    filter_graph = (
        f"[0:v:{video_stream_index}]split=4[hard_input][adaptive_input][periodic_input][hard_dummy_input];"
        f"[hard_input]select=gt(scene\\,{scene_threshold:.8g}),showinfo@hard[hard_measured];"
        f"[hard_dummy_input]trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS[hard_dummy];"
        # Keep the sentinel's PTS distinct from the first measured cut. FFmpeg
        # 7.1's VFR image2 muxer drops equal-PTS frames, while newer builds
        # preserve them; a one-clock-tick offset retains both without changing
        # the measured candidate timestamps (showinfo is upstream).
        f"[hard_measured]setpts=PTS-STARTPTS+1/TB[hard_reset];"
        f"[hard_dummy][hard_reset]concat=n=2:v=1:a=0[hard_output];"
        f"[adaptive_input]fps={adaptive_sample_fps:.8g},"
        f"select=gt(scene\\,{adaptive_threshold:.8g}),showinfo@adaptive,nullsink;"
        f"[periodic_input]{periodic_filter}"
    )
    periodic_pattern = str(output_dir / "periodic-%06d.png")
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        *(["-threads", str(ffmpeg_threads)] if ffmpeg_threads is not None else []),
        "-i",
        str(media_path),
        "-filter_complex",
        filter_graph,
        "-map",
        "[hard_output]",
        "-an",
        "-sn",
        "-fps_mode",
        "vfr",
        "-c:v",
        "png",
        "-f",
        "image2",
        "-start_number",
        "1",
        "-y",
        str(output_dir / "hard-%06d.png"),
        "-map",
        "[periodic_output]",
        "-an",
        "-sn",
        "-fps_mode",
        "vfr",
        "-c:v",
        "png",
        "-f",
        "image2",
        "-start_number",
        "1",
        "-y",
        periodic_pattern,
    ]


def _execute_detection(
    command: list[str], *, ffmpeg_bin: str, timeout_seconds: float, operation: str
) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise BlockedError(f"FFmpeg executable was not found: {ffmpeg_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise BlockedError(f"{operation} exceeded the {timeout_seconds:g}s timeout") from exc
    if completed.returncode != 0:
        detail = re.sub(r"\s+", " ", completed.stderr).strip()[-1000:]
        raise ValidationFailure(
            f"FFmpeg {operation.casefold()} failed with exit code {completed.returncode}: {detail}"
        )
    return completed.stderr


def _labeled_showinfo(stderr: str, label: str) -> str:
    marker = f"showinfo@{label}"
    return "\n".join(line for line in stderr.splitlines() if marker in line)


def _candidates_from_measurements(
    measurements: Sequence[tuple[int, int]],
    *,
    time_base: str | None,
    reason: str,
    score: float,
    timestamp_source: str,
) -> tuple[SurveyCandidate, ...]:
    return tuple(
        SurveyCandidate(
            candidate_id=f"VC{index:06d}",
            requested_ms=actual_ms,
            actual_ms=actual_ms,
            raw_pts=pts,
            time_base=time_base,
            reasons=(reason,),
            score=score,
            timestamp_source=timestamp_source,
        )
        for index, (pts, actual_ms) in enumerate(measurements, 1)
    )


def detect_scene_candidates(
    media_path: str | Path,
    *,
    threshold: float = 0.3,
    video_stream_index: int = 0,
    ffmpeg_threads: int | None = 4,
    ffmpeg_bin: str = "ffmpeg",
    timeout_seconds: float = 600.0,
) -> tuple[SurveyCandidate, ...]:
    path = Path(media_path).expanduser()
    if not path.is_file():
        raise InputError(f"Video source is not a file: {path}")
    command = build_scene_detection_command(
        path,
        threshold=threshold,
        video_stream_index=video_stream_index,
        ffmpeg_threads=ffmpeg_threads,
        ffmpeg_bin=ffmpeg_bin,
    )
    stderr = _execute_detection(
        command, ffmpeg_bin=ffmpeg_bin, timeout_seconds=timeout_seconds, operation="Scene detection"
    )
    measurements = parse_scene_showinfo(stderr)
    time_base = _showinfo_time_base(stderr)
    return _candidates_from_measurements(
        measurements,
        time_base=time_base,
        reason="scene_cut",
        score=0.9,
        timestamp_source="ffmpeg-showinfo",
    )


def detect_adaptive_candidates(
    media_path: str | Path,
    *,
    threshold: float = 0.003,
    sample_fps: float = 2.0,
    video_stream_index: int = 0,
    ffmpeg_threads: int | None = 4,
    ffmpeg_bin: str = "ffmpeg",
    timeout_seconds: float = 600.0,
    max_candidates: int | None = None,
) -> tuple[SurveyCandidate, ...]:
    path = Path(media_path).expanduser()
    if not path.is_file():
        raise InputError(f"Video source is not a file: {path}")
    command = build_adaptive_detection_command(
        path,
        threshold=threshold,
        sample_fps=sample_fps,
        video_stream_index=video_stream_index,
        ffmpeg_threads=ffmpeg_threads,
        ffmpeg_bin=ffmpeg_bin,
    )
    stderr = _execute_detection(
        command,
        ffmpeg_bin=ffmpeg_bin,
        timeout_seconds=timeout_seconds,
        operation="Adaptive frame-difference survey",
    )
    measurements = parse_scene_showinfo(stderr)
    if max_candidates is not None and len(measurements) > max_candidates:
        raise BlockedError(
            f"Adaptive survey found {len(measurements)} changes, exceeding the explicit "
            f"resource limit of {max_candidates}; no candidates were silently truncated"
        )
    time_base = _showinfo_time_base(stderr)
    return _candidates_from_measurements(
        measurements,
        time_base=time_base,
        reason="adaptive_frame_difference",
        score=0.7,
        timestamp_source="ffmpeg-sampled-scene-score-showinfo",
    )


def detect_combined_survey_candidates(
    media_path: str | Path,
    *,
    scene_threshold: float = 0.3,
    adaptive_threshold: float = 0.003,
    adaptive_sample_fps: float = 2.0,
    video_stream_index: int = 0,
    ffmpeg_threads: int | None = 4,
    ffmpeg_bin: str = "ffmpeg",
    timeout_seconds: float = 600.0,
) -> tuple[tuple[SurveyCandidate, ...], tuple[SurveyCandidate, ...]]:
    """Decode hard-cut and adaptive survey branches in one FFmpeg process."""

    path = Path(media_path).expanduser()
    if not path.is_file():
        raise InputError(f"Video source is not a file: {path}")
    command = build_combined_survey_command(
        path,
        scene_threshold=scene_threshold,
        adaptive_threshold=adaptive_threshold,
        adaptive_sample_fps=adaptive_sample_fps,
        video_stream_index=video_stream_index,
        ffmpeg_threads=ffmpeg_threads,
        ffmpeg_bin=ffmpeg_bin,
    )
    stderr = _execute_detection(
        command,
        ffmpeg_bin=ffmpeg_bin,
        timeout_seconds=timeout_seconds,
        operation="combined visual survey",
    )
    hard_stderr = _labeled_showinfo(stderr, "hard")
    adaptive_stderr = _labeled_showinfo(stderr, "adaptive")
    hard = _candidates_from_measurements(
        parse_scene_showinfo(hard_stderr),
        time_base=_showinfo_time_base(hard_stderr),
        reason="scene_cut",
        score=0.9,
        timestamp_source="ffmpeg-showinfo",
    )
    adaptive = _candidates_from_measurements(
        parse_scene_showinfo(adaptive_stderr),
        time_base=_showinfo_time_base(adaptive_stderr),
        reason="adaptive_frame_difference",
        score=0.7,
        timestamp_source="ffmpeg-sampled-scene-score-showinfo",
    )
    return hard, adaptive


def detect_combined_survey_frames(
    media_path: str | Path,
    output_dir: str | Path,
    periodic_times_ms: Sequence[int],
    *,
    scene_threshold: float = 0.3,
    adaptive_threshold: float = 0.003,
    adaptive_sample_fps: float = 2.0,
    video_stream_index: int = 0,
    ffmpeg_threads: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
    timeout_seconds: float = 600.0,
) -> tuple[tuple[SurveyCandidate, ...], tuple[SurveyCandidate, ...], tuple[SurveyFrame, ...]]:
    """Run a shared survey and emit exact-safe hard/periodic PNG frames.

    Hard-cut and periodic frame bytes are equivalent to the existing exact
    extraction path on the reference media. Adaptive measurements are returned
    as candidates but intentionally remain null-muxed; callers should extract
    those requested timestamps through the guarded exact path.
    """

    path = Path(media_path).expanduser()
    directory = Path(output_dir).expanduser()
    if not path.is_file():
        raise InputError(f"Video source is not a file: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    ordered_times = tuple(sorted(set(int(value) for value in periodic_times_ms)))
    command = build_combined_survey_frame_command(
        path,
        directory,
        ordered_times,
        scene_threshold=scene_threshold,
        adaptive_threshold=adaptive_threshold,
        adaptive_sample_fps=adaptive_sample_fps,
        video_stream_index=video_stream_index,
        ffmpeg_threads=ffmpeg_threads,
        ffmpeg_bin=ffmpeg_bin,
    )
    stderr = _execute_detection(
        command,
        ffmpeg_bin=ffmpeg_bin,
        timeout_seconds=timeout_seconds,
        operation="combined survey frame emission",
    )
    hard_stderr = _labeled_showinfo(stderr, "hard")
    adaptive_stderr = _labeled_showinfo(stderr, "adaptive")
    periodic_stderr = _labeled_showinfo(stderr, "periodic")
    hard_timings = parse_survey_frame_timings(hard_stderr)
    adaptive_timings = parse_survey_frame_timings(adaptive_stderr)
    periodic_timings = parse_survey_frame_timings(periodic_stderr)
    hard_files_with_sentinel = sorted(directory.glob("hard-*.png"))
    periodic_files = sorted(directory.glob("periodic-*.png"))
    if len(hard_files_with_sentinel) != len(hard_timings) + 1:
        raise ValidationFailure(
            "shared hard-cut survey did not emit exactly one sentinel plus each measured frame"
        )
    if len(periodic_files) != len(periodic_timings) or len(periodic_timings) != len(ordered_times):
        raise ValidationFailure(
            "shared periodic survey did not emit exactly one measured frame per request"
        )
    # The sentinel is the first frame in the concatenated hard-cut stream. It
    # guarantees a valid image2 output even when no scene cut is detected.
    hard_files = hard_files_with_sentinel[1:]
    hard = _candidates_from_measurements(
        tuple((timing.raw_pts, timing.actual_ms) for timing in hard_timings),
        time_base=_showinfo_time_base(hard_stderr),
        reason="scene_cut",
        score=0.9,
        timestamp_source="ffmpeg-showinfo",
    )
    adaptive = _candidates_from_measurements(
        tuple((timing.raw_pts, timing.actual_ms) for timing in adaptive_timings),
        time_base=_showinfo_time_base(adaptive_stderr),
        reason="adaptive_frame_difference",
        score=0.7,
        timestamp_source="ffmpeg-sampled-scene-score-showinfo",
    )
    hard_frames = [
        SurveyFrame(branch="hard", requested_ms=timing.actual_ms, path=frame, timing=timing)
        for timing, frame in zip(hard_timings, hard_files, strict=True)
    ]
    periodic_frames = [
        SurveyFrame(branch="periodic", requested_ms=requested_ms, path=frame, timing=timing)
        for requested_ms, timing, frame in zip(
            ordered_times, periodic_timings, periodic_files, strict=True
        )
    ]
    return hard, adaptive, tuple((*hard_frames, *periodic_frames))


def merge_survey_candidates(
    candidate_sets: Iterable[Sequence[SurveyCandidate]],
    *,
    merge_tolerance_ms: int = 250,
    adaptive_cluster_tolerance_ms: int = 1_000,
) -> tuple[SurveyCandidate, ...]:
    """Merge equivalent survey timestamps while collapsing sustained motion runs.

    The hard-cut/periodic merge window is intentionally small because a nearby
    subtitle or chapter boundary must remain attributable to its measured
    timestamp.  Adaptive scene sampling is different: at 2fps a presenter
    pan, cursor movement, or codec shimmer can produce a candidate every
    500ms for several seconds.  Those candidates carry no OCR, chapter, or
    hard-cut provenance and do not each warrant a semantic packet.  Collapse
    only adjacent *adaptive-only* candidates inside the bounded cluster
    window; explicit scene/OCR/chapter candidates still use the strict merge
    window and remain separate when they are farther apart.  Periodic safety
    samples are never absorbed by the wider window, so the strict temporal
    coverage contract is unchanged.

    This is an acceleration-only reduction: the representative keeps the
    strongest measured score and unions all reasons, while scene/OCR/context
    candidates retain their existing protection rules.
    """
    if merge_tolerance_ms < 0:
        raise InputError("merge_tolerance_ms cannot be negative")
    if adaptive_cluster_tolerance_ms < merge_tolerance_ms:
        raise InputError(
            "adaptive_cluster_tolerance_ms cannot be lower than merge_tolerance_ms"
        )

    # ``perceptual_change`` is intentionally excluded: its measured
    # importance is very-high and it may represent a real state transition,
    # whereas the two remaining reasons are the noisy motion stream we want
    # to coalesce.
    adaptive_only_reasons = {
        "adaptive_frame_difference",
        "motion_change",
    }

    def is_adaptive_only(candidate: SurveyCandidate) -> bool:
        reasons = {
            _normalized_survey_reason(reason) for reason in candidate.reasons
        }
        # An empty set is deliberately false: custom candidates without a
        # reason must continue to use only the strict merge window.
        return bool(reasons) and reasons <= adaptive_only_reasons

    ordered = sorted(
        (candidate for candidates in candidate_sets for candidate in candidates),
        key=lambda item: (
            item.actual_ms if item.actual_ms is not None else item.requested_ms,
            -item.score,
            item.reasons,
        ),
    )
    groups: list[list[SurveyCandidate]] = []
    for candidate in ordered:
        time_ms = candidate.actual_ms if candidate.actual_ms is not None else candidate.requested_ms
        if groups:
            prior = groups[-1][-1]
            prior_ms = prior.actual_ms if prior.actual_ms is not None else prior.requested_ms
            gap_ms = time_ms - prior_ms
            if gap_ms <= merge_tolerance_ms or (
                gap_ms <= adaptive_cluster_tolerance_ms
                and is_adaptive_only(prior)
                and is_adaptive_only(candidate)
            ):
                groups[-1].append(candidate)
                continue
        groups.append([candidate])

    merged: list[SurveyCandidate] = []
    for index, group in enumerate(groups, 1):
        # Preserve the strongest measured change as the representative when a
        # periodic/context request lands in the same merge window.  Structural
        # provenance remains visible through the unioned reasons below.
        primary = max(
            group,
            key=lambda item: (
                survey_candidate_is_protected(item),
                survey_candidate_importance(item),
                item.actual_ms is not None,
                item.score,
            ),
        )
        reasons = tuple(sorted({reason for item in group for reason in item.reasons}))
        merged.append(
            SurveyCandidate(
                candidate_id=f"VC{index:06d}",
                requested_ms=primary.requested_ms,
                actual_ms=primary.actual_ms,
                raw_pts=primary.raw_pts,
                time_base=primary.time_base,
                reasons=reasons,
                score=max(item.score for item in group),
                timestamp_source=primary.timestamp_source,
            )
        )
    return tuple(merged)


def survey_coverage(
    candidates: Iterable[SurveyCandidate],
    *,
    duration_ms: int | None = None,
    strict_gap_ms: int = 30_000,
) -> dict[str, object]:
    """Return an auditable temporal/importance receipt for survey candidates.

    The helper is intentionally side-effect free and works on either raw or
    merged candidates.  It reports measured candidate times, reason counts,
    protected high-change IDs, and whether the strict periodic ceiling is met;
    callers can persist this alongside their visual-survey cache.
    """

    if duration_ms is not None and duration_ms < 0:
        raise InputError("duration_ms cannot be negative")
    if strict_gap_ms <= 0:
        raise InputError("strict_gap_ms must be positive")
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.actual_ms if item.actual_ms is not None else item.requested_ms,
            item.candidate_id,
        ),
    )
    times = [
        int(item.actual_ms if item.actual_ms is not None else item.requested_ms)
        for item in ordered
    ]
    reason_counts: dict[str, int] = {}
    tier_counts = {tier: 0 for tier in ("very_high", "high", "supporting", "low")}
    protected_ids: list[str] = []
    for candidate in ordered:
        for reason in candidate.reasons:
            normalized = _normalized_survey_reason(reason)
            reason_counts[normalized] = reason_counts.get(normalized, 0) + 1
        tier_counts[survey_candidate_importance_tier(candidate)] += 1
        if survey_candidate_is_protected(candidate):
            protected_ids.append(candidate.candidate_id)
    gaps = [right - left for left, right in zip(times, times[1:], strict=False)]
    max_gap = max(gaps, default=0)
    if duration_ms is None:
        head_gap = 0
        tail_gap = 0
    elif times:
        head_gap = max(0, times[0])
        tail_gap = max(0, duration_ms - times[-1])
    else:
        head_gap = duration_ms
        tail_gap = duration_ms
    effective_max_gap = max(max_gap, head_gap, tail_gap)
    return {
        "candidate_count": len(ordered),
        "measured_time_count": len(times),
        "temporal_start_ms": times[0] if times else None,
        "temporal_end_ms": times[-1] if times else None,
        "max_gap_ms": effective_max_gap,
        "strict_gap_ms": strict_gap_ms,
        "strict_gap_satisfied": effective_max_gap <= strict_gap_ms,
        "reason_counts": dict(sorted(reason_counts.items())),
        "importance_counts": tier_counts,
        "protected_candidate_ids": tuple(protected_ids),
    }


def survey_video_candidates(
    media_path: str | Path,
    *,
    duration_ms: int,
    interval_seconds: float = 30.0,
    strict: bool = True,
    scene_detection: bool = True,
    scene_threshold: float = 0.3,
    adaptive_detection: bool = True,
    adaptive_threshold: float = 0.003,
    adaptive_sample_fps: float = 2.0,
    ffmpeg_threads: int | None = 4,
    ffmpeg_bin: str = "ffmpeg",
    timeout_seconds: float = 600.0,
    signal_samples: Iterable[SurveySignal] = (),
    chapter_times_ms: Iterable[int] = (),
    speech_reference_times_ms: Iterable[int] = (),
) -> tuple[SurveyCandidate, ...]:
    safety = periodic_candidates(duration_ms, interval_seconds=interval_seconds, strict=strict)
    detected: tuple[SurveyCandidate, ...] = ()
    decoded_adaptive: tuple[SurveyCandidate, ...] = ()
    if scene_detection and adaptive_detection:
        try:
            detected, decoded_adaptive = detect_combined_survey_candidates(
                media_path,
                scene_threshold=scene_threshold,
                adaptive_threshold=adaptive_threshold,
                adaptive_sample_fps=adaptive_sample_fps,
                ffmpeg_threads=ffmpeg_threads,
                ffmpeg_bin=ffmpeg_bin,
                timeout_seconds=timeout_seconds,
            )
        except ValidationFailure:
            # Preserve the exact, independently tested single-detector paths
            # if the combined graph cannot be constructed or completed on a
            # particular FFmpeg build. Never treat a failed/absent measurement
            # as a candidate timestamp.
            detected = detect_scene_candidates(
                media_path,
                threshold=scene_threshold,
                ffmpeg_threads=ffmpeg_threads,
                ffmpeg_bin=ffmpeg_bin,
                timeout_seconds=timeout_seconds,
            )
            decoded_adaptive = detect_adaptive_candidates(
                media_path,
                threshold=adaptive_threshold,
                sample_fps=adaptive_sample_fps,
                ffmpeg_threads=ffmpeg_threads,
                ffmpeg_bin=ffmpeg_bin,
                timeout_seconds=timeout_seconds,
            )
    elif scene_detection:
        detected = detect_scene_candidates(
            media_path,
            threshold=scene_threshold,
            ffmpeg_threads=ffmpeg_threads,
            ffmpeg_bin=ffmpeg_bin,
            timeout_seconds=timeout_seconds,
        )
    elif adaptive_detection:
        decoded_adaptive = detect_adaptive_candidates(
            media_path,
            threshold=adaptive_threshold,
            sample_fps=adaptive_sample_fps,
            ffmpeg_threads=ffmpeg_threads,
            ffmpeg_bin=ffmpeg_bin,
            timeout_seconds=timeout_seconds,
        )
    adaptive = adaptive_signal_candidates(signal_samples)
    context = contextual_candidates(
        chapter_times_ms=chapter_times_ms,
        speech_reference_times_ms=speech_reference_times_ms,
    )
    return merge_survey_candidates((safety, detected, decoded_adaptive, adaptive, context))
