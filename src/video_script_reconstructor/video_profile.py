from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, cast

from .errors import InputError
from .media_probe import MediaProbeResult


@dataclass(frozen=True)
class TimedVisualMetric:
    time_ms: int
    motion: float = 0.0
    frame_difference: float = 0.0
    edge_density: float = 0.0
    ocr_character_count: int = 0
    face_area_ratio: float | None = None


@dataclass(frozen=True)
class ProfileRange:
    start_ms: int
    end_ms: int
    characteristics: tuple[str, ...]
    confidence: float
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoSurveyProfile:
    duration_ms: int
    ranges: tuple[ProfileRange, ...]
    periodic_interval_seconds: float
    scene_density_per_minute: float


def adaptive_periodic_interval_seconds(
    duration_ms: int,
    *,
    configured_seconds: float = 30.0,
    strict: bool = True,
    scene_count: int = 0,
    mean_motion: float = 0.0,
) -> float:
    if duration_ms < 0:
        raise InputError("duration_ms cannot be negative")
    if configured_seconds <= 0:
        raise InputError("survey interval must be greater than zero")
    if strict and configured_seconds > 30:
        raise InputError("strict mode requires survey_interval_seconds <= 30")
    upper = min(configured_seconds, 30.0) if strict else configured_seconds
    minutes = max(duration_ms / 60_000, 1 / 60)
    density = scene_count / minutes
    pressure = min(1.0, max(0.0, density / 12.0) + max(0.0, mean_motion) * 0.5)
    # Dense or fast-changing media receives more safety samples; quiet media still honors the cap.
    return max(min(2.0, upper), upper * (1.0 - 0.6 * min(pressure, 0.9)))


def _classify(
    metrics: Sequence[TimedVisualMetric],
) -> tuple[tuple[str, ...], float, dict[str, float]]:
    if not metrics:
        return ("unknown",), 0.0, {}
    motion = fmean(item.motion for item in metrics)
    difference = fmean(item.frame_difference for item in metrics)
    edges = fmean(item.edge_density for item in metrics)
    ocr = fmean(item.ocr_character_count for item in metrics)
    face_values = [item.face_area_ratio for item in metrics if item.face_area_ratio is not None]
    face = fmean(face_values) if face_values else 0.0
    labels: list[str] = []
    evidence_strengths: list[float] = []
    if ocr >= 80 and edges >= 0.08:
        labels.append("text-heavy-screen")
        evidence_strengths.append(min(1.0, ocr / 250 + edges))
    elif ocr >= 25:
        labels.append("slides-or-interface")
        evidence_strengths.append(min(1.0, ocr / 100))
    if face >= 0.08 and motion < 0.35:
        labels.append("talking-head-or-interview")
        evidence_strengths.append(min(1.0, face * 4))
    if motion >= 0.45 or difference >= 0.3:
        labels.append("motion-heavy")
        evidence_strengths.append(min(1.0, max(motion, difference)))
    if motion < 0.08 and difference < 0.04:
        labels.append("static-visual")
        evidence_strengths.append(0.8)
    if not labels:
        labels.append("mixed-or-unclassified")
        evidence_strengths.append(0.25)
    aggregate = {
        "motion": motion,
        "frame_difference": difference,
        "edge_density": edges,
        "ocr_character_count": ocr,
        "face_area_ratio": face,
    }
    return tuple(labels), max(evidence_strengths), aggregate


def profile_video_ranges(
    duration_ms: int,
    metrics: Iterable[TimedVisualMetric | Mapping[str, object]],
    *,
    range_ms: int = 60_000,
) -> tuple[ProfileRange, ...]:
    if duration_ms < 0 or range_ms <= 0:
        raise InputError("duration_ms must be non-negative and range_ms must be positive")
    normalized: list[TimedVisualMetric] = []
    for item in metrics:
        normalized.append(
            item if isinstance(item, TimedVisualMetric) else TimedVisualMetric(**cast(Any, item))
        )
    normalized.sort(key=lambda item: item.time_ms)
    result: list[ProfileRange] = []
    for start in range(0, max(duration_ms, 1), range_ms):
        end = min(duration_ms, start + range_ms)
        bucket = [item for item in normalized if start <= item.time_ms < max(end, start + 1)]
        characteristics, confidence, aggregate = _classify(bucket)
        result.append(ProfileRange(start, end, characteristics, confidence, aggregate))
    return tuple(result)


def build_video_profile(
    probe: MediaProbeResult,
    metrics: Iterable[TimedVisualMetric | Mapping[str, object]] = (),
    *,
    scene_times_ms: Sequence[int] = (),
    configured_interval_seconds: float = 30.0,
    strict: bool = True,
    range_ms: int = 60_000,
) -> VideoSurveyProfile:
    duration = probe.duration_ms
    if duration is None:
        raise InputError("Cannot profile a video whose duration is unknown")
    materialized = [
        item if isinstance(item, TimedVisualMetric) else TimedVisualMetric(**cast(Any, item))
        for item in metrics
    ]
    mean_motion = fmean(item.motion for item in materialized) if materialized else 0.0
    interval = adaptive_periodic_interval_seconds(
        duration,
        configured_seconds=configured_interval_seconds,
        strict=strict,
        scene_count=len(scene_times_ms),
        mean_motion=mean_motion,
    )
    minutes = max(duration / 60_000, 1 / 60)
    return VideoSurveyProfile(
        duration_ms=duration,
        ranges=profile_video_ranges(duration, materialized, range_ms=range_ms),
        periodic_interval_seconds=interval,
        scene_density_per_minute=len(scene_times_ms) / minutes,
    )
