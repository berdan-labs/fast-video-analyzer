from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from .errors import BlockedError, InputError, ValidationFailure


@dataclass(frozen=True)
class MediaStream:
    index: int
    codec_type: str
    codec_name: str | None
    duration_ms: int | None
    start_ms: int | None
    time_base: str | None
    width: int | None = None
    height: int | None = None
    sample_aspect_ratio: str | None = None
    r_frame_rate: str | None = None
    avg_frame_rate: str | None = None
    channels: int | None = None
    sample_rate: int | None = None
    language: str | None = None
    rotation: int = 0
    tags: Mapping[str, str] = field(default_factory=dict)
    disposition: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MediaChapter:
    chapter_id: int
    start_ms: int | None
    end_ms: int | None
    time_base: str | None
    title: str | None


@dataclass(frozen=True)
class MediaProbeResult:
    source_path: Path
    duration_ms: int | None
    size_bytes: int
    container: str | None
    bit_rate: int | None
    streams: tuple[MediaStream, ...]
    chapters: tuple[MediaChapter, ...]
    source_metadata: Mapping[str, str]
    ffprobe_version: str | None = None

    @property
    def video_streams(self) -> tuple[MediaStream, ...]:
        return tuple(stream for stream in self.streams if stream.codec_type == "video")

    @property
    def audio_streams(self) -> tuple[MediaStream, ...]:
        return tuple(stream for stream in self.streams if stream.codec_type == "audio")

    @property
    def subtitle_streams(self) -> tuple[MediaStream, ...]:
        return tuple(stream for stream in self.streams if stream.codec_type == "subtitle")

    @property
    def variable_frame_rate(self) -> bool | None:
        video = self.video_streams
        if not video:
            return None
        measured: list[bool] = []
        for stream in video:
            nominal = _fraction(stream.r_frame_rate)
            average = _fraction(stream.avg_frame_rate)
            if nominal is None or average is None:
                continue
            measured.append(abs(float(nominal - average)) > 0.001)
        return any(measured) if measured else None


def _fraction(value: str | None) -> Fraction | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return result if result > 0 else None


def _milliseconds(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number * 1000)


def _integer(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rotation(stream: Mapping[str, Any]) -> int:
    tags = stream.get("tags") or {}
    tagged = tags.get("rotate") if isinstance(tags, Mapping) else None
    if tagged is not None:
        try:
            converted_rotation: int | None = int(round(float(tagged)))
        except (TypeError, ValueError):
            converted_rotation = None
        if converted_rotation is not None:
            return converted_rotation
    for side_data in stream.get("side_data_list") or []:
        if isinstance(side_data, Mapping) and side_data.get("rotation") is not None:
            try:
                return int(round(float(side_data["rotation"])))
            except (TypeError, ValueError):
                continue
    return 0


def build_ffprobe_command(path: Path, *, ffprobe_bin: str = "ffprobe") -> list[str]:
    """Build the exact safe argument vector used for the media probe."""
    return [
        ffprobe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        "-show_entries",
        "stream=index,codec_type,codec_name,duration,start_time,time_base,width,height,"
        "sample_aspect_ratio,r_frame_rate,avg_frame_rate,channels,sample_rate:"
        "stream_tags:stream_disposition:stream_side_data:"
        "format=format_name,duration,size,bit_rate:format_tags:"
        "chapter=id,start_time,end_time,time_base:chapter_tags",
        "-of",
        "json",
        str(path),
    ]


def parse_ffprobe_json(
    payload: str | bytes | Mapping[str, Any], *, source_path: Path
) -> MediaProbeResult:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="strict")
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValidationFailure(f"FFprobe returned invalid JSON: {exc}") from exc
    else:
        data = dict(payload)
    if not isinstance(data, dict):
        raise ValidationFailure("FFprobe root must be a JSON object")

    streams: list[MediaStream] = []
    for raw in data.get("streams") or []:
        if not isinstance(raw, Mapping):
            raise ValidationFailure("FFprobe stream entry must be an object")
        tags = {str(key): str(value) for key, value in (raw.get("tags") or {}).items()}
        disposition = {
            str(key): int(value)
            for key, value in (raw.get("disposition") or {}).items()
            if isinstance(value, (int, bool)) or str(value).lstrip("-").isdigit()
        }
        index = _integer(raw.get("index"))
        if index is None:
            raise ValidationFailure("FFprobe stream has no integer index")
        streams.append(
            MediaStream(
                index=index,
                codec_type=str(raw.get("codec_type") or "unknown"),
                codec_name=str(raw["codec_name"]) if raw.get("codec_name") else None,
                duration_ms=_milliseconds(raw.get("duration")),
                start_ms=_milliseconds(raw.get("start_time")),
                time_base=str(raw["time_base"]) if raw.get("time_base") else None,
                width=_integer(raw.get("width")),
                height=_integer(raw.get("height")),
                sample_aspect_ratio=(
                    str(raw["sample_aspect_ratio"]) if raw.get("sample_aspect_ratio") else None
                ),
                r_frame_rate=str(raw["r_frame_rate"]) if raw.get("r_frame_rate") else None,
                avg_frame_rate=(str(raw["avg_frame_rate"]) if raw.get("avg_frame_rate") else None),
                channels=_integer(raw.get("channels")),
                sample_rate=_integer(raw.get("sample_rate")),
                language=tags.get("language"),
                rotation=_rotation(raw),
                tags=tags,
                disposition=disposition,
            )
        )

    chapters: list[MediaChapter] = []
    for position, raw in enumerate(data.get("chapters") or []):
        if not isinstance(raw, Mapping):
            raise ValidationFailure("FFprobe chapter entry must be an object")
        tags = raw.get("tags") or {}
        chapters.append(
            MediaChapter(
                chapter_id=_integer(raw.get("id")) or position,
                start_ms=_milliseconds(raw.get("start_time")),
                end_ms=_milliseconds(raw.get("end_time")),
                time_base=str(raw["time_base"]) if raw.get("time_base") else None,
                title=(
                    str(tags["title"]) if isinstance(tags, Mapping) and tags.get("title") else None
                ),
            )
        )

    raw_format = data.get("format") or {}
    if not isinstance(raw_format, Mapping):
        raise ValidationFailure("FFprobe format entry must be an object")
    metadata = {str(key): str(value) for key, value in (raw_format.get("tags") or {}).items()}
    duration = _milliseconds(raw_format.get("duration"))
    if duration is None:
        stream_durations = [
            stream.duration_ms for stream in streams if stream.duration_ms is not None
        ]
        duration = max(stream_durations, default=None)
    reported_size = _integer(raw_format.get("size"))
    size_bytes = reported_size if reported_size is not None else source_path.stat().st_size
    return MediaProbeResult(
        source_path=source_path,
        duration_ms=duration,
        size_bytes=size_bytes,
        container=str(raw_format["format_name"]) if raw_format.get("format_name") else None,
        bit_rate=_integer(raw_format.get("bit_rate")),
        streams=tuple(streams),
        chapters=tuple(chapters),
        source_metadata=metadata,
    )


def probe_media(
    path: str | Path,
    *,
    ffprobe_bin: str = "ffprobe",
    timeout_seconds: float = 30.0,
) -> MediaProbeResult:
    source = Path(path).expanduser()
    if not source.is_file():
        raise InputError(f"Media source is not a readable file: {source}")
    command = build_ffprobe_command(source, ffprobe_bin=ffprobe_bin)
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
        raise BlockedError(f"FFprobe executable was not found: {ffprobe_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise BlockedError(f"FFprobe exceeded the {timeout_seconds:g}s timeout") from exc
    if completed.returncode != 0:
        detail = re.sub(r"\s+", " ", completed.stderr).strip()[-1000:]
        raise ValidationFailure(f"FFprobe failed with exit code {completed.returncode}: {detail}")
    return parse_ffprobe_json(completed.stdout, source_path=source.resolve())


def ffprobe_version(*, ffprobe_bin: str = "ffprobe", timeout_seconds: float = 5.0) -> str:
    command: Sequence[str] = [ffprobe_bin, "-version"]
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise BlockedError(f"Unable to execute FFprobe: {ffprobe_bin}") from exc
    if completed.returncode != 0:
        raise ValidationFailure(
            f"FFprobe version check failed with exit code {completed.returncode}"
        )
    return completed.stdout.splitlines()[0].strip() if completed.stdout else "unknown"
