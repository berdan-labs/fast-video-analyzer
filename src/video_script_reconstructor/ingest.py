from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .errors import BlockedError, InputError, SecurityError, ValidationFailure
from .media_probe import MediaProbeResult, probe_media
from .media_probe import MediaStream as ProbeMediaStream
from .security import ensure_contained, safe_slug, sha256_file

if TYPE_CHECKING:
    from .schemas import MediaIdentity


class SourceKind(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"
    TIMESTAMPED_TRANSCRIPT = "timestamped_transcript"
    PLAIN_TRANSCRIPT = "plain_transcript"


VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".webm"})
AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".m4a"})
SUBTITLE_EXTENSIONS = frozenset({".srt", ".vtt", ".ass", ".ssa"})
JSON_TRANSCRIPT_EXTENSIONS = frozenset({".json"})
PLAIN_TRANSCRIPT_EXTENSIONS = frozenset({".txt", ".md"})


@dataclass(frozen=True)
class IngestedSource:
    media_id: str
    source_path: Path
    preserved_path: Path
    kind: SourceKind
    content_sha256: str
    size_bytes: int
    probe: MediaProbeResult | None
    acquisition: str = "local_user_file"


def classify_local_source(path: str | Path) -> SourceKind:
    suffix = Path(path).suffix.casefold()
    if suffix in VIDEO_EXTENSIONS:
        return SourceKind.VIDEO
    if suffix in AUDIO_EXTENSIONS:
        return SourceKind.AUDIO
    if suffix in SUBTITLE_EXTENSIONS:
        return SourceKind.SUBTITLE
    if suffix in JSON_TRANSCRIPT_EXTENSIONS:
        return SourceKind.TIMESTAMPED_TRANSCRIPT
    if suffix in PLAIN_TRANSCRIPT_EXTENSIONS:
        return SourceKind.PLAIN_TRANSCRIPT
    supported = sorted(
        VIDEO_EXTENSIONS
        | AUDIO_EXTENSIONS
        | SUBTITLE_EXTENSIONS
        | JSON_TRANSCRIPT_EXTENSIONS
        | PLAIN_TRANSCRIPT_EXTENSIONS
    )
    raise InputError(
        f"Unsupported input extension {suffix or '<none>'}; supported: {', '.join(supported)}"
    )


def _copy_verified(source: Path, destination: Path, expected_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with source.open("rb") as input_stream, os.fdopen(handle, "wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        temporary = Path(temporary_name)
        if sha256_file(temporary) != expected_hash:
            raise ValidationFailure("Preserved input copy failed content-hash verification")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def ingest_local_source(
    source: str | Path,
    *,
    preserved_root: str | Path | None = None,
    copy_source: bool = False,
    ffprobe_bin: str = "ffprobe",
    probe_timeout_seconds: float = 30.0,
) -> IngestedSource:
    raw = str(source)
    if urlparse(raw).scheme in {"http", "https"}:
        raise SecurityError(
            "Remote input requires a separately configured, explicitly permitted adapter"
        )
    path = Path(source).expanduser()
    if path.is_symlink():
        raise SecurityError(
            f"Symbolic-link input is not accepted without explicit resolution: {path}"
        )
    if not path.is_file():
        raise InputError(f"Input does not exist or is not a regular file: {path}")
    path = path.resolve()
    kind = classify_local_source(path)
    digest = sha256_file(path)
    media_id = f"M{digest[:16].upper()}"
    preserved_path = path
    if copy_source:
        if preserved_root is None:
            raise InputError("preserved_root is required when copy_source is enabled")
        root = Path(preserved_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve(strict=True)
        filename = f"{media_id}__{safe_slug(path.stem)}{path.suffix.casefold()}"
        preserved_path = ensure_contained(root, root / filename)
        if preserved_path.exists():
            if not preserved_path.is_file() or sha256_file(preserved_path) != digest:
                raise ValidationFailure(f"Existing preserved source conflicts with {media_id}")
        else:
            _copy_verified(path, preserved_path, digest)

    probe = None
    if kind in {SourceKind.VIDEO, SourceKind.AUDIO}:
        probe = probe_media(
            preserved_path,
            ffprobe_bin=ffprobe_bin,
            timeout_seconds=probe_timeout_seconds,
        )
        if kind is SourceKind.VIDEO and not probe.video_streams:
            raise ValidationFailure(
                "A video extension was supplied but FFprobe found no video stream"
            )
        if kind is SourceKind.AUDIO and not probe.audio_streams:
            raise ValidationFailure(
                "An audio extension was supplied but FFprobe found no audio stream"
            )

    return IngestedSource(
        media_id=media_id,
        source_path=path,
        preserved_path=preserved_path,
        kind=kind,
        content_sha256=digest,
        size_bytes=path.stat().st_size,
        probe=probe,
    )


def ingest_source(source: str | Path, **kwargs: Any) -> IngestedSource:
    """Ingest supported local evidence; remote acquisition is deliberately not implicit."""
    raw = str(source)
    parsed = urlparse(raw)
    # A Windows drive path (C:\...) is not a URL scheme.
    if "://" in raw and parsed.scheme:
        if parsed.scheme not in {"http", "https"}:
            raise InputError(f"Unsupported source URL scheme: {parsed.scheme}")
        raise BlockedError(
            "Direct HTTP(S) acquisition is unavailable without an explicit remote adapter"
        )
    return ingest_local_source(source, **kwargs)


def to_media_identity(ingested: IngestedSource) -> MediaIdentity:
    """Convert an ingested source into the canonical strict schema model."""
    from .schemas import MediaChapter, MediaIdentity, MediaStream

    probe = ingested.probe
    video = probe.video_streams if probe else ()
    primary_video = video[0] if video else None

    def stream_model(stream: ProbeMediaStream) -> MediaStream:
        # Kept explicit so FFprobe fields cannot silently spill into the persisted schema.
        item = stream
        return MediaStream(
            index=item.index,
            codec=item.codec_name,
            language=item.language,
            disposition={key: bool(value) for key, value in item.disposition.items()},
            metadata=dict(item.tags),
        )

    return MediaIdentity(
        media_id=ingested.media_id,
        original_source_reference=str(ingested.source_path),
        local_preserved_reference=str(ingested.preserved_path),
        content_hash=ingested.content_sha256,
        byte_size=ingested.size_bytes,
        duration_ms=probe.duration_ms if probe else None,
        container=probe.container if probe else None,
        video_streams=[stream_model(item) for item in probe.video_streams] if probe else [],
        audio_streams=[stream_model(item) for item in probe.audio_streams] if probe else [],
        subtitle_streams=[stream_model(item) for item in probe.subtitle_streams] if probe else [],
        frame_rate=primary_video.r_frame_rate if primary_video else None,
        average_frame_rate=primary_video.avg_frame_rate if primary_video else None,
        time_base=primary_video.time_base if primary_video else None,
        variable_frame_rate=probe.variable_frame_rate if probe else None,
        resolution=(primary_video.width, primary_video.height)
        if primary_video and primary_video.width and primary_video.height
        else None,
        sample_aspect_ratio=primary_video.sample_aspect_ratio if primary_video else None,
        rotation=primary_video.rotation if primary_video else None,
        chapters=[
            MediaChapter(
                chapter_id=f"source-{chapter.chapter_id}",
                start_ms=chapter.start_ms,
                end_ms=chapter.end_ms,
                title=chapter.title,
            )
            for chapter in probe.chapters
        ]
        if probe
        else [],
        source_metadata=dict(probe.source_metadata) if probe else {},
        acquisition_provenance={
            "method": ingested.acquisition,
            "copied": ingested.source_path != ingested.preserved_path,
        },
    )
