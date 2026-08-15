from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import BlockedError, InputError, ValidationFailure
from .security import safe_slug

if TYPE_CHECKING:
    from .schemas import FrameObservation

_TIME_BASE = re.compile(r"config in time_base:\s*(?P<base>\d+/\d+)")
_SHOWINFO = re.compile(
    r"\bn:\s*(?P<n>\d+)\s+pts:\s*(?P<pts>-?\d+)\s+pts_time:\s*(?P<time>-?[0-9.eE+]+)"
    r".*?\bs:(?P<width>\d+)x(?P<height>\d+)"
)


def _executor_context(
    worker_pool: ThreadPoolExecutor | None,
    *,
    max_workers: int,
    thread_name_prefix: str,
) -> AbstractContextManager[ThreadPoolExecutor]:
    """Reuse a caller-owned bounded pool when frame work is already scheduled.

    The public extractor remains self-contained when no pool is supplied.  The
    reconstruction pipeline passes its visual-stage pool so cache restores and
    exact seeks do not repeatedly create and join short-lived pools.
    """

    if worker_pool is not None:
        return nullcontext(worker_pool)
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)


@dataclass(frozen=True)
class DecodedFrameTiming:
    output_index: int
    raw_pts: int
    actual_ms: int
    time_base: str | None
    width: int
    height: int
    timestamp_source: str = "ffmpeg-showinfo"


@dataclass(frozen=True)
class ExtractedFrame:
    frame_id: str
    path: Path
    requested_ms: int
    actual_ms: int
    raw_pts: int
    time_base: str | None
    frame_index: int | None
    offset_ms: int
    timestamp_source: str
    width: int
    height: int


def parse_showinfo(stderr: str) -> tuple[DecodedFrameTiming, ...]:
    time_base_match = _TIME_BASE.search(stderr)
    time_base = time_base_match.group("base") if time_base_match else None
    results: list[DecodedFrameTiming] = []
    for match in _SHOWINFO.finditer(stderr):
        try:
            actual_ms = int((Decimal(match.group("time")) * 1000).to_integral_value())
        except InvalidOperation as exc:
            raise ValidationFailure("FFmpeg showinfo contained an invalid pts_time") from exc
        results.append(
            DecodedFrameTiming(
                output_index=int(match.group("n")),
                raw_pts=int(match.group("pts")),
                actual_ms=actual_ms,
                time_base=time_base,
                width=int(match.group("width")),
                height=int(match.group("height")),
            )
        )
    return tuple(results)


def build_frame_extraction_command(
    media_path: Path,
    requested_ms: int,
    output_path: Path,
    *,
    video_stream_index: int = 0,
    ffmpeg_threads: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    if requested_ms < 0:
        raise InputError("requested frame time cannot be negative")
    if video_stream_index < 0:
        raise InputError("video_stream_index cannot be negative")
    if ffmpeg_threads is not None and ffmpeg_threads <= 0:
        raise InputError("ffmpeg_threads must be positive when provided")
    seconds = Decimal(requested_ms) / Decimal(1000)
    filter_graph = f"select=gte(t\\,{seconds:f}),showinfo"
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        *(["-threads", str(ffmpeg_threads)] if ffmpeg_threads is not None else []),
        # Fast input seeking bounds work for long recordings. ``-copyts``
        # preserves the source clock so the measured showinfo PTS remains
        # comparable with the requested absolute media timestamp; the filter
        # still rejects any pre-target decoded frame after the keyframe seek.
        "-ss",
        f"{seconds:f}",
        "-copyts",
        "-avoid_negative_ts",
        "disabled",
        "-i",
        str(media_path),
        "-map",
        f"0:v:{video_stream_index}",
        "-vf",
        filter_graph,
        "-frames:v",
        "1",
        "-fps_mode",
        "vfr",
        "-an",
        "-sn",
        "-c:v",
        "png",
        "-update",
        "1",
        "-y",
        str(output_path),
    ]


def format_frame_timestamp(actual_ms: int) -> str:
    if actual_ms < 0:
        raise InputError("actual frame time cannot be negative")
    hours, remainder = divmod(actual_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s{milliseconds:03d}"


def _run_extract(
    media_path: Path,
    requested_ms: int,
    temporary_path: Path,
    *,
    video_stream_index: int,
    ffmpeg_threads: int | None,
    ffmpeg_bin: str,
    timeout_seconds: float,
) -> DecodedFrameTiming:
    command = build_frame_extraction_command(
        media_path,
        requested_ms,
        temporary_path,
        video_stream_index=video_stream_index,
        ffmpeg_threads=ffmpeg_threads,
        ffmpeg_bin=ffmpeg_bin,
    )
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
        raise BlockedError(f"Frame extraction exceeded the {timeout_seconds:g}s timeout") from exc
    if completed.returncode != 0:
        detail = re.sub(r"\s+", " ", completed.stderr).strip()[-1000:]
        raise ValidationFailure(
            f"FFmpeg frame extraction failed with exit code {completed.returncode}: {detail}"
        )
    timings = parse_showinfo(completed.stderr)
    if not timings:
        raise ValidationFailure(
            "FFmpeg emitted an image without measurable showinfo timing; "
            "actual time was not guessed"
        )
    # FFmpeg may allow one look-ahead frame through the filter graph even with
    # ``-frames:v 1``. The first showinfo record is the frame written by image2.
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        raise ValidationFailure("FFmpeg reported success but did not emit a non-empty PNG frame")
    return timings[0]


def extract_frame(
    media_path: str | Path,
    requested_ms: int,
    output_path: str | Path,
    *,
    frame_id: str = "F000001",
    video_stream_index: int = 0,
    ffmpeg_threads: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
    timeout_seconds: float = 120.0,
) -> ExtractedFrame:
    source = Path(media_path).expanduser()
    destination = Path(output_path).expanduser()
    if not source.is_file():
        raise InputError(f"Video source is not a file: {source}")
    if destination.suffix.casefold() != ".png":
        raise InputError("Evidence frames must be written as lossless PNG files")
    if source.resolve() == destination.resolve():
        raise InputError("Frame output cannot overwrite source media")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.png", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    try:
        timing = _run_extract(
            source,
            requested_ms,
            temporary,
            video_stream_index=video_stream_index,
            ffmpeg_threads=ffmpeg_threads,
            ffmpeg_bin=ffmpeg_bin,
            timeout_seconds=timeout_seconds,
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return ExtractedFrame(
        frame_id=frame_id,
        path=destination.resolve(),
        requested_ms=requested_ms,
        actual_ms=timing.actual_ms,
        raw_pts=timing.raw_pts,
        time_base=timing.time_base,
        frame_index=None,
        offset_ms=timing.actual_ms - requested_ms,
        timestamp_source=timing.timestamp_source,
        width=timing.width,
        height=timing.height,
    )


def extract_evidence_frame(
    media_path: str | Path,
    requested_ms: int,
    output_dir: str | Path,
    *,
    frame_id: str,
    video_stream_index: int = 0,
    ffmpeg_threads: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
    timeout_seconds: float = 120.0,
) -> ExtractedFrame:
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    safe_id = safe_slug(frame_id, fallback="frame")
    temporary_destination = directory / f".{safe_id}__pending.png"
    extracted = extract_frame(
        media_path,
        requested_ms,
        temporary_destination,
        frame_id=frame_id,
        video_stream_index=video_stream_index,
        ffmpeg_threads=ffmpeg_threads,
        ffmpeg_bin=ffmpeg_bin,
        timeout_seconds=timeout_seconds,
    )
    final_path = directory / f"{safe_id}__{format_frame_timestamp(extracted.actual_ms)}__full.png"
    if final_path.exists() and final_path != temporary_destination:
        from .frame_quality import normalized_pixel_hash

        if normalized_pixel_hash(final_path) != normalized_pixel_hash(temporary_destination):
            temporary_destination.unlink(missing_ok=True)
            raise ValidationFailure(
                f"Existing evidence frame has conflicting pixels: {final_path.name}"
            )
        temporary_destination.unlink(missing_ok=True)
        return ExtractedFrame(**{**extracted.__dict__, "path": final_path.resolve()})
    os.replace(temporary_destination, final_path)
    return ExtractedFrame(**{**extracted.__dict__, "path": final_path.resolve()})


_BATCH_WINDOW_SECONDS = Decimal("0.250")
# A guarded batch decodes every frame between its first and last request.  On
# long recordings, sparse requests are faster as bounded exact seeks even when
# there are more than two of them.  This density is deliberately conservative:
# it only changes the extraction route, never the requested timestamps or
# measured-frame validation.
# A guarded batch decodes the complete span once; independent exact seeks pay a
# process/seek startup for every request. The guarded filter intentionally keeps
# overlapping 250 ms windows out of one batch, so a valid non-overlapping group
# tops out near four requests per second. Real 1080p measurements show that
# even this upper-bound density is usually slower than exact seeks; keep the
# current route conservative until a provider/media-specific crossover is
# measured. This is acceleration-only: measured PTS and pixels remain the
# evidence authority.
# The route remains a performance choice only: every emitted frame is still
# checked by measured PTS, dimensions, and the normal pixel/hash invariants.
_EXACT_SEEK_DENSITY_THRESHOLD = 4.0
_LONG_SPAN_EXACT_SEEK_MIN_MS = 10 * 60 * 1000
_LONG_SPAN_EXACT_SEEK_DENSITY_THRESHOLD = 4.0


def _batch_select_filter(requested_times_ms: Iterable[int]) -> str:
    """Build a one-pass select filter that emits one frame per target.

    ``prev_selected_t`` is used as a per-window guard in seconds: once the first
    frame in a target's 250 ms look-ahead window is selected, later frames in
    that same window are ignored.  ``prev_selected_pts`` is expressed in the
    stream time base and must not be compared with the second-based request
    bounds. The caller partitions requests whose windows overlap, so a
    low-frame-rate source can safely fall back to individual extraction without
    changing evidence semantics.
    """

    clauses: list[str] = []
    for index, requested_ms in enumerate(requested_times_ms):
        start = Decimal(requested_ms) / Decimal(1000)
        end = start + _BATCH_WINDOW_SECONDS
        guard = f"lt(prev_selected_t\\,{start:f})"
        if index == 0:
            guard = f"(isnan(prev_selected_pts)+{guard})"
        clauses.append(
            f"(gte(t\\,{start:f})*lt(t\\,{end:f})*{guard})"
        )
    if not clauses:
        raise InputError("at least one requested frame time is required")
    return f"select='{'+'.join(clauses)}',showinfo"


def _batch_request_groups(requested_times_ms: list[int]) -> tuple[tuple[int, ...], ...]:
    groups: list[list[int]] = []
    for requested_ms in requested_times_ms:
        if not groups or requested_ms - groups[-1][-1] < int(_BATCH_WINDOW_SECONDS * 1000):
            if groups:
                # Overlapping look-ahead windows cannot be disambiguated by
                # ``prev_selected_pts``. Keep the close request outside the
                # batch and let the caller use the exact single-frame path.
                groups.append([requested_ms])
            else:
                groups.append([requested_ms])
        else:
            groups.append([requested_ms])
    # The loop above deliberately creates one-item groups for overlapping
    # requests. Merge only groups with a non-overlapping boundary; this keeps
    # the common periodic/survey points in one FFmpeg pass while preserving
    # exact behavior for dense scene bursts.
    merged: list[list[int]] = []
    for group in groups:
        if (
            merged
            and len(merged[-1]) > 0
            and group[0] - merged[-1][-1] >= int(_BATCH_WINDOW_SECONDS * 1000)
        ):
            merged[-1].extend(group)
        else:
            merged.append(group)
    return tuple(tuple(group) for group in merged)


def _prefer_exact_group(requested_times_ms: tuple[int, ...]) -> bool:
    """Choose exact seeks when a guarded batch would decode a sparse span."""

    # Two-request groups always stay on exact seeks. Preserve the historical
    # three-request crossover as well: dense three-frame fixture groups use a
    # batch, while genuinely sparse three-frame targets remain exact.
    if len(requested_times_ms) <= 2:
        return True
    span_ms = requested_times_ms[-1] - requested_times_ms[0]
    if span_ms <= 0:
        return True
    density = len(requested_times_ms) / (span_ms / 1000.0)
    if len(requested_times_ms) == 3 and density <= 0.15:
        return True
    if (
        span_ms >= _LONG_SPAN_EXACT_SEEK_MIN_MS
        and density <= _LONG_SPAN_EXACT_SEEK_DENSITY_THRESHOLD
    ):
        return True
    return density <= _EXACT_SEEK_DENSITY_THRESHOLD


def build_batch_frame_extraction_command(
    media_path: Path,
    requested_times_ms: tuple[int, ...],
    output_pattern: Path,
    *,
    ffmpeg_threads: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Build one bounded, lossless FFmpeg batch extraction command.

    The output stop time is absolute because ``-copyts`` keeps measured media
    timestamps intact.  Bounding it to the final request's look-ahead window
    prevents a sparse early group from decoding the rest of a long recording.
    """

    if not requested_times_ms:
        raise InputError("at least one requested frame time is required")
    if any(requested_ms < 0 for requested_ms in requested_times_ms):
        raise InputError("requested frame times cannot be negative")
    if ffmpeg_threads is not None and ffmpeg_threads <= 0:
        raise InputError("ffmpeg_threads must be positive when provided")
    first_seconds = Decimal(requested_times_ms[0]) / Decimal(1000)
    stop_seconds = Decimal(requested_times_ms[-1]) / Decimal(1000) + _BATCH_WINDOW_SECONDS
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        *(["-threads", str(ffmpeg_threads)] if ffmpeg_threads is not None else []),
        "-copyts",
        "-avoid_negative_ts",
        "disabled",
    ]
    # Keep stream timestamps absolute while avoiding a redundant decode of the
    # prefix for groups that begin late in a long recording. Exact measured PTS
    # remains authoritative; the caller falls back to exact extraction when a
    # seek/window cannot emit every request.
    if first_seconds > 0:
        command.extend(["-ss", f"{first_seconds:f}"])
    command.extend(
        [
            "-i",
            str(media_path),
            "-to",
            f"{stop_seconds:f}",
            "-map",
            "0:v:0",
            "-vf",
            _batch_select_filter(requested_times_ms),
            "-fps_mode",
            "vfr",
            "-an",
            "-sn",
            "-c:v",
            "png",
            "-start_number",
            "1",
            "-y",
            str(output_pattern),
        ]
    )
    return command


def _run_batch_group(
    media_path: Path,
    requested_times_ms: tuple[int, ...],
    output_dir: Path,
    *,
    first_frame_number: int,
    ffmpeg_threads: int | None,
    ffmpeg_bin: str,
    timeout_seconds: float,
) -> tuple[ExtractedFrame, ...]:
    """Decode one non-overlapping target group in a single FFmpeg process."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vsr-batch-", dir=output_dir) as temporary_name:
        temporary_dir = Path(temporary_name)
        pattern = temporary_dir / "frame-%06d.png"
        command = build_batch_frame_extraction_command(
            media_path,
            requested_times_ms,
            pattern,
            ffmpeg_threads=ffmpeg_threads,
            ffmpeg_bin=ffmpeg_bin,
        )
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
            raise BlockedError(f"Batch frame extraction exceeded {timeout_seconds:g}s") from exc
        if completed.returncode != 0:
            detail = re.sub(r"\s+", " ", completed.stderr).strip()[-1000:]
            raise ValidationFailure(
                f"FFmpeg batch frame extraction failed with exit code {completed.returncode}: {detail}"
            )
        timings = parse_showinfo(completed.stderr)
        files = sorted(temporary_dir.glob("frame-*.png"))
        if len(timings) != len(requested_times_ms) or len(files) != len(requested_times_ms):
            raise ValidationFailure(
                "FFmpeg batch frame extraction did not emit exactly one measured frame per request"
            )
        extracted: list[ExtractedFrame] = []
        for offset, (requested_ms, timing, temporary_path) in enumerate(
            zip(requested_times_ms, timings, files, strict=True)
        ):
            if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                raise ValidationFailure("FFmpeg batch extraction emitted an empty PNG frame")
            frame_id = f"F{first_frame_number + offset:06d}"
            safe_id = safe_slug(frame_id, fallback="frame")
            final_path = output_dir / (
                f"{safe_id}__{format_frame_timestamp(timing.actual_ms)}__full.png"
            )
            if final_path.exists():
                from .frame_quality import normalized_pixel_hash

                if normalized_pixel_hash(final_path) != normalized_pixel_hash(temporary_path):
                    raise ValidationFailure(
                        f"Existing evidence frame has conflicting pixels: {final_path.name}"
                    )
                temporary_path.unlink(missing_ok=True)
            else:
                os.replace(temporary_path, final_path)
            extracted.append(
                ExtractedFrame(
                    frame_id=frame_id,
                    path=final_path.resolve(),
                    requested_ms=requested_ms,
                    actual_ms=timing.actual_ms,
                    raw_pts=timing.raw_pts,
                    time_base=timing.time_base,
                    frame_index=None,
                    offset_ms=timing.actual_ms - requested_ms,
                    timestamp_source=timing.timestamp_source,
                    width=timing.width,
                    height=timing.height,
                )
            )
        return tuple(extracted)


def _extract_frames_batched(
    media_path: Path,
    requested_times_ms: list[int],
    output_dir: Path,
    *,
    first_frame_number: int,
    ffmpeg_threads: int | None,
    ffmpeg_bin: str,
    timeout_seconds: float,
    max_workers: int,
    worker_pool: ThreadPoolExecutor | None = None,
) -> tuple[ExtractedFrame, ...]:
    groups = _batch_request_groups(requested_times_ms)
    group_offsets: list[tuple[int, tuple[int, ...]]] = []
    offset = 0
    for group in groups:
        group_offsets.append((offset, group))
        offset += len(group)

    # Very small groups are faster and less resource-intensive as independent
    # exact seeks.  A guarded batch still decodes every intervening frame; on
    # long high-resolution recordings two sparse requests can otherwise spend
    # seconds decoding a large interval while two keyframe seeks finish almost
    # immediately.  Larger survey groups retain the one-pass batch path.
    exact_group_offsets = {
        group_offset
        for group_offset, group in group_offsets
        if _prefer_exact_group(group)
    }
    exact_offsets = [
        (group_offset + position, requested_ms)
        for group_offset, group in group_offsets
        if group_offset in exact_group_offsets
        for position, requested_ms in enumerate(group)
    ]
    singleton_results: dict[int, ExtractedFrame] = {}

    def extract_single(item: tuple[int, int]) -> tuple[int, ExtractedFrame]:
        group_offset, requested_ms = item
        return group_offset, extract_evidence_frame(
            media_path,
            requested_ms,
            output_dir,
            frame_id=f"F{first_frame_number + group_offset:06d}",
            ffmpeg_threads=ffmpeg_threads,
            ffmpeg_bin=ffmpeg_bin,
            timeout_seconds=timeout_seconds,
        )

    if exact_offsets:
        with _executor_context(
            worker_pool,
            max_workers=max_workers,
            thread_name_prefix="vsr-frame-single",
        ) as pool:
            for group_offset, extracted in pool.map(extract_single, exact_offsets):
                singleton_results[group_offset] = extracted

    extracted_by_offset: dict[int, ExtractedFrame] = dict(singleton_results)
    # Batch groups have disjoint temporary/output names and independent FFmpeg
    # seek windows, so run them on the same bounded pool as exact seeks.  The
    # pool's per-process FFmpeg thread budget is already derived from
    # ``max_workers``; parallelizing here removes serial waits across dense
    # bursts without changing any measured PTS or emitted PNG bytes. Collect
    # futures in group order so result ordering remains deterministic.
    batch_groups = tuple(
        (group_offset, group)
        for group_offset, group in group_offsets
        if group_offset not in exact_group_offsets
    )
    batch_fallback_offsets: list[tuple[int, int]] = []
    if batch_groups:
        def run_batch_group(
            item: tuple[int, tuple[int, ...]],
        ) -> tuple[int, tuple[ExtractedFrame, ...]]:
            group_offset, group = item
            return group_offset, _run_batch_group(
                media_path,
                group,
                output_dir,
                first_frame_number=first_frame_number + group_offset,
                ffmpeg_threads=ffmpeg_threads,
                ffmpeg_bin=ffmpeg_bin,
                timeout_seconds=timeout_seconds,
            )

        with _executor_context(
            worker_pool,
            max_workers=max_workers,
            thread_name_prefix="vsr-frame-batch",
        ) as pool:
            batch_futures = [pool.submit(run_batch_group, item) for item in batch_groups]
            for (group_offset, group), future in zip(batch_groups, batch_futures, strict=True):
                try:
                    _completed_offset, batch = future.result()
                except ValidationFailure:
                    # A group can have moved one or more final PNGs before a
                    # later measured frame failed. Remove only this group's
                    # frame-ID prefixes so successful neighboring groups stay
                    # intact before exact fallback retries the failed group.
                    for position in range(len(group)):
                        prefix = f"F{first_frame_number + group_offset + position:06d}"
                        for partial_path in output_dir.glob(f"{prefix}*.png"):
                            try:
                                partial_path.unlink()
                            except FileNotFoundError:
                                pass
                    batch_fallback_offsets.extend(
                        (group_offset + position, requested_ms)
                        for position, requested_ms in enumerate(group)
                    )
                    continue
                extracted_by_offset.update(
                    {group_offset + position: frame for position, frame in enumerate(batch)}
                )
    if batch_fallback_offsets:
        with _executor_context(
            worker_pool,
            max_workers=max_workers,
            thread_name_prefix="vsr-frame-batch-fallback",
        ) as pool:
            for offset, extracted in pool.map(extract_single, batch_fallback_offsets):
                extracted_by_offset[offset] = extracted
    return tuple(extracted_by_offset[position] for position in range(len(requested_times_ms)))


def extract_frames(
    media_path: str | Path,
    requested_times_ms: list[int] | tuple[int, ...],
    output_dir: str | Path,
    *,
    first_frame_number: int = 1,
    ffmpeg_bin: str = "ffmpeg",
    max_workers: int = 1,
    batch: bool = False,
    ffmpeg_threads: int | None = None,
    timeout_seconds: float = 120.0,
    worker_pool: ThreadPoolExecutor | None = None,
) -> tuple[ExtractedFrame, ...]:
    if len(set(requested_times_ms)) != len(requested_times_ms):
        raise InputError("requested frame times must be unique")
    if max_workers <= 0:
        raise InputError("max_workers must be positive")
    if ffmpeg_threads is not None and ffmpeg_threads <= 0:
        raise InputError("ffmpeg_threads must be positive when provided")
    effective_ffmpeg_threads = ffmpeg_threads
    if effective_ffmpeg_threads is None:
        # A bounded process pool already multiplies decoder threads. Limiting
        # each FFmpeg process keeps total runnable work near the machine's
        # logical CPU count instead of letting four exact seeks each spawn a
        # full decoder pool. This is a performance setting only; measured PTS
        # and pixel bytes remain the evidence authority.
        effective_ffmpeg_threads = max(
            1, min(4, (os.cpu_count() or 1) // max_workers)
        )
    ordered_times = sorted(requested_times_ms)

    source = Path(media_path).expanduser()
    directory = Path(output_dir).expanduser()
    if not source.is_file():
        raise InputError(f"Video source is not a file: {source}")
    directory.mkdir(parents=True, exist_ok=True)
    if batch and len(ordered_times) >= 2:
        try:
            return _extract_frames_batched(
                source,
                ordered_times,
                directory,
                first_frame_number=first_frame_number,
                ffmpeg_threads=effective_ffmpeg_threads,
                ffmpeg_bin=ffmpeg_bin,
                timeout_seconds=max(timeout_seconds, 600.0),
                max_workers=max_workers,
                worker_pool=worker_pool,
            )
        except ValidationFailure:
            # A low-frame-rate/VFR stream can still reject an exact fallback
            # window. Preserve the exact legacy extraction semantics rather
            # than guessing an actual timestamp; ordinary batch failures are
            # already isolated to their group above.  A failed batch may have
            # moved some outputs before a later group failed, though; clear
            # that run-local directory before retrying so the fallback cannot
            # observe a partial schedule or leave orphan PNGs behind.
            for partial_path in directory.rglob("*.png"):
                try:
                    partial_path.unlink()
                except FileNotFoundError:
                    pass
            pass

    def extract_one(item: tuple[int, int]) -> ExtractedFrame:
        offset, requested_ms = item
        return extract_evidence_frame(
            media_path,
            requested_ms,
            output_dir,
            frame_id=f"F{first_frame_number + offset:06d}",
            ffmpeg_threads=effective_ffmpeg_threads,
            ffmpeg_bin=ffmpeg_bin,
            timeout_seconds=timeout_seconds,
        )

    # Frame requests are independent and each extraction records its own
    # showinfo PTS. A bounded pool reduces process-start/seek overhead on long
    # recordings while preserving deterministic result order and failure
    # propagation. Keep the default sequential for callers that need the
    # smallest possible resource footprint.
    indexed = list(enumerate(ordered_times))
    if max_workers == 1 or len(indexed) < 2:
        return tuple(extract_one(item) for item in indexed)
    with _executor_context(
        worker_pool,
        max_workers=max_workers,
        thread_name_prefix="vsr-frame",
    ) as pool:
        return tuple(pool.map(extract_one, indexed))


def to_frame_observation(
    frame: ExtractedFrame,
    *,
    project_root: str | Path,
    selection_reason: str,
    evidence_role: str = "context",
    scene_id: str | None = None,
) -> FrameObservation:
    from .frame_quality import (
        PERCEPTUAL_DHASH_ALGORITHM,
        PERCEPTUAL_DHASH_VERIFIED,
        assess_frame_quality,
        normalized_pixel_hash,
        perceptual_dhash,
    )
    from .schemas import FrameObservation, PixelHash

    root = Path(project_root).resolve(strict=True)
    relative = frame.path.resolve(strict=True).relative_to(root).as_posix()
    quality = assess_frame_quality(frame.path)
    return FrameObservation(
        frame_id=frame.frame_id,
        requested_ms=frame.requested_ms,
        actual_ms=frame.actual_ms,
        pts=frame.raw_pts,
        time_base=frame.time_base,
        frame_index=frame.frame_index,
        offset_ms=frame.offset_ms,
        timestamp_source=frame.timestamp_source,
        timing_estimated=False,
        full_frame_path=relative,
        scene_id=scene_id,
        quality_scores={
            "sharpness": quality.sharpness,
            "brightness": quality.brightness,
            "contrast": quality.contrast,
            "edge_density": quality.edge_density,
            "transition_risk": quality.transition_risk,
            "overall": quality.overall,
        },
        perceptual_hashes={
            "dhash-8": perceptual_dhash(frame.path),
            "dhash-8-algorithm": PERCEPTUAL_DHASH_ALGORITHM,
            "dhash-8-verified": PERCEPTUAL_DHASH_VERIFIED,
        },
        pixel_hash=PixelHash(value=normalized_pixel_hash(frame.path)),
        selection_reason=selection_reason,
        evidence_role=evidence_role,
    )
