from __future__ import annotations

import hashlib
import io
import math
import re
import struct
import unicodedata
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageCms, ImageFilter, ImageOps, ImageStat

from .errors import InputError

# Persisted alongside each dHash so semantic shortcuts can distinguish hashes
# produced by the current, frame-local pipeline from legacy/corrupted metadata.
# Marker values stay strings because ``FrameObservation.perceptual_hashes`` is
# intentionally typed as ``dict[str, str]`` for stable JSON/schema round-trips.
PERCEPTUAL_DHASH_ALGORITHM = "dhash-8-v1"
PERCEPTUAL_DHASH_VERIFIED = "true"


@dataclass(frozen=True)
class FrameQuality:
    sharpness: float
    brightness: float
    contrast: float
    edge_density: float
    clipped_dark_ratio: float
    clipped_light_ratio: float
    transition_risk: float
    overall: float


@dataclass(frozen=True)
class DifferenceRegion:
    xywh: tuple[int, int, int, int]
    changed_ratio: float
    mean_difference: float


@dataclass(frozen=True)
class FrameDifference:
    perceptual_hamming: int
    changed_pixel_ratio: float
    mean_pixel_difference: float
    edge_difference: float
    maximum_region_change: float
    regions: tuple[DifferenceRegion, ...]


@dataclass(frozen=True)
class DeduplicationDecision:
    is_duplicate: bool
    difference: FrameDifference
    ocr_changed: bool
    protected_reasons: tuple[str, ...]


def _open_rgb(path_or_image: str | Path | Image.Image) -> Image.Image:
    if isinstance(path_or_image, Image.Image):
        # Analysis is read-only.  Reuse an already-normalized in-memory image
        # instead of allocating another full pixel buffer for every quality /
        # difference call.  Non-RGB inputs still receive the historical
        # conversion behavior.
        return path_or_image if path_or_image.mode == "RGB" else path_or_image.convert("RGB")
    with Image.open(path_or_image) as image:
        image.load()
        return image.convert("RGB")


def _variance(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _pixels(image: Image.Image) -> list[int]:
    flattened = getattr(image, "get_flattened_data", None)
    return list(flattened()) if flattened is not None else list(image.getdata())


def _histogram_sum(histogram: Sequence[int]) -> int:
    """Return the exact integer sum represented by an 8-bit histogram."""

    return sum(value * count for value, count in enumerate(histogram))


def _assess_frame_quality_comparison(image: Image.Image) -> FrameQuality:
    """Assess an already bounded RGB comparison image without another resize."""

    gray = image.convert("L")
    histogram = gray.histogram()
    total = max(1, image.width * image.height)
    stats = ImageStat.Stat(gray)
    brightness = stats.mean[0] / 255.0
    contrast = stats.stddev[0] / 127.5
    clipped_dark = sum(histogram[:5]) / total
    clipped_light = sum(histogram[251:]) / total

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_image = edges.resize((min(256, image.width), min(256, image.height)))
    edge_values = _pixels(edge_image)
    sharpness = min(1.0, _variance(edge_values) / 2500.0)
    edge_histogram = edge_image.histogram()
    edge_density = sum(edge_histogram[24:]) / max(1, sum(edge_histogram))
    exposure_penalty = min(1.0, clipped_dark + clipped_light)
    low_contrast_penalty = max(0.0, 0.18 - contrast) / 0.18
    transition_risk = min(1.0, 0.65 * low_contrast_penalty + 0.35 * exposure_penalty)
    overall = max(
        0.0,
        min(
            1.0,
            0.50 * sharpness
            + 0.20 * min(1.0, contrast)
            + 0.15 * min(1.0, edge_density * 3)
            + 0.15 * (1.0 - transition_risk),
        ),
    )
    return FrameQuality(
        sharpness=sharpness,
        brightness=brightness,
        contrast=min(1.0, contrast),
        edge_density=edge_density,
        clipped_dark_ratio=clipped_dark,
        clipped_light_ratio=clipped_light,
        transition_risk=transition_risk,
        overall=overall,
    )


def assess_frame_quality(path_or_image: str | Path | Image.Image) -> FrameQuality:
    image = _comparison_copy(_open_rgb(path_or_image))
    return _assess_frame_quality_comparison(image)


def perceptual_dhash(path_or_image: str | Path | Image.Image, *, hash_size: int = 8) -> str:
    if hash_size <= 0:
        raise InputError("hash_size must be positive")
    image = (
        _open_rgb(path_or_image)
        .convert("L")
        .resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    )
    pixels = _pixels(image)
    bits = []
    stride = hash_size + 1
    for row in range(hash_size):
        for column in range(hash_size):
            bits.append(pixels[row * stride + column] > pixels[row * stride + column + 1])
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:0{math.ceil(hash_size * hash_size / 4)}x}"


def perceptual_hamming(left_hash: str, right_hash: str) -> int:
    if len(left_hash) != len(right_hash):
        raise InputError("perceptual hashes must have equal length")
    return (int(left_hash, 16) ^ int(right_hash, 16)).bit_count()


def normalized_pixel_hash(path_or_image: str | Path | Image.Image) -> str:
    if isinstance(path_or_image, Image.Image):
        image = path_or_image
        should_close = False
    else:
        image = Image.open(path_or_image)
        image.load()
        should_close = True
    assert image is not None
    try:
        oriented = ImageOps.exif_transpose(image)
        icc = image.info.get("icc_profile")
        if icc:
            alpha = oriented.getchannel("A") if "A" in oriented.getbands() else None
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            converted = ImageCms.profileToProfile(
                oriented.convert("RGB"),
                source_profile,
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
            assert converted is not None
            rgba = converted.convert("RGBA")
            if alpha is not None:
                rgba.putalpha(alpha)
        else:
            rgba = oriented.convert("RGBA")
    finally:
        if should_close:
            image.close()
    digest = hashlib.sha256()
    digest.update(b"sha256-rgba8-srgb-v1\x00")
    digest.update(struct.pack(">II", rgba.width, rgba.height))
    digest.update(rgba.tobytes())
    return digest.hexdigest()


def _changed_regions(
    diff: Image.Image, *, tiles_x: int, tiles_y: int, threshold: int
) -> tuple[DifferenceRegion, ...]:
    regions: list[DifferenceRegion] = []
    width, height = diff.size
    for row in range(tiles_y):
        top = row * height // tiles_y
        bottom = (row + 1) * height // tiles_y
        for column in range(tiles_x):
            left = column * width // tiles_x
            right = (column + 1) * width // tiles_x
            tile = diff.crop((left, top, right, bottom)).convert("L")
            histogram = tile.histogram()
            value_count = sum(histogram)
            if not value_count:
                continue
            ratio = sum(histogram[threshold:]) / value_count
            mean = _histogram_sum(histogram) / (255.0 * value_count)
            if ratio > 0 or mean > 0:
                regions.append(
                    DifferenceRegion((left, top, right - left, bottom - top), ratio, mean)
                )
    regions.sort(key=lambda item: (item.changed_ratio, item.mean_difference), reverse=True)
    return tuple(regions)


def _comparison_copy(image: Image.Image, *, max_dimension: int = 640) -> Image.Image:
    """Create a bounded comparison image while retaining source dimensions elsewhere."""
    if max(image.size) <= max_dimension:
        return image
    scale = max_dimension / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


def _compare_comparison_images(
    comparison_left: Image.Image,
    comparison_right: Image.Image,
    *,
    original_size: tuple[int, int],
    left_dhash: str,
    right_dhash: str,
    pixel_threshold: int,
    tiles_x: int,
    tiles_y: int,
) -> FrameDifference:
    """Compare bounded images whose resize/conversion work is already shared."""

    difference = ImageChops.difference(comparison_left, comparison_right)
    grayscale = difference.convert("L")
    grayscale_histogram = grayscale.histogram()
    value_count = max(1, sum(grayscale_histogram))
    changed = sum(grayscale_histogram[pixel_threshold:]) / value_count
    mean = _histogram_sum(grayscale_histogram) / max(1, 255 * value_count)
    left_edges = comparison_left.convert("L").filter(ImageFilter.FIND_EDGES)
    right_edges = comparison_right.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_histogram = ImageChops.difference(left_edges, right_edges).histogram()
    edge_count = max(1, sum(edge_histogram))
    edge_difference = _histogram_sum(edge_histogram) / max(1, 255 * edge_count)
    small_regions = _changed_regions(
        difference, tiles_x=tiles_x, tiles_y=tiles_y, threshold=pixel_threshold
    )
    scale_x = original_size[0] / comparison_left.width
    scale_y = original_size[1] / comparison_left.height
    regions = tuple(
        DifferenceRegion(
            (
                round(region.xywh[0] * scale_x),
                round(region.xywh[1] * scale_y),
                round(region.xywh[2] * scale_x),
                round(region.xywh[3] * scale_y),
            ),
            region.changed_ratio,
            region.mean_difference,
        )
        for region in small_regions
    )
    maximum_region_change = max((region.changed_ratio for region in regions), default=0.0)
    return FrameDifference(
        perceptual_hamming=perceptual_hamming(left_dhash, right_dhash),
        changed_pixel_ratio=changed,
        mean_pixel_difference=mean,
        edge_difference=edge_difference,
        maximum_region_change=maximum_region_change,
        regions=regions,
    )


def compare_frames(
    left: str | Path | Image.Image,
    right: str | Path | Image.Image,
    *,
    pixel_threshold: int = 12,
    tiles_x: int = 8,
    tiles_y: int = 8,
    left_dhash: str | None = None,
    right_dhash: str | None = None,
) -> FrameDifference:
    if not 0 <= pixel_threshold <= 255 or tiles_x <= 0 or tiles_y <= 0:
        raise InputError("invalid frame-comparison thresholds")
    left_image = _open_rgb(left)
    right_image = _open_rgb(right)
    if left_image.size != right_image.size:
        return FrameDifference(
            64, 1.0, 1.0, 1.0, 1.0, (DifferenceRegion((0, 0, *right_image.size), 1.0, 1.0),)
        )
    comparison_left = _comparison_copy(left_image)
    comparison_right = _comparison_copy(right_image)
    left_hash = left_dhash or perceptual_dhash(left_image)
    right_hash = right_dhash or perceptual_dhash(right_image)
    return _compare_comparison_images(
        comparison_left,
        comparison_right,
        original_size=left_image.size,
        left_dhash=left_hash,
        right_dhash=right_hash,
        pixel_threshold=pixel_threshold,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
    )


def analyze_frame_pair(
    current: str | Path,
    previous: str | Path | None = None,
) -> tuple[FrameQuality, FrameDifference | None]:
    """Decode the current/previous pair once for quality and difference analysis.

    The public quality and comparison functions intentionally retain their
    path-friendly behavior.  The visual pipeline, however, needs both results
    for every frame; opening the same PNG independently for each operation
    doubled disk decode work.  This helper owns the bounded two-image lifetime
    and delegates the exact existing calculations to the same pure functions.
    """

    def load_rgb(path: str | Path) -> Image.Image:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")

    current_image = load_rgb(current)
    previous_image = load_rgb(previous) if previous is not None else None
    quality = assess_frame_quality(current_image)
    difference = (
        compare_frames(previous_image, current_image) if previous_image is not None else None
    )
    return quality, difference


def analyze_frame_pair_with_hash(
    current: str | Path,
    previous: str | Path | None = None,
) -> tuple[FrameQuality, FrameDifference | None, str]:
    """Analyze a frame pair and return the current frame's dHash as a by-product.

    The visual pipeline needs the dHash again when it commits deterministic
    metadata.  Returning the hash from this already-decoded pair avoids opening
    every retained PNG a second time, while :func:`analyze_frame_pair` keeps its
    smaller historical return contract for callers that do not need the hash.
    """

    def load_rgb(path: str | Path) -> Image.Image:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")

    current_image = load_rgb(current)
    previous_image = load_rgb(previous) if previous is not None else None
    current_hash = perceptual_dhash(current_image)
    previous_hash = perceptual_dhash(previous_image) if previous_image is not None else None
    comparison_current = _comparison_copy(current_image)
    quality = _assess_frame_quality_comparison(comparison_current)
    if previous_image is None:
        difference = None
    elif previous_image.size != current_image.size:
        difference = compare_frames(
            previous_image,
            current_image,
            left_dhash=previous_hash,
            right_dhash=current_hash,
        )
    else:
        comparison_previous = _comparison_copy(previous_image)
        difference = _compare_comparison_images(
            comparison_previous,
            comparison_current,
            original_size=previous_image.size,
            left_dhash=previous_hash or perceptual_dhash(previous_image),
            right_dhash=current_hash,
            pixel_threshold=12,
            tiles_x=8,
            tiles_y=8,
        )
    return quality, difference, current_hash


def analyze_frame_sequence_with_hash(
    paths: Sequence[str | Path],
    *,
    max_workers: int = 1,
) -> tuple[tuple[FrameQuality, FrameDifference | None, str], ...]:
    """Analyze an ordered frame sequence with a one-frame decode window.

    Pair-oriented callers retain :func:`analyze_frame_pair_with_hash`, but a
    visual survey compares every frame with its predecessor.  Calling that
    helper independently would decode each interior PNG twice.  This variant
    retains only the previous decoded RGB image, producing the same quality,
    difference, and dHash values while reading each source PNG exactly once.
    """

    if max_workers <= 0:
        raise InputError("max_workers must be positive")

    def analyze_window(
        window: Sequence[str | Path],
    ) -> tuple[tuple[FrameQuality, FrameDifference | None, str], ...]:
        results: list[tuple[FrameQuality, FrameDifference | None, str]] = []
        previous_image: Image.Image | None = None
        previous_comparison: Image.Image | None = None
        previous_hash: str | None = None
        try:
            for path in window:
                with Image.open(path) as source:
                    source.load()
                    current_image = source.convert("RGB")
                current_comparison = _comparison_copy(current_image)
                try:
                    current_hash = perceptual_dhash(current_image)
                    quality = _assess_frame_quality_comparison(current_comparison)
                    if previous_image is None:
                        difference = None
                    elif previous_image.size != current_image.size:
                        difference = compare_frames(
                            previous_image,
                            current_image,
                            left_dhash=previous_hash,
                            right_dhash=current_hash,
                        )
                    else:
                        assert previous_comparison is not None
                        assert previous_hash is not None
                        difference = _compare_comparison_images(
                            previous_comparison,
                            current_comparison,
                            original_size=previous_image.size,
                            left_dhash=previous_hash,
                            right_dhash=current_hash,
                            pixel_threshold=12,
                            tiles_x=8,
                            tiles_y=8,
                        )
                    results.append((quality, difference, current_hash))
                except Exception:
                    if current_comparison is not current_image:
                        current_comparison.close()
                    current_image.close()
                    raise
                if previous_image is not None:
                    previous_image.close()
                    if previous_comparison is not None and previous_comparison is not previous_image:
                        previous_comparison.close()
                previous_image = current_image
                previous_comparison = current_comparison
                previous_hash = current_hash
        finally:
            if previous_image is not None:
                previous_image.close()
                if previous_comparison is not None and previous_comparison is not previous_image:
                    previous_comparison.close()
        return tuple(results)

    if len(paths) <= 1 or max_workers == 1:
        return analyze_window(paths)

    worker_count = min(max_workers, len(paths))
    chunk_size = math.ceil(len(paths) / worker_count)
    ranges = tuple(
        (start, min(start + chunk_size, len(paths)))
        for start in range(0, len(paths), chunk_size)
    )

    def analyze_range(bounds: tuple[int, int]) -> tuple[tuple[FrameQuality, FrameDifference | None, str], ...]:
        start, end = bounds
        # Include one predecessor so the first result in every non-initial
        # chunk has the same before/after comparison as the global sequence.
        window = paths[max(0, start - 1) : end]
        results = analyze_window(window)
        return results if start == 0 else results[1:]

    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="vsr-frame-analysis"
    ) as pool:
        chunks = tuple(pool.map(analyze_range, ranges))
    return tuple(result for chunk in chunks for result in chunk)


def normalize_ocr_for_comparison(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip()


def deduplication_decision(
    left: str | Path | Image.Image,
    right: str | Path | Image.Image,
    *,
    left_ocr: str | None = None,
    right_ocr: str | None = None,
    left_role: str | None = None,
    right_role: str | None = None,
    consequential_change: bool = False,
    protect_small_changes: bool = True,
) -> DeduplicationDecision:
    difference = compare_frames(left, right)
    left_text = normalize_ocr_for_comparison(left_ocr)
    right_text = normalize_ocr_for_comparison(right_ocr)
    ocr_changed = bool(left_text or right_text) and left_text != right_text
    protected: list[str] = []
    if protect_small_changes and ocr_changed:
        protected.append("ocr_change")
    if consequential_change:
        protected.append("consequential_event")
    sequence_roles = {"before", "action", "after", "result"}
    if left_role in sequence_roles and right_role in sequence_roles and left_role != right_role:
        protected.append("before_action_after_role")
    # A localized structural mutation can be tiny globally. Protect it even when dHash is unchanged.
    if protect_small_changes and (
        difference.maximum_region_change >= 0.012 or difference.edge_difference >= 0.0025
    ):
        protected.append("localized_pixel_or_structure_change")
    globally_similar = (
        difference.perceptual_hamming <= 5
        and difference.changed_pixel_ratio <= 0.008
        and difference.mean_pixel_difference <= 0.006
    )
    return DeduplicationDecision(
        is_duplicate=globally_similar and not protected,
        difference=difference,
        ocr_changed=ocr_changed,
        protected_reasons=tuple(dict.fromkeys(protected)),
    )
