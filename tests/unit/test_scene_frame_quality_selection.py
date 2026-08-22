from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep

import pytest
from PIL import Image, ImageDraw

import video_script_reconstructor.frame_extract as frame_extract_module
import video_script_reconstructor.frame_quality as frame_quality_module
import video_script_reconstructor.frame_selection as frame_selection_module
from video_script_reconstructor.errors import InputError, ValidationFailure
from video_script_reconstructor.frame_extract import (
    ExtractedFrame,
    build_batch_frame_extraction_command,
    build_concat_seek_frame_extraction_command,
    build_frame_extraction_command,
    extract_evidence_frame,
    extract_frames,
    format_frame_timestamp,
    parse_showinfo,
)
from video_script_reconstructor.frame_quality import (
    analyze_frame_pair,
    analyze_frame_pair_with_hash,
    analyze_frame_sequence_with_hash,
    assess_frame_quality,
    compare_frames,
    deduplication_decision,
    normalized_pixel_hash,
)
from video_script_reconstructor.frame_selection import FrameCandidate, select_frames
from video_script_reconstructor.media_probe import probe_media
from video_script_reconstructor.scene_detection import (
    SurveySignal,
    adaptive_signal_candidates,
    build_adaptive_detection_command,
    build_combined_survey_command,
    build_combined_survey_frame_command,
    detect_adaptive_candidates,
    detect_combined_survey_candidates,
    detect_combined_survey_frames,
    merge_survey_candidates,
    periodic_candidate_times,
    periodic_candidates,
    survey_candidate_importance_tier,
    survey_coverage,
)


def test_periodic_candidates_enforce_strict_ceiling_and_cover_duration() -> None:
    with pytest.raises(InputError, match="no greater than 30"):
        periodic_candidate_times(100_000, interval_seconds=31, strict=True)
    times = periodic_candidate_times(95_000, interval_seconds=30, strict=True)
    assert times == (0, 30_000, 60_000, 90_000)
    gaps = [right - left for left, right in zip(times, times[1:], strict=False)]
    assert max(gaps) <= 30_000
    assert 95_000 - times[-1] <= 30_000


def test_periodic_candidates_guard_a_container_tail_without_guessing_a_frame() -> None:
    # A muxed duration can extend a few milliseconds beyond the final video
    # frame (for example, because of audio padding).  The last safety request
    # must stay measurable while preserving the <=30-second coverage bound.
    times = periodic_candidate_times(18_000_002, interval_seconds=30, strict=True)
    assert times[-1] == 17_999_752
    assert times[-1] < 18_000_002
    assert max(
        right - left for left, right in zip(times, times[1:], strict=False)
    ) <= 30_000
    assert 18_000_002 - times[-1] == 250


def test_periodic_candidates_allow_disabling_the_tail_guard_explicitly() -> None:
    times = periodic_candidate_times(
        30_001,
        interval_seconds=30,
        strict=True,
        tail_guard_ms=0,
    )
    assert times == (0, 30_000)


def test_periodic_tail_guard_keeps_short_intervals_sorted() -> None:
    times = periodic_candidate_times(
        1_000,
        interval_seconds=0.1,
        strict=True,
        tail_guard_ms=250,
    )
    assert times == tuple(sorted(set(times)))
    assert times[-1] == 750
    assert max(
        right - left for left, right in zip(times, times[1:], strict=False)
    ) <= 100


def test_merge_preserves_scene_and_periodic_reasons() -> None:
    safety = periodic_candidates(50_000, interval_seconds=25)
    scene = safety[1].__class__(
        "VC999999", 25_100, 25_100, 123, "1/1000", ("scene_cut",), 0.9, "ffmpeg-showinfo"
    )
    merged = merge_survey_candidates((safety, (scene,)), merge_tolerance_ms=150)
    joined = next(item for item in merged if item.actual_ms == 25_100)
    assert joined.reasons == ("periodic_safety", "scene_cut")
    assert joined.raw_pts == 123


def test_survey_importance_bands_protect_measured_changes_and_report_coverage() -> None:
    periodic = periodic_candidates(65_000, interval_seconds=30)
    scene = periodic[1].__class__(
        "VC999999", 30_100, 30_100, 123, "1/1000", ("scene_cut",), 0.1, "ffmpeg-showinfo"
    )
    ocr = periodic[1].__class__(
        "VC999998", 30_200, 30_200, 124, "1/1000", ("ocr_change",), 0.1, "decoded"
    )
    assert survey_candidate_importance_tier(periodic[0]) == "low"
    assert survey_candidate_importance_tier(scene) == "very_high"
    assert survey_candidate_importance_tier(ocr) == "very_high"
    receipt = survey_coverage((*periodic, scene, ocr), duration_ms=65_000)
    assert receipt["candidate_count"] == 5
    assert receipt["importance_counts"]["very_high"] == 2
    assert receipt["protected_candidate_ids"] == ("VC999999", "VC999998")
    assert receipt["strict_gap_satisfied"] is True


def test_merge_prefers_protected_change_over_higher_scoring_periodic_context() -> None:
    periodic = periodic_candidates(40_000, interval_seconds=30)[1]
    weak_scene = periodic.__class__(
        "VC999997",
        periodic.requested_ms + 100,
        periodic.requested_ms + 100,
        12,
        "1/1000",
        ("scene_cut",),
        0.01,
        "ffmpeg-showinfo",
    )
    merged = merge_survey_candidates(((periodic,), (weak_scene,)), merge_tolerance_ms=150)
    assert len(merged) == 1
    assert merged[0].candidate_id == "VC000001"
    assert merged[0].actual_ms == weak_scene.actual_ms
    assert merged[0].reasons == ("periodic_safety", "scene_cut")


def test_merge_collapses_sustained_adaptive_motion_without_absorbing_safety_samples() -> None:
    adaptive = tuple(
        periodic_candidates(10_000, interval_seconds=30)[0].__class__(
            f"VC{index:06d}",
            time_ms,
            time_ms,
            index,
            "1/1000",
            ("adaptive_frame_difference",),
            0.7,
            "ffmpeg-sampled-scene-score-showinfo",
        )
        for index, time_ms in enumerate((1_000, 1_500, 2_000, 2_500), 1)
    )
    safety = periodic_candidates(10_000, interval_seconds=30)[0]
    merged = merge_survey_candidates((adaptive, (safety,)))

    # One semantic packet is enough for the sustained motion run, while the
    # explicit periodic safety sample remains independently attributable.
    assert len(merged) == 2
    motion = next(item for item in merged if item.reasons == ("adaptive_frame_difference",))
    assert motion.actual_ms == 1_000
    assert safety.actual_ms in {item.actual_ms for item in merged}


def test_merge_adaptive_cluster_keeps_post_motion_and_ocr_boundaries() -> None:
    base = periodic_candidates(10_000, interval_seconds=30)[0]
    adaptive = base.__class__(
        "VC000010", 1_000, 1_000, 10, "1/1000", ("adaptive_frame_difference",), 0.7, "decoded"
    )
    settled = base.__class__(
        "VC000011", 1_500, 1_500, 11, "1/1000", ("post_motion_stable",), 0.7, "decoded"
    )
    ocr = base.__class__(
        "VC000012", 2_000, 2_000, 12, "1/1000", ("ocr_change",), 1.0, "decoded"
    )
    merged = merge_survey_candidates(((adaptive, settled, ocr),))

    assert [item.actual_ms for item in merged] == [1_000, 1_500, 2_000]


def test_merge_adaptive_cluster_keeps_high_importance_perceptual_change() -> None:
    base = periodic_candidates(10_000, interval_seconds=30)[0]
    first = base.__class__(
        "VC000020", 1_000, 1_000, 20, "1/1000", ("adaptive_frame_difference",), 0.7, "decoded"
    )
    perceptual = base.__class__(
        "VC000021", 1_500, 1_500, 21, "1/1000", ("perceptual_change",), 0.9, "decoded"
    )
    last = base.__class__(
        "VC000022", 2_000, 2_000, 22, "1/1000", ("adaptive_frame_difference",), 0.7, "decoded"
    )

    merged = merge_survey_candidates(((first, perceptual, last),))

    assert [item.actual_ms for item in merged] == [1_000, 1_500, 2_000]


def test_merge_rejects_an_adaptive_cluster_window_narrower_than_strict_window() -> None:
    with pytest.raises(InputError, match="cannot be lower"):
        merge_survey_candidates((), merge_tolerance_ms=250, adaptive_cluster_tolerance_ms=100)


def test_adaptive_signals_create_ocr_and_small_state_change_candidates() -> None:
    candidates = adaptive_signal_candidates(
        [
            SurveySignal(0, frame_difference=0.001, ocr_text="total 42"),
            SurveySignal(
                1000,
                raw_pts=90_000,
                time_base="1/90000",
                frame_difference=0.003,
                ocr_text="total 43",
            ),
        ]
    )
    assert len(candidates) == 1
    assert candidates[0].reasons == ("ocr_change",)
    assert candidates[0].raw_pts == 90_000


def test_adaptive_detection_command_is_bounded_sampling_and_safe_argv(tmp_path: Path) -> None:
    source = tmp_path / "clip ; ignored.mp4"
    command = build_adaptive_detection_command(source, sample_fps=2, threshold=0.00001)
    assert command[command.index("-i") + 1] == str(source)
    assert "fps=2" in command[command.index("-vf") + 1]
    with pytest.raises(InputError, match="no more than 30"):
        build_adaptive_detection_command(source, sample_fps=31)


def test_adaptive_detection_default_filters_codec_noise() -> None:
    command = build_adaptive_detection_command(Path("clip.mp4"))
    assert "select=gt(scene\\,0.003)" in command[command.index("-vf") + 1]


def test_standalone_survey_commands_accept_bounded_codec_threads() -> None:
    source = Path("clip.mp4")
    command = build_adaptive_detection_command(source, ffmpeg_threads=4)
    assert command[command.index("-threads") + 1] == "4"
    with pytest.raises(InputError, match="ffmpeg_threads must be positive"):
        build_adaptive_detection_command(source, ffmpeg_threads=0)


def test_combined_survey_command_shares_decode_and_labels_branches() -> None:
    command = build_combined_survey_command(Path("clip.mp4"))
    graph = command[command.index("-filter_complex") + 1]
    assert "split=3" in graph
    assert "showinfo@hard" in graph
    assert "showinfo@adaptive" in graph
    assert "nullsink" in graph
    assert "[keepalive]" in graph
    assert "[survey_output]null[keepalive]" in graph
    assert "-frames:v" not in command
    assert command.count("-map") == 1
    assert command[command.index("-map") + 1] == "[keepalive]"


def test_combined_candidate_survey_keeps_empty_branches_one_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "generated" / "slide-lecture.mp4"
    import video_script_reconstructor.scene_detection as scene_detection_module

    calls = 0
    original = scene_detection_module._execute_detection

    def counted(command: list[str], **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return original(command, **kwargs)

    monkeypatch.setattr(scene_detection_module, "_execute_detection", counted)
    hard, adaptive = detect_combined_survey_candidates(source)

    assert calls == 1
    assert hard == ()
    assert any(candidate.actual_ms == 2_000 for candidate in adaptive)


def test_combined_candidate_survey_consumes_late_scene_cuts(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is required for the real survey regression fixture")
    source = tmp_path / "late-cuts.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x180:r=10:d=8",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=320x180:r=10:d=8",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x180:r=10:d=8",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
            "-map",
            "[v]",
            "-c:v",
            "ffv1",
            "-y",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    candidate_hard, candidate_adaptive = detect_combined_survey_candidates(
        source, timeout_seconds=30
    )
    emitted_hard, emitted_adaptive, _frames = detect_combined_survey_frames(
        source,
        tmp_path / "emitted",
        (0, 8_000, 16_000),
        timeout_seconds=30,
    )

    assert [item.actual_ms for item in candidate_hard] == [8_000, 16_000]
    assert candidate_hard == emitted_hard
    assert candidate_adaptive == emitted_adaptive


def test_shared_survey_frame_command_emits_only_safe_branches() -> None:
    command = build_combined_survey_frame_command(
        Path("clip.mp4"), Path("frames"), (0, 30_000)
    )
    rendered = " ".join(command)
    assert rendered.count("-map") == 2
    assert "showinfo@periodic" in rendered
    assert "nullsink" in rendered
    assert "hard-%06d.png" in rendered
    assert "periodic-%06d.png" in rendered


def test_shared_survey_frame_command_accepts_bounded_codec_threads() -> None:
    command = build_combined_survey_frame_command(
        Path("clip.mp4"), Path("frames"), (0, 30_000), ffmpeg_threads=4
    )
    assert command[command.index("-threads") + 1] == "4"


def test_showinfo_parser_uses_measured_pts_not_requested_time(tmp_path: Path) -> None:
    stderr = """
[Parsed_showinfo_1 @ abc] config in time_base: 1/90000, frame_rate: 30000/1001
[Parsed_showinfo_1 @ abc] n:   0 pts: 1641600 pts_time:18.24 duration:3003 s:1920x1080
"""
    timing = parse_showinfo(stderr)[0]
    assert timing.actual_ms == 18_240
    assert timing.raw_pts == 1_641_600
    assert timing.time_base == "1/90000"
    command = build_frame_extraction_command(
        tmp_path / "a b;$(x).mp4", 18_200, tmp_path / "f.png", ffmpeg_threads=4
    )
    assert command[-1] == str(tmp_path / "f.png")
    assert command[command.index("-ss") + 1] == "18.2"
    assert "-copyts" in command
    assert command[command.index("-threads") + 1] == "4"
    assert "182/10" not in command
    assert format_frame_timestamp(timing.actual_ms) == "00h00m18s240"


def _state_images(tmp_path: Path) -> tuple[Path, Path, Path]:
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    same = tmp_path / "same.png"
    for path, text in ((before, "Value: 42"), (after, "Value: 43"), (same, "Value: 42")):
        image = Image.new("RGB", (640, 360), "white")
        ImageDraw.Draw(image).text((40, 120), text, fill="black")
        image.save(path)
    return before, after, same


def test_small_one_character_change_is_never_deduplicated(tmp_path: Path) -> None:
    before, after, same = _state_images(tmp_path)
    mutation = deduplication_decision(before, after, left_ocr="Value: 42", right_ocr="Value: 43")
    duplicate = deduplication_decision(before, same, left_ocr="Value: 42", right_ocr="Value: 42")
    assert mutation.is_duplicate is False
    assert "ocr_change" in mutation.protected_reasons
    assert duplicate.is_duplicate is True
    assert normalized_pixel_hash(before) == normalized_pixel_hash(same)


def test_quality_and_selection_retain_consequential_after_frame(tmp_path: Path) -> None:
    before, after, same = _state_images(tmp_path)
    quality = assess_frame_quality(after)
    assert 0 <= quality.overall <= 1
    candidates = [
        FrameCandidate(
            "F000001",
            before,
            0,
            relevance=0.8,
            importance=0.8,
            ocr_readability=0.8,
            evidence_role="before",
            ocr_text="Value: 42",
        ),
        FrameCandidate(
            "F000002",
            same,
            100,
            relevance=0.2,
            importance=0.2,
            ocr_readability=0.8,
            evidence_role="context",
            ocr_text="Value: 42",
        ),
        FrameCandidate(
            "F000003",
            after,
            200,
            relevance=0.9,
            importance=1,
            ocr_readability=0.9,
            evidence_role="after",
            ocr_text="Value: 43",
            consequential_change=True,
        ),
    ]
    result = select_frames(
        candidates, duration_ms=1_000, important_event_count=1, evidence_density_per_minute=1
    )
    selected_ids = {item.candidate.frame_id for item in result.selected}
    assert {"F000001", "F000003"} <= selected_ids
    assert "F000002" in result.duplicate_frame_ids


def test_selection_reuses_precomputed_perceptual_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after, _same = _state_images(tmp_path)
    quality = assess_frame_quality(before)
    candidates = [
        FrameCandidate(
            "F000001",
            before,
            0,
            relevance=1.0,
            importance=1.0,
            mandatory=True,
            quality=quality,
            perceptual_hash="0123456789abcdef",
        ),
        FrameCandidate(
            "F000002",
            after,
            100,
            relevance=0.9,
            importance=0.9,
            mandatory=True,
            quality=quality,
            perceptual_hash="fedcba9876543210",
        ),
    ]

    def fail_recompute(_path: object, **_kwargs: object) -> str:
        raise AssertionError("selection reopened a PNG despite a precomputed dHash")

    monkeypatch.setattr(frame_selection_module, "perceptual_dhash", fail_recompute)
    result = select_frames(candidates, duration_ms=1_000, deduplicate=False)
    assert {item.candidate.frame_id for item in result.selected} == {"F000001", "F000002"}


def test_selection_uses_equal_pixel_hash_before_perceptual_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, _after, same = _state_images(tmp_path)
    quality = assess_frame_quality(before)
    pixel_hash = normalized_pixel_hash(before)
    candidates = [
        FrameCandidate(
            "F000001",
            before,
            0,
            relevance=0.2,
            importance=0.2,
            quality=quality,
            pixel_hash=pixel_hash,
        ),
        FrameCandidate(
            "F000002",
            same,
            100,
            relevance=0.2,
            importance=0.2,
            quality=quality,
            pixel_hash=pixel_hash,
        ),
    ]

    def fail_recompute(_path: object, **_kwargs: object) -> str:
        raise AssertionError("equal canonical pixels must not trigger perceptual decoding")

    monkeypatch.setattr(frame_selection_module, "perceptual_dhash", fail_recompute)
    result = select_frames(
        candidates,
        duration_ms=1_000,
        important_event_count=1,
        evidence_density_per_minute=1,
    )
    assert [item.candidate.frame_id for item in result.selected] == ["F000001"]
    assert result.duplicate_frame_ids == ("F000002",)
    assert result.provenance[1].reason == "exact_pixel_hash"
    assert result.provenance[1].perceptual_hamming == 0


def test_importance_aware_selection_collapses_repetitive_low_context_with_receipt(
    tmp_path: Path,
) -> None:
    """Codec-sized visual jitter is deduped only for low-importance context."""

    base = tmp_path / "base.png"
    jitter = tmp_path / "jitter.png"
    Image.new("RGB", (640, 360), "white").save(base)
    noisy = Image.new("RGB", (640, 360), "white")
    # A bounded static overlay is large enough to miss the strict duplicate
    # ratio but remains a near-duplicate under the low-importance policy.
    ImageDraw.Draw(noisy).rectangle((10, 10, 110, 110), fill=(220, 220, 220))
    noisy.save(jitter)
    quality = assess_frame_quality(base)
    candidates = [
        FrameCandidate(
            "F000001",
            base,
            0,
            relevance=0.2,
            importance=0.2,
            quality=quality,
        ),
        FrameCandidate(
            "F000002",
            jitter,
            100,
            relevance=0.2,
            importance=0.2,
            quality=quality,
        ),
    ]
    result = select_frames(
        candidates,
        duration_ms=1_000,
        important_event_count=1,
        evidence_density_per_minute=1,
    )

    assert [item.candidate.frame_id for item in result.selected] == ["F000001"]
    assert result.duplicate_frame_ids == ("F000002",)
    assert result.coverage["duplicate_count"] == 1
    assert result.coverage["coverage_ratio"] == 1.0
    representative = result.provenance[0]
    duplicate = result.provenance[1]
    assert representative.covered_frame_ids == ("F000001", "F000002")
    assert duplicate.representative_frame_id == "F000001"
    assert duplicate.reason == "near_duplicate_low_importance"
    assert duplicate.perceptual_hamming is not None
    assert duplicate.changed_pixel_ratio is not None


def test_importance_aware_selection_never_deduplicates_high_change_frames(
    tmp_path: Path,
) -> None:
    image = tmp_path / "same.png"
    Image.new("RGB", (640, 360), "white").save(image)
    quality = assess_frame_quality(image)
    candidates = [
        FrameCandidate(
            "F000001",
            image,
            0,
            relevance=0.4,
            importance=0.8,
            quality=quality,
        ),
        FrameCandidate(
            "F000002",
            image,
            100,
            relevance=0.4,
            importance=0.95,
            consequential_change=True,
            quality=quality,
            reasons=("ocr_change",),
        ),
    ]
    result = select_frames(
        candidates,
        duration_ms=1_000,
        important_event_count=0,
        evidence_density_per_minute=1,
    )

    assert {item.candidate.frame_id for item in result.selected} == {
        "F000001",
        "F000002",
    }
    assert not result.duplicate_frame_ids
    assert result.coverage["importance_coverage"]["high"] == 1.0
    assert result.coverage["importance_coverage"]["very_high"] == 1.0


def test_selection_provenance_is_deterministic_and_marks_budget_rejections(
    tmp_path: Path,
) -> None:
    image = tmp_path / "state.png"
    Image.new("RGB", (320, 180), "gray").save(image)
    quality = assess_frame_quality(image)
    candidates = [
        FrameCandidate(
            f"F00000{index}",
            image,
            index * 100,
            relevance=0.1 + index * 0.01,
            importance=0.1,
            quality=quality,
        )
        for index in range(1, 5)
    ]
    result = select_frames(
        candidates,
        duration_ms=1_000,
        important_event_count=0,
        evidence_density_per_minute=1,
        deduplicate=False,
    )

    assert result.coverage["candidate_count"] == 4
    assert result.coverage["selected_count"] == 1
    assert result.coverage["low_score_count"] == 3
    assert [item.frame_id for item in result.provenance] == [
        "F000001",
        "F000002",
        "F000003",
        "F000004",
    ]
    assert all(
        item.reason == "duration_aware_budget_exhausted"
        for item in result.provenance
        if item.status == "low_score"
    )


def test_pair_analysis_matches_independent_quality_and_difference(tmp_path: Path) -> None:
    before, after, _same = _state_images(tmp_path)
    quality, difference = analyze_frame_pair(after, before)
    assert quality == assess_frame_quality(after)
    assert difference == compare_frames(before, after)


def test_pair_analysis_with_hash_reuses_the_current_frame_hash(tmp_path: Path) -> None:
    before, after, _same = _state_images(tmp_path)
    quality, difference, current_hash = analyze_frame_pair_with_hash(after, before)
    expected_quality, expected_difference = analyze_frame_pair(after, before)
    assert quality == expected_quality
    assert difference == expected_difference
    assert current_hash == frame_quality_module.perceptual_dhash(after)


def test_pair_analysis_decodes_each_png_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before, after, _same = _state_images(tmp_path)
    original_open = frame_quality_module.Image.open
    opened = 0

    def counted_open(*args: object, **kwargs: object) -> object:
        nonlocal opened
        opened += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(frame_quality_module.Image, "open", counted_open)
    analyze_frame_pair(after, before)
    assert opened == 2


def test_sequence_analysis_reuses_a_sliding_decode_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after, same = _state_images(tmp_path)
    paths = (before, after, same)
    expected = tuple(
        analyze_frame_pair_with_hash(path, paths[index - 1] if index else None)
        for index, path in enumerate(paths)
    )
    original_open = frame_quality_module.Image.open
    opened = 0

    def counted_open(*args: object, **kwargs: object) -> object:
        nonlocal opened
        opened += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(frame_quality_module.Image, "open", counted_open)
    actual = analyze_frame_sequence_with_hash(paths)

    assert actual == expected
    assert opened == len(paths)

    parallel = analyze_frame_sequence_with_hash(paths, max_workers=2)
    assert parallel == expected


def test_real_generated_video_is_probed_surveyed_and_extracted_with_measured_pts(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "generated" / "slide-lecture.mp4"
    probe = probe_media(source)
    assert probe.duration_ms == 4_000
    adaptive = detect_adaptive_candidates(source)
    assert any(candidate.actual_ms == 2_000 for candidate in adaptive)
    frame = extract_evidence_frame(source, 1_851, tmp_path, frame_id="F000001")
    assert frame.actual_ms == 1_900
    assert frame.actual_ms != frame.requested_ms
    assert frame.raw_pts == 19_456
    assert frame.time_base == "1/10240"
    with Image.open(frame.path) as image:
        assert image.size == (640, 360)


def test_shared_survey_emits_measured_hard_and_periodic_frames(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "generated" / "slide-lecture.mp4"
    hard, adaptive, frames = detect_combined_survey_frames(source, tmp_path, (0, 2_000))
    periodic = [item for item in frames if item.branch == "periodic"]
    assert len(periodic) == 2
    assert all(item.path.is_file() for item in frames)
    assert {item.branch for item in frames} <= {"hard", "periodic"}
    assert all(item.timing.width == 640 and item.timing.height == 360 for item in frames)
    assert all(candidate.actual_ms is not None for candidate in hard + adaptive)


def test_shared_survey_discards_hard_sentinel_and_keeps_measured_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")

    def fake_execute(command: list[str], **_kwargs: object) -> str:
        hard_pattern = Path(command[command.index("-y") + 1])
        periodic_pattern = Path(command[-1])
        hard_pattern.parent.mkdir(parents=True, exist_ok=True)
        # The first hard file is the guaranteed mux sentinel; the second is
        # the one measured scene frame that the parser should retain.
        (hard_pattern.parent / "hard-000001.png").write_bytes(b"sentinel")
        (hard_pattern.parent / "hard-000002.png").write_bytes(b"hard")
        (periodic_pattern.parent / "periodic-000001.png").write_bytes(b"periodic-0")
        (periodic_pattern.parent / "periodic-000002.png").write_bytes(b"periodic-1")
        return """
[showinfo@hard @ x] config in time_base: 1/1000, frame_rate: 1/1
[showinfo@hard @ x] n: 0 pts: 100 pts_time: 0.1 s:640x360
[showinfo@periodic @ x] config in time_base: 1/1000, frame_rate: 1/1
[showinfo@periodic @ x] n: 0 pts: 0 pts_time: 0 s:640x360
[showinfo@periodic @ x] n: 1 pts: 1000 pts_time: 1 s:640x360
"""

    monkeypatch.setattr(
        "video_script_reconstructor.scene_detection._execute_detection", fake_execute
    )
    hard, _adaptive, frames = detect_combined_survey_frames(source, tmp_path / "frames", (0, 1_000))

    assert [candidate.actual_ms for candidate in hard] == [100]
    hard_frames = [item for item in frames if item.branch == "hard"]
    periodic_frames = [item for item in frames if item.branch == "periodic"]
    assert [item.timing.actual_ms for item in hard_frames] == [100]
    assert [item.requested_ms for item in periodic_frames] == [0, 1_000]


def test_batched_frame_extraction_keeps_one_measured_frame_per_request(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "generated" / "slide-lecture.mp4"
    frames = extract_frames(
        source,
        [0, 1_851, 3_000],
        tmp_path,
        batch=True,
        timeout_seconds=30.0,
    )
    assert [frame.requested_ms for frame in frames] == [0, 1_851, 3_000]
    assert [frame.actual_ms for frame in frames] == [0, 1_900, 3_000]
    assert all(frame.path.is_file() for frame in frames)


def test_ffmpeg_thread_budget_preserves_measured_frames_and_pixels(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "generated" / "slide-lecture.mp4"
    single_thread = extract_frames(
        source,
        [0, 1_851, 3_000],
        tmp_path / "threads-1",
        max_workers=2,
        batch=True,
        ffmpeg_threads=1,
        timeout_seconds=30.0,
    )
    bounded = extract_frames(
        source,
        [0, 1_851, 3_000],
        tmp_path / "threads-4",
        max_workers=2,
        batch=True,
        ffmpeg_threads=4,
        timeout_seconds=30.0,
    )
    assert [(frame.actual_ms, frame.raw_pts, frame.time_base) for frame in single_thread] == [
        (frame.actual_ms, frame.raw_pts, frame.time_base) for frame in bounded
    ]
    assert [normalized_pixel_hash(frame.path) for frame in single_thread] == [
        normalized_pixel_hash(frame.path) for frame in bounded
    ]


def test_frame_extraction_reuses_caller_pool_without_shutting_it_down(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "generated" / "slide-lecture.mp4"
    with ThreadPoolExecutor(max_workers=2) as pool:
        frames = extract_frames(
            source,
            [0, 1_851, 3_000],
            tmp_path,
            max_workers=2,
            batch=True,
            timeout_seconds=30.0,
            worker_pool=pool,
        )
        assert [frame.actual_ms for frame in frames] == [0, 1_900, 3_000]
        assert pool.submit(lambda: "pool-still-alive").result() == "pool-still-alive"


def test_batch_command_stops_at_last_request_lookahead(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    pattern = tmp_path / "frame-%06d.png"
    command = build_batch_frame_extraction_command(source, (2_000, 3_000), pattern)
    assert command[command.index("-i") + 1] == str(source)
    assert command[command.index("-to") + 1] == "3.250"
    assert command[command.index("-vf") + 1].startswith("select=")


def test_concat_seek_command_uses_stateful_guarded_windows(tmp_path: Path) -> None:
    schedule = tmp_path / "schedule.ffconcat"
    pattern = tmp_path / "frame-%06d.png"
    command = build_concat_seek_frame_extraction_command(
        schedule, pattern, ffmpeg_threads=4
    )
    assert command[command.index("-i") + 1] == str(schedule)
    assert command[command.index("-f") + 1] == "concat"
    assert command[command.index("-segment_time_metadata") + 1] == "1"
    assert "concatdec_select" in command[command.index("-vf") + 1]
    assert "prev_selected_t" in command[command.index("-vf") + 1]
    assert command[command.index("-threads") + 1] == "4"


def test_concat_seek_rebases_measured_pts_to_source_clock() -> None:
    synthetic = frame_extract_module.DecodedFrameTiming(
        output_index=1,
        raw_pts=10_742,
        actual_ms=1_049,
        time_base="1/10240",
        width=640,
        height=360,
    )
    source = frame_extract_module._source_timing_from_concat(
        synthetic,
        position=1,
        requested_ms=1_851,
    )
    assert source.actual_ms == 1_900
    assert source.raw_pts == 19_456
    assert source.time_base == "1/10240"


def test_large_sparse_concat_seek_matches_independent_exact_frames(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "generated" / "slide-lecture.mp4"
    requested = list(range(0, 3_751, 250))
    accelerated = extract_frames(
        source,
        requested,
        tmp_path / "accelerated",
        batch=True,
        max_workers=2,
        timeout_seconds=30.0,
    )
    exact = extract_frames(
        source,
        requested,
        tmp_path / "exact",
        batch=False,
        max_workers=2,
        timeout_seconds=30.0,
    )
    assert [(frame.actual_ms, frame.raw_pts, frame.time_base) for frame in accelerated] == [
        (frame.actual_ms, frame.raw_pts, frame.time_base) for frame in exact
    ]
    assert [normalized_pixel_hash(frame.path) for frame in accelerated] == [
        normalized_pixel_hash(frame.path) for frame in exact
    ]


def test_concat_seek_failure_falls_back_to_individual_exact_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    requested = list(range(0, 16_000, 1_000))
    exact_calls: list[int] = []

    def fail_concat(*_args: object, **_kwargs: object) -> tuple[ExtractedFrame, ...]:
        raise ValidationFailure("fixture concat route is unavailable")

    def fake_exact(
        _media_path: Path,
        requested_ms: int,
        output_dir: Path,
        *,
        frame_id: str,
        **_kwargs: object,
    ) -> ExtractedFrame:
        exact_calls.append(requested_ms)
        return ExtractedFrame(
            frame_id=frame_id,
            path=output_dir / f"{frame_id}.png",
            requested_ms=requested_ms,
            actual_ms=requested_ms,
            raw_pts=requested_ms,
            time_base="1/1000",
            frame_index=None,
            offset_ms=0,
            timestamp_source="fixture",
            width=640,
            height=360,
        )

    monkeypatch.setattr(frame_extract_module, "_run_concat_seek_group", fail_concat)
    monkeypatch.setattr(frame_extract_module, "extract_evidence_frame", fake_exact)
    frames = extract_frames(
        source,
        requested,
        tmp_path / "frames",
        batch=True,
        max_workers=2,
    )
    assert sorted(exact_calls) == requested
    assert [frame.requested_ms for frame in frames] == requested


def test_concat_seek_preserves_frame_ids_across_noncontiguous_exact_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    first = tuple(range(0, 16_000, 1_000))
    dense = (20_000, 20_100, 20_500)
    second = tuple(range(30_000, 46_000, 1_000))
    groups = (first, dense, second)
    concat_first_numbers: list[int] = []

    def make_frames(
        requested_times_ms: tuple[int, ...],
        output_dir: Path,
        first_frame_number: int,
    ) -> tuple[ExtractedFrame, ...]:
        return tuple(
            ExtractedFrame(
                frame_id=f"F{first_frame_number + offset:06d}",
                path=output_dir / f"F{first_frame_number + offset:06d}.png",
                requested_ms=requested_ms,
                actual_ms=requested_ms,
                raw_pts=requested_ms,
                time_base="1/1000",
                frame_index=None,
                offset_ms=0,
                timestamp_source="fixture",
                width=640,
                height=360,
            )
            for offset, requested_ms in enumerate(requested_times_ms)
        )

    def fake_concat(
        _media_path: Path,
        requested_times_ms: tuple[int, ...],
        output_dir: Path,
        *,
        first_frame_number: int,
        **_kwargs: object,
    ) -> tuple[ExtractedFrame, ...]:
        concat_first_numbers.append(first_frame_number)
        return make_frames(requested_times_ms, output_dir, first_frame_number)

    def fake_batch(
        _media_path: Path,
        requested_times_ms: tuple[int, ...],
        output_dir: Path,
        *,
        first_frame_number: int,
        **_kwargs: object,
    ) -> tuple[ExtractedFrame, ...]:
        return make_frames(requested_times_ms, output_dir, first_frame_number)

    monkeypatch.setattr(frame_extract_module, "_batch_request_groups", lambda _times: groups)
    monkeypatch.setattr(
        frame_extract_module, "_prefer_exact_group", lambda group: group != dense
    )
    monkeypatch.setattr(frame_extract_module, "_run_concat_seek_group", fake_concat)
    monkeypatch.setattr(frame_extract_module, "_run_batch_group", fake_batch)
    frames = extract_frames(
        source,
        first + dense + second,
        tmp_path / "frames",
        batch=True,
        max_workers=2,
    )
    assert concat_first_numbers == [1, 20]
    assert [frame.frame_id for frame in frames] == [
        f"F{number:06d}" for number in range(1, len(frames) + 1)
    ]


def test_batched_selector_emits_all_requests_without_exact_seek_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "generated" / "slide-lecture.mp4"

    def fail_exact_seek(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("batch extraction unexpectedly fell back to exact seeking")

    monkeypatch.setattr(frame_extract_module, "extract_evidence_frame", fail_exact_seek)
    monkeypatch.setattr(frame_extract_module, "_prefer_exact_group", lambda _group: False)
    frames = extract_frames(
        source,
        [0, 1_851, 3_000],
        tmp_path,
        batch=True,
        timeout_seconds=30.0,
    )
    assert [frame.actual_ms for frame in frames] == [0, 1_900, 3_000]


def test_batch_failure_falls_back_only_for_the_failed_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    groups = ((0, 1_000, 2_000), (3_000, 4_000, 5_000))
    batch_calls: list[tuple[int, ...]] = []
    exact_calls: list[int] = []

    def fake_exact(
        _media_path: Path,
        requested_ms: int,
        output_dir: Path,
        *,
        frame_id: str,
        **_kwargs: object,
    ) -> ExtractedFrame:
        exact_calls.append(requested_ms)
        return ExtractedFrame(
            frame_id=frame_id,
            path=output_dir / f"{frame_id}.png",
            requested_ms=requested_ms,
            actual_ms=requested_ms,
            raw_pts=requested_ms,
            time_base="1/1000",
            frame_index=None,
            offset_ms=0,
            timestamp_source="fixture",
            width=640,
            height=360,
        )

    def fake_batch(
        _media_path: Path,
        requested_times_ms: tuple[int, ...],
        output_dir: Path,
        *,
        first_frame_number: int,
        **_kwargs: object,
    ) -> tuple[ExtractedFrame, ...]:
        batch_calls.append(requested_times_ms)
        if requested_times_ms == groups[0]:
            raise ValidationFailure("fixture batch timing mismatch")
        return tuple(
            ExtractedFrame(
                frame_id=f"F{first_frame_number + offset:06d}",
                path=output_dir / f"F{first_frame_number + offset:06d}.png",
                requested_ms=requested_ms,
                actual_ms=requested_ms,
                raw_pts=requested_ms,
                time_base="1/1000",
                frame_index=None,
                offset_ms=0,
                timestamp_source="fixture",
                width=640,
                height=360,
            )
            for offset, requested_ms in enumerate(requested_times_ms)
        )

    monkeypatch.setattr(frame_extract_module, "_batch_request_groups", lambda _times: groups)
    monkeypatch.setattr(frame_extract_module, "_prefer_exact_group", lambda _group: False)
    monkeypatch.setattr(frame_extract_module, "_run_batch_group", fake_batch)
    monkeypatch.setattr(frame_extract_module, "extract_evidence_frame", fake_exact)

    frames = extract_frames(
        source,
        [0, 1_000, 2_000, 3_000, 4_000, 5_000],
        tmp_path / "frames",
        batch=True,
        max_workers=2,
    )

    assert sorted(batch_calls) == sorted(groups)
    assert exact_calls == [0, 1_000, 2_000]
    assert [frame.requested_ms for frame in frames] == [0, 1_000, 2_000, 3_000, 4_000, 5_000]


def test_batch_failure_clears_partial_outputs_before_full_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    output_dir = tmp_path / "frames"
    orphan = output_dir / "F000001__partial.png"

    def fake_exact(
        _media_path: Path,
        requested_ms: int,
        target_dir: Path,
        *,
        frame_id: str,
        **_kwargs: object,
    ) -> ExtractedFrame:
        assert not list(target_dir.rglob("*.png"))
        return ExtractedFrame(
            frame_id=frame_id,
            path=target_dir / f"{frame_id}.png",
            requested_ms=requested_ms,
            actual_ms=requested_ms,
            raw_pts=requested_ms,
            time_base="1/1000",
            frame_index=None,
            offset_ms=0,
            timestamp_source="fixture",
            width=640,
            height=360,
        )

    def fail_batch(*_args: object, **kwargs: object) -> tuple[ExtractedFrame, ...]:
        target_dir = kwargs["output_dir"] if "output_dir" in kwargs else _args[2]
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(b"partial")
        raise ValidationFailure("fixture batch timing mismatch")

    monkeypatch.setattr(frame_extract_module, "_run_batch_group", fail_batch)
    monkeypatch.setattr(frame_extract_module, "extract_evidence_frame", fake_exact)
    monkeypatch.setattr(frame_extract_module, "_prefer_exact_group", lambda _group: False)

    frames = extract_frames(
        source,
        [0, 1_000],
        output_dir,
        batch=True,
        max_workers=2,
    )

    assert [frame.requested_ms for frame in frames] == [0, 1_000]
    assert not list(output_dir.rglob("*.png"))


def test_batch_groups_run_concurrently_but_results_keep_request_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    groups = ((0, 1_000, 2_000), (10_000, 11_000, 12_000))
    active = 0
    peak_active = 0
    lock = Lock()

    def fake_batch(
        _media_path: Path,
        requested_times_ms: tuple[int, ...],
        output_dir: Path,
        *,
        first_frame_number: int,
        **_kwargs: object,
    ) -> tuple[ExtractedFrame, ...]:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        sleep(0.03)
        with lock:
            active -= 1
        return tuple(
            ExtractedFrame(
                frame_id=f"F{first_frame_number + offset:06d}",
                path=output_dir / f"F{first_frame_number + offset:06d}.png",
                requested_ms=requested_ms,
                actual_ms=requested_ms,
                raw_pts=requested_ms,
                time_base="1/1000",
                frame_index=None,
                offset_ms=0,
                timestamp_source="fixture",
                width=640,
                height=360,
            )
            for offset, requested_ms in enumerate(requested_times_ms)
        )

    monkeypatch.setattr(frame_extract_module, "_batch_request_groups", lambda _times: groups)
    monkeypatch.setattr(frame_extract_module, "_prefer_exact_group", lambda _group: False)
    monkeypatch.setattr(frame_extract_module, "_run_batch_group", fake_batch)

    frames = extract_frames(
        source,
        [0, 1_000, 2_000, 10_000, 11_000, 12_000],
        tmp_path / "frames",
        batch=True,
        max_workers=2,
    )

    assert peak_active == 2
    assert [frame.requested_ms for frame in frames] == [0, 1_000, 2_000, 10_000, 11_000, 12_000]


def test_batched_late_group_uses_input_seek_and_keeps_absolute_pts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "generated" / "slide-lecture.mp4"

    def fail_exact_seek(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("late batch unexpectedly fell back to exact seeking")

    monkeypatch.setattr(frame_extract_module, "extract_evidence_frame", fail_exact_seek)
    monkeypatch.setattr(frame_extract_module, "_prefer_exact_group", lambda _group: False)
    frames = extract_frames(
        source,
        [2_000, 2_500, 3_000],
        tmp_path,
        batch=True,
        timeout_seconds=30.0,
    )
    assert [frame.actual_ms for frame in frames] == [2_000, 2_500, 3_000]
    assert all(frame.raw_pts is not None and frame.time_base for frame in frames)


def test_two_frame_group_prefers_exact_seeks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "generated" / "slide-lecture.mp4"

    def fail_batch(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("small group unexpectedly used a guarded batch decode")

    monkeypatch.setattr(frame_extract_module, "_run_batch_group", fail_batch)
    frames = extract_frames(
        source,
        [2_000, 3_000],
        tmp_path,
        batch=True,
        timeout_seconds=30.0,
    )
    assert [frame.actual_ms for frame in frames] == [2_000, 3_000]


def test_sparse_multi_frame_group_prefers_exact_seeks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "long-source.mp4"
    source.write_bytes(b"placeholder")
    calls: list[int] = []

    def fake_exact(
        _media_path: Path,
        requested_ms: int,
        output_dir: Path,
        *,
        frame_id: str,
        **_kwargs: object,
    ) -> ExtractedFrame:
        calls.append(requested_ms)
        output_dir.mkdir(parents=True, exist_ok=True)
        return ExtractedFrame(
            frame_id=frame_id,
            path=output_dir / f"{frame_id}.png",
            requested_ms=requested_ms,
            actual_ms=requested_ms,
            raw_pts=requested_ms,
            time_base="1/1000",
            frame_index=None,
            offset_ms=0,
            timestamp_source="fixture",
            width=640,
            height=360,
        )

    def fail_batch(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("sparse group unexpectedly used a guarded batch decode")

    monkeypatch.setattr(frame_extract_module, "extract_evidence_frame", fake_exact)
    monkeypatch.setattr(frame_extract_module, "_run_batch_group", fail_batch)
    frames = extract_frames(
        source,
        [0, 10_000, 20_000],
        tmp_path / "frames",
        batch=True,
        max_workers=2,
    )
    assert calls == [0, 10_000, 20_000]
    assert [frame.requested_ms for frame in frames] == calls


def test_moderately_sparse_group_prefers_exact_seeks() -> None:
    # Four requests over a minute are still sparse relative to the guarded
    # batch filter. Exact seeks avoid decoding the whole interval.
    assert frame_extract_module._prefer_exact_group((0, 20_000, 40_000, 60_000)) is True
    assert frame_extract_module._prefer_exact_group((0, 10_000, 20_000)) is True


def test_guarded_batch_is_reserved_for_dense_groups() -> None:
    # The guarded filter can only be worthwhile when several requests share
    # a short decode span. Sparse survey points must stay on exact seeks.
    assert frame_extract_module._prefer_exact_group((0, 100, 500)) is False
    assert frame_extract_module._prefer_exact_group((0, 250, 750)) is True


def test_long_span_moderate_density_prefers_exact_seeks() -> None:
    # A long-form survey can be just above the ordinary density crossover
    # while still making one guarded batch traverse many minutes of video.
    # Keep this choice performance-only; measured PTS and pixel validation are
    # unchanged for either extraction route.
    requests = tuple(range(0, 600_001, 15_000))
    assert len(requests) == 41
    assert frame_extract_module._prefer_exact_group(requests) is True
