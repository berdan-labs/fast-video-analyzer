from __future__ import annotations

import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from video_script_reconstructor.errors import ValidationFailure
from video_script_reconstructor.frame_extract import ExtractedFrame, extract_frames
from video_script_reconstructor.frame_quality import normalized_pixel_hash
from video_script_reconstructor.pipeline import (
    _asr_shared_cache_dir,
    _bounded_asr_progress_event,
    _bounded_packet_frames,
    _bounded_shared_survey_emission_times,
    _bounded_visual_block_points,
    _copy_file_atomic,
    _crop_covers_full_frame,
    _executor_context,
    _guarded_motion_only_enabled,
    _is_guarded_motion_only,
    _link_or_copy_file_atomic,
    _load_or_extract_visual_frames,
    _load_or_run_visual_survey,
    _load_or_run_visual_survey_with_frames,
    _next_review_id,
    _parallel_visual_survey_enabled,
    _precompute_visual_survey,
    _prune_visual_frame_checkpoints,
    _reuse_visual_state,
    _rotate_incomplete_visual_state,
    _scheduler_snapshot,
    _tool_version,
    _visual_crop_workers,
    _visual_frame_workers,
    _visual_shared_cache_dir,
)
from video_script_reconstructor.scene_detection import (
    SurveyCandidate,
    detect_combined_survey_frames,
)


def test_shared_survey_emission_times_bound_long_filter_graphs() -> None:
    requested = tuple(range(0, 100_000, 137))

    bounded = _bounded_shared_survey_emission_times(requested)

    assert len(requested) > 256
    assert len(bounded) == 256
    assert bounded[0] == requested[0]
    assert bounded[-1] == requested[-1]
    assert bounded == _bounded_shared_survey_emission_times(requested)


def test_reuse_visual_state_copies_collections_and_block_fields_without_aliasing() -> None:
    prior = {
        "frames": [{"frame_id": "F000001", "path": "evidence/full/F000001.png"}],
        "evidence_image_metadata": [{"image": {"image_id": "F000001"}}],
        "metadata_revisions": [{"revision_id": "MR000001", "image_id": "F000001"}],
        "visual_events": [{"event_id": "V000001", "evidence_frame_ids": ["F000001"]}],
        "review_items": [{"review_id": "R000001", "frame_ids": ["F000001"]}],
        "ocr_observations": [{"observation_id": "O000001", "frame_id": "F000001"}],
        "script_blocks": [
            {
                "block_id": "B000001",
                "visual_description": "A visible frame.",
                "frame_ids": ["F000001"],
                "uncertainty": ["semantic review pending"],
            }
        ],
    }
    blocks = [{"block_id": "B000001", "visual_description": "new transcript block"}]

    frames, payloads, revisions, events, reviews, ocr = _reuse_visual_state(prior, blocks)

    assert frames == prior["frames"]
    assert payloads == prior["evidence_image_metadata"]
    assert revisions == prior["metadata_revisions"]
    assert events == prior["visual_events"]
    assert reviews == prior["review_items"]
    assert ocr == prior["ocr_observations"]
    assert blocks[0]["visual_description"] == "A visible frame."
    assert blocks[0]["frame_ids"] == ["F000001"]

    frames[0]["path"] = "changed.png"
    blocks[0]["frame_ids"].append("F000002")
    assert prior["frames"][0]["path"] == "evidence/full/F000001.png"
    assert prior["script_blocks"][0]["frame_ids"] == ["F000001"]


def test_tool_version_is_cached_until_executable_signature_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import video_script_reconstructor.pipeline as pipeline_module

    executable = tmp_path / "ffmpeg.exe"
    executable.write_bytes(b"v1")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        return type("Completed", (), {"returncode": 0, "stdout": "ffmpeg version fixture\n", "stderr": ""})()

    monkeypatch.setattr(pipeline_module.subprocess, "run", fake_run)
    first = _tool_version(str(executable))
    second = _tool_version(str(executable))
    executable.write_bytes(b"v2-longer")
    third = _tool_version(str(executable))

    assert first == second == third == "ffmpeg version fixture"
    assert len(calls) == 2


def test_tool_versions_for_cache_key_probes_ffmpeg_tools_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_script_reconstructor.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module.shutil,
        "which",
        lambda name: f"{name}-fixture",
    )
    monkeypatch.setattr(
        pipeline_module,
        "_tool_version",
        lambda binary: f"version:{binary}",
    )

    assert pipeline_module._tool_versions_for_cache_key() == {
        "ffmpeg": "version:ffmpeg-fixture",
        "ffprobe": "version:ffprobe-fixture",
    }


def test_full_frame_crop_detection_rejects_only_parent_sized_regions() -> None:
    assert _crop_covers_full_frame((0, 0, 2240, 1138), 2240, 1138)
    assert not _crop_covers_full_frame((1, 0, 2239, 1138), 2240, 1138)
    assert not _crop_covers_full_frame((0, 0, 2240, 1137), 2240, 1138)


def test_review_id_allocator_handles_sampling_gaps_and_blockers() -> None:
    reviews = [
        {"review_id": "R000001"},
        {"review_id": "R000003"},
        {"review_id": "not-numeric"},
    ]

    assert _next_review_id(reviews) == "R000004"
    reviews.append({"review_id": _next_review_id(reviews)})
    assert _next_review_id(reviews) == "R000005"


def test_scheduler_snapshot_is_bounded_and_records_gpu_cpu_policy() -> None:
    snapshot = _scheduler_snapshot()

    assert {
        "frame_extract_workers",
        "frame_analysis_workers",
        "survey_ffmpeg_threads",
        "ocr_workers",
        "asr_cpu_threads",
        "asr_num_workers",
        "parallel_visual_survey",
        "guarded_motion_dedup",
    } <= set(snapshot)
    assert 1 <= snapshot["frame_extract_workers"] <= 8
    assert 1 <= snapshot["frame_analysis_workers"] <= 8
    assert 1 <= snapshot["survey_ffmpeg_threads"] <= 8
    assert 1 <= snapshot["ocr_workers"] <= 16
    assert 0 <= snapshot["asr_cpu_threads"] <= 32
    assert 1 <= snapshot["asr_num_workers"] <= 8
    assert isinstance(snapshot["parallel_visual_survey"], bool)
    assert isinstance(snapshot["asr_shared_cache"], bool)
    assert isinstance(snapshot["visual_shared_cache"], bool)
    assert isinstance(snapshot["guarded_motion_dedup"], bool)


def test_guarded_motion_dedup_is_opt_in_and_rejects_content_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    difference = SimpleNamespace(
        perceptual_hamming=3,
        changed_pixel_ratio=0.03,
        maximum_region_change=0.12,
        mean_pixel_difference=0.03,
        edge_difference=0.04,
        regions=(SimpleNamespace(xywh=(1700, 700, 220, 300)),),
    )
    monkeypatch.delenv("VSR_GUARDED_MOTION_DEDUP", raising=False)
    assert not _guarded_motion_only_enabled()
    assert not _is_guarded_motion_only(
        difference,
        current_dhash="0000000000000001",
        next_dhash="0000000000000003",
        ocr_stable=True,
        selection_reason="adaptive_frame_difference",
        width=1920,
        height=1140,
        is_boundary=False,
    )

    monkeypatch.setenv("VSR_GUARDED_MOTION_DEDUP", "1")
    assert _guarded_motion_only_enabled()
    assert _is_guarded_motion_only(
        difference,
        current_dhash="0000000000000001",
        next_dhash="0000000000000003",
        ocr_stable=True,
        selection_reason="adaptive_frame_difference",
        width=1920,
        height=1140,
        is_boundary=False,
    )
    assert not _is_guarded_motion_only(
        difference,
        current_dhash="0000000000000001",
        next_dhash="0000000000000003",
        ocr_stable=False,
        selection_reason="adaptive_frame_difference",
        width=1920,
        height=1140,
        is_boundary=False,
    )
    assert not _is_guarded_motion_only(
        difference,
        current_dhash="0000000000000001",
        next_dhash="0000000000000003",
        ocr_stable=True,
        selection_reason="periodic_safety; adaptive_frame_difference",
        width=1920,
        height=1140,
        is_boundary=False,
    )


def test_crop_prepare_workers_are_bounded_and_independently_tunable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VSR_CROP_PREP_WORKERS", "99")
    assert _visual_crop_workers() == 8
    monkeypatch.setenv("VSR_CROP_PREP_WORKERS", "2")
    assert _visual_crop_workers() == 2
    monkeypatch.setenv("VSR_CROP_PREP_WORKERS", "invalid")
    assert 1 <= _visual_crop_workers() <= 8


def test_frame_extract_workers_keep_conservative_default_and_bound_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_script_reconstructor.pipeline as pipeline_module

    monkeypatch.delenv("VSR_FRAME_EXTRACT_WORKERS", raising=False)
    monkeypatch.setattr(pipeline_module.os, "cpu_count", lambda: 16)
    assert _visual_frame_workers() == 4
    monkeypatch.setattr(pipeline_module.os, "cpu_count", lambda: 8)
    assert _visual_frame_workers() == 4
    monkeypatch.setattr(pipeline_module.os, "cpu_count", lambda: 1)
    assert _visual_frame_workers() == 1

    monkeypatch.setenv("VSR_FRAME_EXTRACT_WORKERS", "99")
    assert _visual_frame_workers() == 8


def test_shared_asr_cache_location_is_local_and_opt_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VSR_ASR_SHARED_CACHE_DIR", str(tmp_path / "shared-asr"))
    assert _asr_shared_cache_dir() == (tmp_path / "shared-asr").resolve()
    monkeypatch.setenv("VSR_DISABLE_ASR_SHARED_CACHE", "1")
    assert _asr_shared_cache_dir() is None


def test_shared_visual_cache_location_is_local_and_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VSR_VISUAL_SHARED_CACHE_DIR", str(tmp_path / "shared-visual"))
    assert _visual_shared_cache_dir() == (tmp_path / "shared-visual").resolve()
    monkeypatch.setenv("VSR_DISABLE_VISUAL_SHARED_CACHE", "1")
    assert _visual_shared_cache_dir() is None


def test_executor_context_reuses_pool_without_shutting_it_down() -> None:
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        with _executor_context(
            pool, max_workers=4, thread_name_prefix="unused-local-pool"
        ) as reused:
            assert reused is pool
            assert reused.submit(lambda: 7).result() == 7
        assert pool.submit(lambda: 9).result() == 9
    finally:
        pool.shutdown(wait=True)


def test_visual_block_midpoints_are_temporally_bounded() -> None:
    blocks = [
        {"block_id": f"B{index:06d}", "start_ms": index * 1_000, "end_ms": (index + 1) * 1_000}
        for index in range(120)
    ]

    points = _bounded_visual_block_points(blocks, duration_ms=120_000, spacing_ms=30_000)

    assert points[0][0] == 500
    assert points[-1][0] == 119_500
    assert len(points) <= 6
    interior = points[:-1]
    assert all(
        right[0] - left[0] >= 30_000
        for left, right in zip(interior, interior[1:], strict=False)
    )


def test_visual_block_tail_guard_avoids_unmeasurable_duration_minus_one() -> None:
    points = _bounded_visual_block_points(
        [{"block_id": "B000001", "start_ms": 11_000, "end_ms": 20_000}],
        duration_ms=10_000,
        spacing_ms=1_000,
    )

    assert points == [(9_750, "B000001")]


def test_persisted_asr_progress_keeps_only_a_bounded_timing_window() -> None:
    timings = [{"chunk_index": index} for index in range(40)]
    payload = {"event": "chunk_completed", "chunk_timings": timings}

    bounded = _bounded_asr_progress_event(payload)

    assert len(timings) == 40
    assert len(bounded["chunk_timings"]) == 32
    assert bounded["chunk_timings"][0] == {"chunk_index": 8}
    assert bounded["chunk_timings"][-1] == {"chunk_index": 39}
    assert bounded["chunk_timings_omitted"] == 8


def test_packet_window_bounds_complete_span_and_keeps_nearest_focus() -> None:
    frames = [
        {"frame_id": "F000001", "actual_ms": 0},
        {"frame_id": "F000002", "actual_ms": 10_000},
        {"frame_id": "F000003", "actual_ms": 20_000},
    ]

    bounded = _bounded_packet_frames(frames, focus_ms=10_000, max_span_ms=15_000)

    assert [frame["frame_id"] for frame in bounded] == ["F000001", "F000002"]
    assert max(frame["actual_ms"] for frame in bounded) - min(
        frame["actual_ms"] for frame in bounded
    ) <= 15_000


def test_packet_window_deduplicates_frame_ids_after_crop_enrichment() -> None:
    frames = [
        {"frame_id": "F000001", "actual_ms": 3_320},
        {"frame_id": "F000001-C01", "actual_ms": 3_320},
        {"frame_id": "F000002", "actual_ms": 10_000},
        {"frame_id": "F000003", "actual_ms": 29_000},
    ]

    bounded = _bounded_packet_frames(frames, focus_ms=10_000)

    assert [frame["frame_id"] for frame in bounded] == [
        "F000001",
        "F000001-C01",
        "F000002",
    ]


def test_incomplete_visual_state_is_rotated_out_of_canonical_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "evidence" / "full").mkdir(parents=True)
    (project / "evidence" / "crops").mkdir(parents=True)
    (project / ".state" / "vision" / "packets").mkdir(parents=True)
    (project / "evidence" / "full" / "F000001.png").write_bytes(b"frame")
    (project / ".state" / "vision" / "packets" / "V000001.json").write_text("{}")

    _rotate_incomplete_visual_state(project)

    assert not list((project / "evidence" / "full").glob("*.png"))
    assert not list((project / ".state" / "vision" / "packets").glob("*.json"))
    history = list((project / ".state" / "visual-history").rglob("*"))
    assert any(item.name == "F000001.png" for item in history)
    assert any(item.name == "V000001.json" for item in history)


def test_atomic_checkpoint_copy_is_safe_across_windows_workers(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(range(256)) * 4096)
    destination = tmp_path / "copies"
    destination.mkdir()

    def copy(index: int) -> tuple[str, int]:
        return _copy_file_atomic(source, destination / f"copy-{index:02d}.bin")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(copy, range(32)))

    assert len(results) == 32
    assert len({digest for digest, _size in results}) == 1
    assert {size for _digest, size in results} == {source.stat().st_size}
    assert len(list(destination.glob("copy-*.bin"))) == 32


def test_checkpoint_materialization_hardlinks_when_available_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable-checkpoint")
    linked = tmp_path / "linked.bin"

    digest, size = _link_or_copy_file_atomic(source, linked)

    assert digest
    assert size == source.stat().st_size
    assert linked.read_bytes() == source.read_bytes()
    assert linked.stat().st_ino == source.stat().st_ino

    fallback = tmp_path / "fallback.bin"
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.os.link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no hardlinks")),
    )
    _link_or_copy_file_atomic(source, fallback)
    assert fallback.read_bytes() == source.read_bytes()
    assert fallback.stat().st_ino != source.stat().st_ino


def test_visual_survey_cache_reuses_only_matching_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VSR_DISABLE_VISUAL_SHARED_CACHE", "1")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"deterministic source")
    project = tmp_path / "project"
    calls: list[tuple[int, ...]] = []

    def fake_survey(
        *_args: object, speech_reference_times_ms: Sequence[int] = (), **_kwargs: object
    ) -> tuple[SurveyCandidate, ...]:
        references = tuple(speech_reference_times_ms)
        calls.append(references)
        return (
            SurveyCandidate(
                candidate_id="VC000001",
                requested_ms=1_000,
                actual_ms=1_002,
                raw_pts=42,
                time_base="1/1000",
                reasons=("periodic_safety_sample",),
                score=0.5,
                timestamp_source="decoded-survey",
            ),
        )

    monkeypatch.setattr("video_script_reconstructor.pipeline._tool_version", lambda _path: "ffmpeg-test")
    monkeypatch.setattr(
        "video_script_reconstructor.scene_detection.survey_video_candidates", fake_survey
    )

    first = _load_or_run_visual_survey(
        source,
        project,
        duration_ms=10_000,
        interval_seconds=30.0,
        strict=True,
        scene_detection=True,
        adaptive_detection=True,
        speech_reference_times_ms=(0, 5_000),
    )
    second = _load_or_run_visual_survey(
        source,
        project,
        duration_ms=10_000,
        interval_seconds=30.0,
        strict=True,
        scene_detection=True,
        adaptive_detection=True,
        speech_reference_times_ms=(0, 5_000),
    )
    changed_context = _load_or_run_visual_survey(
        source,
        project,
        duration_ms=10_000,
        interval_seconds=30.0,
        strict=True,
        scene_detection=True,
        adaptive_detection=True,
        speech_reference_times_ms=(0, 6_000),
    )

    assert first == second
    assert first != changed_context
    assert calls == [()]
    assert (project / ".state" / "checkpoints" / "visual-survey.json").exists()
    assert (project / ".state" / "checkpoints" / "visual-survey-structural.json").exists()


def test_parallel_visual_survey_precompute_has_no_transcript_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"deterministic source")
    project = tmp_path / "project"
    seen: dict[str, object] = {}

    def fake_survey(
        _source: Path,
        _project: Path,
        **kwargs: object,
    ) -> tuple[tuple[SurveyCandidate, ...], tuple[object, ...]]:
        seen["speech_reference_times_ms"] = kwargs["speech_reference_times_ms"]
        seen["periodic_times_ms"] = kwargs["periodic_times_ms"]
        return (
            (
                SurveyCandidate(
                    candidate_id="VC000001",
                    requested_ms=0,
                    actual_ms=0,
                    raw_pts=0,
                    time_base="1/1000",
                    reasons=("periodic_safety",),
                    score=0.25,
                    timestamp_source="requested-candidate",
                ),
            ),
            (),
        )

    monkeypatch.setattr(
        "video_script_reconstructor.pipeline._load_or_run_visual_survey_with_frames",
        fake_survey,
    )
    result = _precompute_visual_survey(
        source,
        project,
        duration_ms=10_000,
        interval_seconds=30.0,
        strict=True,
        scene_detection=True,
        adaptive_detection=True,
        source_sha256="source-digest",
    )

    assert seen["speech_reference_times_ms"] == ()
    assert seen["periodic_times_ms"] == (0,)
    assert result.candidates[0].requested_ms == 0
    assert result.shared_frame_dir.is_dir()
    shutil.rmtree(result.shared_frame_dir)


def test_context_change_reuses_structural_survey_with_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VSR_DISABLE_VISUAL_SHARED_CACHE", "1")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"deterministic source")
    project = tmp_path / "project"
    calls = 0
    canonical_candidates = (
        SurveyCandidate(
            candidate_id="VC000001",
            requested_ms=1_000,
            actual_ms=1_002,
            raw_pts=42,
            time_base="1/1000",
            reasons=("scene_cut",),
            score=0.9,
            timestamp_source="decoded-survey",
        ),
    )

    def fake_combined(*_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        return (
            (
                SurveyCandidate(
                    candidate_id="VC000001",
                    requested_ms=1_000,
                    actual_ms=1_002,
                    raw_pts=42,
                    time_base="1/1000",
                    reasons=("scene_cut",),
                    score=0.9,
                    timestamp_source="decoded-survey",
                ),
            ),
            (),
            (),
        )

    monkeypatch.setattr(
        "video_script_reconstructor.scene_detection.detect_combined_survey_frames",
        fake_combined,
    )
    monkeypatch.setattr(
        "video_script_reconstructor.scene_detection.survey_video_candidates",
        lambda *_args, **_kwargs: canonical_candidates,
    )
    first, first_frames = _load_or_run_visual_survey_with_frames(
        source,
        project,
        duration_ms=10_000,
        interval_seconds=30.0,
        strict=True,
        scene_detection=True,
        adaptive_detection=True,
        speech_reference_times_ms=(0, 5_000),
        periodic_times_ms=(0,),
        frame_output_dir=project / "survey-1",
    )
    changed, changed_frames = _load_or_run_visual_survey_with_frames(
        source,
        project,
        duration_ms=10_000,
        interval_seconds=30.0,
        strict=True,
        scene_detection=True,
        adaptive_detection=True,
        speech_reference_times_ms=(0, 6_000),
        periodic_times_ms=(0,),
        frame_output_dir=project / "survey-2",
    )

    assert calls == 1
    assert first_frames == changed_frames == ()
    assert first != changed
    assert [item.requested_ms for item in changed if "deictic_speech_reference" in item.reasons] == [
        0,
        6_000,
    ]


def test_shared_survey_retries_filter_allocation_with_smaller_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"deterministic source")
    project = tmp_path / "project"
    periodic_times = tuple(range(0, 160 * 30_000, 30_000))
    attempted_counts: list[int] = []

    def fake_combined(_source: Path, _output: Path, requested: Sequence[int], **_kwargs: object):
        attempted_counts.append(len(requested))
        if len(attempted_counts) == 1:
            raise ValidationFailure(
                "FFmpeg combined survey frame emission failed: "
                "Error initializing filters: Cannot allocate memory"
            )
        return ((), (), ())

    monkeypatch.setenv("VSR_DISABLE_VISUAL_SHARED_CACHE", "1")
    monkeypatch.setattr(
        "video_script_reconstructor.scene_detection.detect_combined_survey_frames",
        fake_combined,
    )
    monkeypatch.setattr(
        "video_script_reconstructor.scene_detection.survey_video_candidates",
        lambda *_args, **_kwargs: (
            SurveyCandidate(
                candidate_id="VC000001",
                requested_ms=0,
                actual_ms=None,
                raw_pts=None,
                time_base=None,
                reasons=("periodic_safety",),
                score=0.25,
                timestamp_source="requested-candidate",
            ),
        ),
    )
    candidates, shared_frames = _load_or_run_visual_survey_with_frames(
        source,
        project,
        duration_ms=4_800_000,
        interval_seconds=30.0,
        strict=True,
        scene_detection=True,
        adaptive_detection=True,
        speech_reference_times_ms=(),
        periodic_times_ms=periodic_times,
        frame_output_dir=project / "survey",
    )

    assert attempted_counts == [160, 128]
    assert candidates
    assert shared_frames == ()


def test_structural_survey_receipt_reuses_across_projects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"shared survey source")
    shared = tmp_path / "shared-visual"
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    calls = 0
    canonical_candidates = (
        SurveyCandidate(
            candidate_id="VC000001",
            requested_ms=1_000,
            actual_ms=1_002,
            raw_pts=42,
            time_base="1/1000",
            reasons=("scene_cut",),
            score=0.9,
            timestamp_source="decoded-survey",
        ),
    )

    def fake_combined(*_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        return (
            (
                SurveyCandidate(
                    candidate_id="VC000001",
                    requested_ms=1_000,
                    actual_ms=1_002,
                    raw_pts=42,
                    time_base="1/1000",
                    reasons=("scene_cut",),
                    score=0.9,
                    timestamp_source="decoded-survey",
                ),
            ),
            (),
            (),
        )

    monkeypatch.setenv("VSR_VISUAL_SHARED_CACHE_DIR", str(shared))
    monkeypatch.setattr("video_script_reconstructor.pipeline._tool_version", lambda _path: "ffmpeg-test")
    monkeypatch.setattr(
        "video_script_reconstructor.scene_detection.detect_combined_survey_frames",
        fake_combined,
    )
    monkeypatch.setattr(
        "video_script_reconstructor.scene_detection.survey_video_candidates",
        lambda *_args, **_kwargs: canonical_candidates,
    )
    first, _ = _load_or_run_visual_survey_with_frames(
        source,
        project_a,
        duration_ms=10_000,
        interval_seconds=30.0,
        strict=True,
        scene_detection=True,
        adaptive_detection=True,
        speech_reference_times_ms=(0,),
        periodic_times_ms=(0,),
        frame_output_dir=project_a / "survey",
    )
    second, _ = _load_or_run_visual_survey_with_frames(
        source,
        project_b,
        duration_ms=10_000,
        interval_seconds=30.0,
        strict=True,
        scene_detection=True,
        adaptive_detection=True,
        speech_reference_times_ms=(0,),
        periodic_times_ms=(0,),
        frame_output_dir=project_b / "survey",
    )

    assert calls == 1
    assert first == second
    assert list((shared / "surveys").glob("*.json"))


def test_parallel_visual_survey_is_explicitly_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VSR_PARALLEL_VISUAL_SURVEY", raising=False)
    assert _parallel_visual_survey_enabled() is False
    monkeypatch.setenv("VSR_PARALLEL_VISUAL_SURVEY", "1")
    assert _parallel_visual_survey_enabled() is True
    monkeypatch.setenv("VSR_PARALLEL_VISUAL_SURVEY", "0")
    assert _parallel_visual_survey_enabled() is False


def test_parallel_visual_survey_auto_requires_long_gpu_capable_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VSR_PARALLEL_VISUAL_SURVEY", raising=False)
    monkeypatch.setattr("video_script_reconstructor.pipeline.os.cpu_count", lambda: 16)
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.shutil.which",
        lambda name: "C:/Windows/System32/nvidia-smi.exe" if name == "nvidia-smi" else None,
    )
    assert _parallel_visual_survey_enabled(duration_ms=299_999) is False
    assert _parallel_visual_survey_enabled(duration_ms=300_000) is True
    assert _parallel_visual_survey_enabled(duration_ms=300_000, automatic_adapter=False) is False


def test_visual_frame_checkpoint_reuses_exact_bytes_and_repairs_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"deterministic source")
    project = tmp_path / "project"
    calls = 0

    def fake_extract(
        _source: Path,
        requested_times_ms: list[int],
        output_dir: Path,
        *,
        max_workers: int,
        batch: bool,
        timeout_seconds: float,
    ) -> tuple[ExtractedFrame, ...]:
        nonlocal calls
        assert max_workers == 2
        assert batch is True
        assert timeout_seconds == 600.0
        calls += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[ExtractedFrame] = []
        for index, requested_ms in enumerate(requested_times_ms, 1):
            path = output_dir / f"F{index:06d}__00h00m00s{requested_ms:03d}__full.png"
            path.write_bytes(f"raw-{requested_ms}".encode())
            frames.append(
                ExtractedFrame(
                    frame_id=f"F{index:06d}",
                    path=path.resolve(),
                    requested_ms=requested_ms,
                    actual_ms=requested_ms + 2,
                    raw_pts=requested_ms + 20,
                    time_base="1/1000",
                    frame_index=None,
                    offset_ms=2,
                    timestamp_source="decoded-survey",
                    width=640,
                    height=360,
                )
            )
        return tuple(frames)

    monkeypatch.setattr("video_script_reconstructor.pipeline._tool_version", lambda _path: "ffmpeg-test")
    monkeypatch.setattr("video_script_reconstructor.frame_extract.extract_frames", fake_extract)

    first = _load_or_extract_visual_frames(
        source,
        project,
        project / "evidence" / "full",
        (0, 1_000),
        duration_ms=2_000,
        max_workers=2,
    )
    second = _load_or_extract_visual_frames(
        source,
        project,
        project / "rebuild" / "evidence" / "full",
        (0, 1_000),
        duration_ms=2_000,
        max_workers=2,
    )

    assert calls == 1
    assert [(frame.frame_id, frame.actual_ms, frame.raw_pts) for frame in first] == [
        (frame.frame_id, frame.actual_ms, frame.raw_pts) for frame in second
    ]
    assert [frame.path.read_bytes() for frame in first] == [
        frame.path.read_bytes() for frame in second
    ]

    cached_png = next((project / ".state" / "checkpoints" / "visual-frames").rglob("*.png"))
    # Warm restores may hardlink immutable raw bytes instead of copying the
    # entire PNG. The later metadata phase atomically replaces evidence paths,
    # so sharing this inode does not mutate the checkpoint receipt.
    matching_restore = next(frame.path for frame in second if frame.path.name == cached_png.name)
    assert matching_restore.stat().st_ino == cached_png.stat().st_ino
    cached_png.write_bytes(b"corrupted")
    third = _load_or_extract_visual_frames(
        source,
        project,
        project / "retry" / "evidence" / "full",
        (0, 1_000),
        duration_ms=2_000,
        max_workers=2,
    )
    assert calls == 2
    assert [frame.path.read_bytes() for frame in third] == [frame.path.read_bytes() for frame in first]

    monkeypatch.setenv("VSR_VISUAL_FRAME_CACHE_MAX_BYTES", "1")
    limited_project = tmp_path / "limited"
    _load_or_extract_visual_frames(
        source,
        limited_project,
        limited_project / "evidence" / "full",
        (0, 1_000),
        duration_ms=2_000,
        max_workers=2,
    )
    _load_or_extract_visual_frames(
        source,
        limited_project,
        limited_project / "retry" / "evidence" / "full",
        (0, 1_000),
        duration_ms=2_000,
        max_workers=2,
    )
    assert calls == 4
    assert not (limited_project / ".state" / "checkpoints" / "visual-frames").exists()


def test_visual_frame_checkpoint_reuses_exact_bytes_from_shared_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"cross-project visual source")
    shared = tmp_path / "shared-visual"
    calls = 0

    def fake_extract(
        _source: Path,
        requested_times_ms: list[int],
        output_dir: Path,
        *,
        max_workers: int,
        batch: bool,
        timeout_seconds: float,
    ) -> tuple[ExtractedFrame, ...]:
        nonlocal calls
        assert max_workers == 2
        assert batch is True
        assert timeout_seconds == 600.0
        calls += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[ExtractedFrame] = []
        for index, requested_ms in enumerate(requested_times_ms, 1):
            path = output_dir / f"F{index:06d}__{requested_ms:06d}__full.png"
            path.write_bytes(f"visual-{requested_ms}".encode())
            frames.append(
                ExtractedFrame(
                    frame_id=f"F{index:06d}",
                    path=path.resolve(),
                    requested_ms=requested_ms,
                    actual_ms=requested_ms + 4,
                    raw_pts=requested_ms + 40,
                    time_base="1/1000",
                    frame_index=None,
                    offset_ms=4,
                    timestamp_source="guarded-exact",
                    width=640,
                    height=360,
                )
            )
        return tuple(frames)

    monkeypatch.setattr("video_script_reconstructor.pipeline._tool_version", lambda _path: "ffmpeg-test")
    monkeypatch.setattr("video_script_reconstructor.frame_extract.extract_frames", fake_extract)
    first = _load_or_extract_visual_frames(
        source,
        tmp_path / "project-a",
        tmp_path / "project-a" / "evidence",
        (0, 1_000),
        duration_ms=2_000,
        max_workers=2,
        shared_cache_dir=shared,
    )
    second = _load_or_extract_visual_frames(
        source,
        tmp_path / "project-b",
        tmp_path / "project-b" / "evidence",
        (0, 1_000),
        duration_ms=2_000,
        max_workers=2,
        shared_cache_dir=shared,
    )

    assert calls == 1
    assert [(frame.actual_ms, frame.raw_pts) for frame in second] == [
        (frame.actual_ms, frame.raw_pts) for frame in first
    ]
    assert [frame.path.read_bytes() for frame in second] == [
        frame.path.read_bytes() for frame in first
    ]
    assert list((shared / "frames").rglob("manifest.json"))


def test_visual_frame_shared_cache_keeps_bounded_partial_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bounded shared cache still works when the local schedule is too large."""

    source = tmp_path / "source.mp4"
    source.write_bytes(b"bounded cross-project visual source")
    shared = tmp_path / "shared-visual"
    # Four equal 16-byte PNG stand-ins; retain exactly one frame in the shared
    # acceleration budget while the project-local raw-frame cache rejects the
    # complete schedule. The shared receipt must still be written and reused.
    monkeypatch.setenv("VSR_VISUAL_SHARED_CACHE_MAX_BYTES", "16")
    monkeypatch.setenv("VSR_VISUAL_FRAME_CACHE_MAX_BYTES", "1")
    calls: list[tuple[int, ...]] = []

    def fake_extract(
        _source: Path,
        requested_times_ms: list[int],
        output_dir: Path,
        *,
        max_workers: int,
        batch: bool,
        timeout_seconds: float,
    ) -> tuple[ExtractedFrame, ...]:
        assert max_workers == 2
        assert batch is True
        assert timeout_seconds == 600.0
        calls.append(tuple(requested_times_ms))
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[ExtractedFrame] = []
        for index, requested_ms in enumerate(requested_times_ms, 1):
            path = output_dir / f"F{index:06d}__{requested_ms:06d}__full.png"
            path.write_bytes(b"0123456789abcdef")
            frames.append(
                ExtractedFrame(
                    frame_id=f"F{index:06d}",
                    path=path.resolve(),
                    requested_ms=requested_ms,
                    actual_ms=requested_ms + 4,
                    raw_pts=requested_ms + 40,
                    time_base="1/1000",
                    frame_index=None,
                    offset_ms=4,
                    timestamp_source="guarded-exact",
                    width=640,
                    height=360,
                )
            )
        return tuple(frames)

    monkeypatch.setattr("video_script_reconstructor.pipeline._tool_version", lambda _path: "ffmpeg-test")
    monkeypatch.setattr("video_script_reconstructor.frame_extract.extract_frames", fake_extract)
    requested = (0, 1_000, 2_000, 3_000)
    first = _load_or_extract_visual_frames(
        source,
        tmp_path / "project-a",
        tmp_path / "project-a" / "evidence",
        requested,
        duration_ms=4_000,
        max_workers=2,
        shared_cache_dir=shared,
    )
    second = _load_or_extract_visual_frames(
        source,
        tmp_path / "project-b",
        tmp_path / "project-b" / "evidence",
        requested,
        duration_ms=4_000,
        max_workers=2,
        shared_cache_dir=shared,
    )

    # The first schedule is fully extracted. The second reuses the one shared
    # frame and extracts only the three missing requests even though neither
    # project has a complete local raw-frame checkpoint.
    assert calls == [requested, requested[1:]]
    assert [frame.path.read_bytes() for frame in second] == [
        frame.path.read_bytes() for frame in first
    ]
    manifests = list((shared / "frames").rglob("manifest.json"))
    assert len(manifests) == 1
    payload = manifests[0].read_text(encoding="utf-8")
    assert '"partial":true' in payload.replace(" ", "")
    assert len(list((shared / "frames").rglob("*.png"))) == 1
    assert not (tmp_path / "project-a" / ".state" / "checkpoints" / "visual-frames").exists()
    assert not (tmp_path / "project-b" / ".state" / "checkpoints" / "visual-frames").exists()
    assert source.read_bytes() == b"bounded cross-project visual source"


def test_visual_frame_checkpoint_reuses_overlapping_requests_after_schedule_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"schedule-independent source")
    project = tmp_path / "project"
    calls: list[tuple[int, ...]] = []

    def fake_extract(
        _source: Path,
        requested_times_ms: list[int],
        output_dir: Path,
        *,
        max_workers: int,
        batch: bool,
        timeout_seconds: float,
    ) -> tuple[ExtractedFrame, ...]:
        assert max_workers == 2
        assert batch is True
        assert timeout_seconds == 600.0
        calls.append(tuple(requested_times_ms))
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[ExtractedFrame] = []
        for index, requested_ms in enumerate(requested_times_ms, 1):
            path = output_dir / f"F{index:06d}__00h00m00s{requested_ms:03d}__full.png"
            path.write_bytes(f"raw-{requested_ms}".encode())
            frames.append(
                ExtractedFrame(
                    frame_id=f"F{index:06d}",
                    path=path.resolve(),
                    requested_ms=requested_ms,
                    actual_ms=requested_ms + 2,
                    raw_pts=requested_ms + 20,
                    time_base="1/1000",
                    frame_index=None,
                    offset_ms=2,
                    timestamp_source="guarded-exact",
                    width=640,
                    height=360,
                )
            )
        return tuple(frames)

    monkeypatch.setattr("video_script_reconstructor.pipeline._tool_version", lambda _path: "ffmpeg-test")
    monkeypatch.setattr("video_script_reconstructor.frame_extract.extract_frames", fake_extract)

    first = _load_or_extract_visual_frames(
        source,
        project,
        project / "evidence" / "full",
        (0, 1_000),
        duration_ms=2_000,
        max_workers=2,
    )
    changed = _load_or_extract_visual_frames(
        source,
        project,
        project / "changed" / "evidence" / "full",
        (0, 1_000, 1_500),
        duration_ms=2_000,
        max_workers=2,
    )

    assert calls == [(0, 1_000), (1_500,)]
    assert [frame.requested_ms for frame in changed] == [0, 1_000, 1_500]
    assert [frame.actual_ms for frame in changed] == [2, 1_002, 1_502]
    assert [frame.path.read_bytes() for frame in changed[:2]] == [
        frame.path.read_bytes() for frame in first
    ]


def test_visual_frame_checkpoint_key_ignores_scheduler_worker_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Changing concurrency must not force identical frames through FFmpeg again."""

    source = tmp_path / "source.mp4"
    source.write_bytes(b"scheduler-independent source")
    project = tmp_path / "project"
    calls: list[int] = []

    def fake_extract(
        _source: Path,
        requested_times_ms: list[int],
        output_dir: Path,
        *,
        max_workers: int,
        batch: bool,
        timeout_seconds: float,
    ) -> tuple[ExtractedFrame, ...]:
        assert batch is True
        assert timeout_seconds == 600.0
        calls.append(max_workers)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[ExtractedFrame] = []
        for index, requested_ms in enumerate(requested_times_ms, 1):
            path = output_dir / f"F{index:06d}__00h00m00s{requested_ms:03d}__full.png"
            path.write_bytes(f"raw-{requested_ms}".encode())
            frames.append(
                ExtractedFrame(
                    frame_id=f"F{index:06d}",
                    path=path.resolve(),
                    requested_ms=requested_ms,
                    actual_ms=requested_ms + 2,
                    raw_pts=requested_ms + 20,
                    time_base="1/1000",
                    frame_index=None,
                    offset_ms=2,
                    timestamp_source="guarded-exact",
                    width=640,
                    height=360,
                )
            )
        return tuple(frames)

    monkeypatch.setattr("video_script_reconstructor.pipeline._tool_version", lambda _path: "ffmpeg-test")
    monkeypatch.setattr("video_script_reconstructor.frame_extract.extract_frames", fake_extract)

    first = _load_or_extract_visual_frames(
        source,
        project,
        project / "evidence" / "full",
        (0, 1_000),
        duration_ms=2_000,
        max_workers=1,
    )
    resumed = _load_or_extract_visual_frames(
        source,
        project,
        project / "resumed" / "evidence" / "full",
        (0, 1_000),
        duration_ms=2_000,
        max_workers=4,
    )

    assert calls == [1]
    assert [(frame.actual_ms, frame.raw_pts) for frame in resumed] == [
        (frame.actual_ms, frame.raw_pts) for frame in first
    ]
    assert [frame.path.read_bytes() for frame in resumed] == [
        frame.path.read_bytes() for frame in first
    ]


def test_visual_prior_schedule_reuses_matching_shared_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"shared prior schedule source")
    shared = tmp_path / "shared"
    first_project = tmp_path / "first"
    second_project = tmp_path / "second"
    calls: list[tuple[int, ...]] = []

    def fake_extract(
        _source: Path,
        requested_times_ms: list[int],
        output_dir: Path,
        *,
        max_workers: int,
        batch: bool,
        timeout_seconds: float,
    ) -> tuple[ExtractedFrame, ...]:
        del max_workers, batch, timeout_seconds
        calls.append(tuple(requested_times_ms))
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[ExtractedFrame] = []
        for index, requested_ms in enumerate(requested_times_ms, 1):
            path = output_dir / f"F{index:06d}__00h00m00s{requested_ms:03d}__full.png"
            path.write_bytes(f"raw-{requested_ms}".encode())
            frames.append(
                ExtractedFrame(
                    frame_id=f"F{index:06d}",
                    path=path.resolve(),
                    requested_ms=requested_ms,
                    actual_ms=requested_ms + 2,
                    raw_pts=requested_ms + 20,
                    time_base="1/1000",
                    frame_index=None,
                    offset_ms=2,
                    timestamp_source="guarded-exact",
                    width=640,
                    height=360,
                )
            )
        return tuple(frames)

    monkeypatch.setenv("VSR_VISUAL_SHARED_CACHE_DIR", str(shared))
    monkeypatch.setattr("video_script_reconstructor.frame_extract.extract_frames", fake_extract)
    first = _load_or_extract_visual_frames(
        source,
        first_project,
        first_project / "evidence" / "full",
        (0, 1_000),
        duration_ms=2_000,
        max_workers=2,
        shared_cache_dir=shared,
    )
    second = _load_or_extract_visual_frames(
        source,
        second_project,
        second_project / "evidence" / "full",
        (0, 1_500),
        duration_ms=2_000,
        max_workers=2,
        shared_cache_dir=shared,
    )

    assert calls == [(0, 1_000), (1_500,)]
    assert [frame.requested_ms for frame in first] == [0, 1_000]
    assert [frame.requested_ms for frame in second] == [0, 1_500]
    assert (second_project / ".state" / "checkpoints" / "visual-frames").exists()
    # The 0-ms frame is byte-identical across the two schedule receipts.  It
    # is acceleration-only state, so the shared cache should retain two paths
    # to one inode instead of writing a second PNG.  The evidence paths remain
    # independent copies and therefore cannot be mutated through the cache.
    shared_pngs = sorted((shared / "frames").rglob("*.png"))
    assert len(shared_pngs) == 4
    assert len({path.stat().st_ino for path in shared_pngs}) == 3
    first_evidence = first[0].path
    second_evidence = second[0].path
    assert first_evidence.stat().st_ino != second_evidence.stat().st_ino


def test_shared_periodic_frames_mix_with_exact_fallback_without_pixel_drift(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "generated" / "slide-lecture.mp4"
    shared_dir = tmp_path / "shared"
    _hard, _adaptive, shared = detect_combined_survey_frames(source, shared_dir, (0, 2_000))
    project = tmp_path / "mixed-project"
    mixed = _load_or_extract_visual_frames(
        source,
        project,
        project / "evidence" / "full",
        (0, 1_851, 2_000),
        duration_ms=4_000,
        max_workers=2,
        shared_frames=shared,
    )

    assert [frame.requested_ms for frame in mixed] == [0, 1_851, 2_000]
    assert [frame.actual_ms for frame in mixed] == [0, 1_900, 2_000]
    exact = extract_frames(
        source,
        [0, 1_851, 2_000],
        tmp_path / "exact",
        max_workers=2,
        batch=True,
        timeout_seconds=30.0,
    )
    assert [normalized_pixel_hash(frame.path) for frame in mixed] == [
        normalized_pixel_hash(frame.path) for frame in exact
    ]
    # The shared branch is measured, not guessed: its metadata retains the
    # source timing and exact-safe PNG dimensions just like normal extraction.
    assert all(frame.width == 640 and frame.height == 360 for frame in mixed)


def test_visual_frame_checkpoint_budget_prunes_old_schedules_without_touching_current(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints" / "visual-frames"
    old_a = root / "old-a"
    old_b = root / "old-b"
    current = root / "current"
    for directory in (old_a, old_b, current):
        directory.mkdir(parents=True)
        (directory / "frame.png").write_bytes(b"12345678")
        (directory / "manifest.json").write_text(
            '{"schema_version":"1.0","cache_bytes":8}', encoding="utf-8"
        )

    _prune_visual_frame_checkpoints(
        root,
        current_cache_dir=current,
        cache_limit=16,
    )

    assert current.is_dir()
    assert old_b.is_dir()
    assert not old_a.exists()
