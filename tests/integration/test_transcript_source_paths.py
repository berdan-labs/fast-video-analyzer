from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from video_script_reconstructor.pipeline import (
    _clip_source_transcript_bounds_to_media,
    plan_input,
    run_pipeline,
)
from video_script_reconstructor.subtitle_parse import ParsedTranscriptSegment
from video_script_reconstructor.whisper_adapter import ModelIndependentASRAdapter

REPOSITORY = Path(__file__).resolve().parents[2]


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("FFmpeg is required for mandatory media integration tests")
    return executable


def _generate(root: Path) -> Path:
    subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts" / "generate_fixtures.py"), str(root)],
        check=True,
    )
    return root


def _canonical(result) -> dict:
    return json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )


def _segment(identifier: str, start: int, end: int, text: str) -> ParsedTranscriptSegment:
    return ParsedTranscriptSegment(
        identifier,
        start,
        end,
        "model-independent-fixture",
        text,
        text,
        confidence=0.95,
    )


def test_source_subtitle_tail_is_clipped_to_media_with_adjustment_record() -> None:
    clipped, adjustments = _clip_source_transcript_bounds_to_media(
        [
            {
                "segment_id": "T000001",
                "start_ms": 900,
                "end_ms": 1_500,
                "timing_provenance": "source_srt",
                "uncertainty_items": [],
            },
            {
                "segment_id": "T000002",
                "start_ms": 900,
                "end_ms": 1_500,
                "timing_provenance": "faster_whisper_word_timestamps",
                "uncertainty_items": [],
            },
        ],
        1_000,
    )

    assert clipped[0]["end_ms"] == 1_000
    assert clipped[0]["timing_provenance"] == "source_srt_clipped_to_media"
    assert clipped[0]["uncertainty_items"]
    assert adjustments == [
        {
            "segment_id": "T000001",
            "start_ms": 900,
            "original_end_ms": 1_500,
            "clipped_end_ms": 1_000,
        }
    ]
    assert clipped[1]["end_ms"] == 1_500


def test_public_pipeline_discovers_and_selects_embedded_srt_and_ass(tmp_path: Path) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    srt = tmp_path / "embedded.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nEmbedded exact sentence.\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nEmbedded value 42.\n",
        encoding="utf-8",
    )
    ass = tmp_path / "embedded.ass"
    ass.write_text(
        "[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Embedded exact sentence.\n"
        "Dialogue: 0,0:00:02.00,0:00:04.00,Default,,0,0,0,,Embedded value 42.\n",
        encoding="utf-8",
    )
    media = tmp_path / "embedded-tracks.mkv"
    subprocess.run(
        [
            _ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(fixtures / "talking-head.mp4"),
            "-i",
            str(srt),
            "-i",
            str(ass),
            "-map",
            "0:v",
            "-map",
            "0:a",
            "-map",
            "1:0",
            "-map",
            "2:0",
            "-c",
            "copy",
            "-metadata:s:s:0",
            "language=eng",
            "-metadata:s:s:0",
            "title=Human authored English",
            "-disposition:s:0",
            "default",
            "-metadata:s:s:1",
            "language=fra",
            "-metadata:s:s:1",
            "title=Human authored French",
            str(media),
        ],
        check=True,
    )
    plan = plan_input(media, output_root=tmp_path / "planned")
    assert not plan["asr_expected"]
    assert (
        sum(str(item).startswith("embedded:stream:") for item in plan["likely_transcript_sources"])
        == 2
    )

    result = run_pipeline(media, output_root=tmp_path / "out")
    assert result.validation is not None and result.validation.valid
    canonical = _canonical(result)
    assert len(canonical["transcript_candidates"]) == 2
    selected = next(
        item
        for item in canonical["transcript_candidates"]
        if item["decision_rationale"].startswith("Selected as")
    )
    assert selected["source_type"] == "embedded_human_subtitle"
    assert selected["language"] == "eng"
    assert "disposition=default" in selected["segments"][0]["source_track"]
    assert (result.project_dir / selected["raw_preservation_path"]).is_file()
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert markdown.count("Embedded exact sentence.") == 1
    assert markdown.count("Embedded value 42.") == 1


def test_public_video_without_subtitle_uses_model_independent_asr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    monkeypatch.setenv("VSR_PARALLEL_VISUAL_SURVEY", "1")
    calls: list[Path] = []
    progress_events: list[dict[str, object]] = []

    def recognize(path: Path, **kwargs: object):
        calls.append(path)
        return [
            _segment("asr-1", 0, 2000, "ASR exact first sentence."),
            _segment("asr-2", 2000, 4000, "ASR exact second sentence with 42."),
        ]

    result = run_pipeline(
        fixtures / "talking-head.mp4",
        output_root=tmp_path / "out",
        asr_adapter=ModelIndependentASRAdapter(recognize),
        progress_callback=lambda payload: progress_events.append(dict(payload)),
    )
    canonical = _canonical(result)
    assert calls == [fixtures / "talking-head.mp4"]
    assert len(canonical["transcript_candidates"]) == 1
    assert canonical["transcript_candidates"][0]["source_type"] == "local_asr"
    assert [item["raw_text"] for item in canonical["transcript_segments"]] == [
        "ASR exact first sentence.",
        "ASR exact second sentence with 42.",
    ]
    assert "model-independent ASR adapter" in canonical["transcript_source_decision"]
    manifest = json.loads(
        (result.project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_versions"]["asr"] == "model-independent"
    assert manifest["model_versions"]["asr_config"] == "faster-whisper-large-v3"
    assert manifest["performance"]["asr"]["event"] == "completed"
    assert manifest["performance"]["visual"]["event"] == "completed"
    assert manifest["performance"]["visual"]["retained_frame_count"] >= 1
    assert {
        "frame_decode_completed",
        "selection_completed",
        "completed",
    } <= {str(item["event"]) for item in manifest["performance"]["visual_events"]}
    assert {
        "survey_parallel_started",
        "survey_parallel_completed",
    } <= {str(item["event"]) for item in manifest["performance"]["visual_events"]}
    assert all(
        float(item["elapsed_seconds"]) >= 0
        for item in manifest["performance"]["visual_events"]
    )
    assert (result.project_dir / ".state" / "asr-progress.json").is_file()
    assert progress_events[-1]["event"] == "completed"


def test_completed_empty_asr_uses_explicit_visual_only_fallback(tmp_path: Path) -> None:
    fixtures = _generate(tmp_path / "fixtures")

    def recognize(_path: Path, **_kwargs: object):
        return []

    result = run_pipeline(
        fixtures / "talking-head.mp4",
        output_root=tmp_path / "out",
        asr_adapter=ModelIndependentASRAdapter(recognize),
        vision_mode="none",
        language="en",
    )
    assert result.status == "review_required"
    assert result.validation is not None and result.validation.valid
    canonical = _canonical(result)
    assert canonical["transcript_segments"] == []
    assert canonical["primary_language"] == "und"
    assert canonical["transcript_candidates"][0]["language"] is None
    assert canonical["state_metadata"]["reconstruction_mode"] == "visual_only_no_speech"
    assert canonical["state_metadata"]["speech_recovery"] == {
        "asr_completed_without_segments": True,
        "asr_had_failure": False,
        "dialogue_inferred": False,
    }
    review = next(
        item
        for item in canonical["review_items"]
        if item["category"] == "no_speech_visual_only_fallback"
    )
    assert review["blocking"] is False
    assert "visual evidence only" in review["problem"]
    assert "No safe transcript" not in canonical["transcript_source_decision"]


def test_blocked_video_resume_retries_transcript_before_cache_return(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")

    class ToggleASRAdapter:
        backend_name = "toggle-fixture-asr"
        supports_full_media_passthrough = True
        is_production = False
        fail = True

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, _path: Path, **_kwargs: object) -> list[object]:
            self.calls += 1
            if type(self).fail:
                raise RuntimeError("temporary ASR prerequisite failure")
            return []

    adapter = ToggleASRAdapter()
    first = run_pipeline(
        fixtures / "talking-head.mp4",
        output_root=tmp_path / "out",
        asr_adapter=adapter,
        vision_mode="none",
    )
    assert first.status == "blocked"

    ToggleASRAdapter.fail = False
    resumed = run_pipeline(
        fixtures / "talking-head.mp4",
        output_root=tmp_path / "out",
        asr_adapter=adapter,
        vision_mode="none",
    )
    assert resumed.status == "review_required"
    assert adapter.calls >= 2
    canonical = _canonical(resumed)
    assert canonical["state_metadata"]["reconstruction_mode"] == "visual_only_no_speech"


def test_interrupted_transcript_stage_is_durable_and_resumable(tmp_path: Path) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    audio = tmp_path / "interruptible.wav"
    subprocess.run(
        [
            _ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(fixtures / "talking-head.mp4"),
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(audio),
        ],
        check=True,
    )
    interrupted = True

    def recognize(path: Path, **kwargs: object):
        del path, kwargs
        if interrupted:
            raise KeyboardInterrupt("simulated user stop")
        return [_segment("resume-1", 0, 4_000, "Resumed exact audio transcript.")]

    adapter = ModelIndependentASRAdapter(
        recognize,
        name="interruptible-fixture",
        cache_identity="interruptible-fixture-v1",
    )
    output_root = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt, match="simulated user stop"):
        run_pipeline(audio, output_root=output_root, asr_adapter=adapter, vision_mode="none")

    project_dir = output_root / "interruptible"
    interrupted_manifest = json.loads(
        (project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert interrupted_manifest["stages"]["identity"]["status"] == "completed"
    assert interrupted_manifest["stages"]["transcript"]["status"] == "failed"
    assert "KeyboardInterrupt" in interrupted_manifest["stages"]["transcript"]["detail"]
    assert not (project_dir / ".state" / "canonical-project.json").exists()

    interrupted = False
    resumed = run_pipeline(
        audio,
        output_root=output_root,
        asr_adapter=adapter,
        vision_mode="none",
        resume=True,
    )
    assert resumed.validation is not None and resumed.validation.valid
    resumed_manifest = json.loads(
        (project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert resumed_manifest["stages"]["transcript"]["status"] == "completed"
    assert _canonical(resumed)["transcript_segments"][0]["raw_text"] == (
        "Resumed exact audio transcript."
    )


def test_interrupted_visual_stage_is_durable_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    import video_script_reconstructor.pipeline as pipeline_module

    original_extract_visual_evidence = pipeline_module._extract_visual_evidence
    interrupted = True

    def extract_visual_evidence(*args: object, **kwargs: object):
        if interrupted:
            raise KeyboardInterrupt("simulated visual stop")
        return original_extract_visual_evidence(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_module, "_extract_visual_evidence", extract_visual_evidence
    )
    output_root = tmp_path / "out"
    subtitles = [fixtures / "talking-head.srt"]
    with pytest.raises(KeyboardInterrupt, match="simulated visual stop"):
        run_pipeline(
            fixtures / "talking-head.mp4",
            output_root=output_root,
            subtitles=subtitles,
            vision_mode="none",
        )

    project_dir = output_root / "talking-head"
    interrupted_manifest = json.loads(
        (project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert interrupted_manifest["stages"]["transcript"]["status"] == "completed"
    assert interrupted_manifest["stages"]["visual_evidence"]["status"] == "failed"
    assert "KeyboardInterrupt" in interrupted_manifest["stages"]["visual_evidence"]["detail"]
    assert not (project_dir / ".state" / "canonical-project.json").exists()

    interrupted = False
    resumed = run_pipeline(
        fixtures / "talking-head.mp4",
        output_root=output_root,
        subtitles=subtitles,
        vision_mode="none",
        resume=True,
    )
    assert resumed.validation is not None and resumed.validation.valid
    resumed_manifest = json.loads(
        (project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert resumed_manifest["stages"]["visual_evidence"]["status"] == "completed"
    assert _canonical(resumed)["frames"]


def test_partial_finalization_marker_forces_rebuild_before_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    audio = tmp_path / "partial-finalization.wav"
    subprocess.run(
        [
            _ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(fixtures / "talking-head.mp4"),
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(audio),
        ],
        check=True,
    )
    adapter = ModelIndependentASRAdapter(
        lambda path, **kwargs: [_segment("partial-1", 0, 4_000, "Partial write transcript.")],
        name="partial-finalization-fixture",
        cache_identity="partial-finalization-fixture-v1",
    )
    output_root = tmp_path / "out"
    import video_script_reconstructor.pipeline as pipeline_module

    original_atomic_write_json = pipeline_module.atomic_write_json
    injected = False

    def fail_after_canonical(path: Path, payload: object, **kwargs: object) -> None:
        nonlocal injected
        if (
            path.name == "run-manifest.json"
            and isinstance(payload, dict)
            and payload.get("run_state") == "finalizing"
            and payload.get("run_cache_key")
            and not injected
        ):
            injected = True
            raise OSError("simulated finalization write failure")
        original_atomic_write_json(path, payload, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(pipeline_module, "atomic_write_json", fail_after_canonical)
        with pytest.raises(OSError, match="simulated finalization write failure"):
            run_pipeline(
                audio,
                output_root=output_root,
                asr_adapter=adapter,
                vision_mode="none",
            )

    assert injected
    project_dir = output_root / "partial-finalization"
    interrupted_manifest = json.loads(
        (project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert interrupted_manifest["run_state"] == "finalizing"
    interrupted_canonical = json.loads(
        (project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    assert interrupted_canonical["manifest"]["run_state"] == "finalizing"
    assert not (project_dir / ".state" / "validation-receipt.json").exists()

    resumed = run_pipeline(
        audio,
        output_root=output_root,
        asr_adapter=adapter,
        vision_mode="none",
        resume=True,
    )
    assert resumed.validation is not None and resumed.validation.valid
    completed_manifest = json.loads(
        (project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert completed_manifest["run_state"] == "completed"
    assert _canonical(resumed)["manifest"]["run_state"] == "completed"
    assert (project_dir / ".state" / "validation-receipt.json").exists()


def test_compare_all_runs_asr_alongside_supplied_subtitles(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    supplied = tmp_path / "supplied.srt"
    supplied.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nSupplied transcript sentence.\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nSupplied value 42.\n",
        encoding="utf-8",
    )
    calls: list[Path] = []

    def recognize(path: Path, **kwargs: object):
        calls.append(path)
        return [
            _segment("asr-1", 0, 2000, "ASR transcript sentence."),
            _segment("asr-2", 2000, 4000, "ASR value 42."),
        ]

    result = run_pipeline(
        fixtures / "talking-head.mp4",
        output_root=tmp_path / "out",
        subtitles=[supplied],
        subtitle_mode="compare-all",
        asr_adapter=ModelIndependentASRAdapter(recognize),
        vision_mode="none",
    )
    assert result.validation is not None and result.validation.valid
    canonical = _canonical(result)
    assert calls == [fixtures / "talking-head.mp4"]
    assert {item["source_type"] for item in canonical["transcript_candidates"]} == {
        "user_subtitle",
        "local_asr",
    }
    selected = next(
        item
        for item in canonical["transcript_candidates"]
        if item["decision_rationale"].startswith("Selected as")
    )
    assert selected["source_type"] == "local_asr"
    assert "independently validating 2 transcript candidate(s)" in canonical[
        "transcript_source_decision"
    ]


def test_provided_only_subtitle_never_invokes_whisper(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    supplied = tmp_path / "supplied.srt"
    supplied.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nTimestamped subtitle only.\n",
        encoding="utf-8",
    )
    calls: list[Path] = []

    def recognize(path: Path, **_kwargs: object):
        calls.append(path)
        return [_segment("should-not-run", 0, 2000, "ASR must not run.")]

    result = run_pipeline(
        fixtures / "talking-head.mp4",
        output_root=tmp_path / "out",
        subtitles=[supplied],
        subtitle_mode="provided-only",
        asr_adapter=ModelIndependentASRAdapter(recognize),
        vision_mode="none",
    )

    assert result.validation is not None and result.validation.valid
    assert calls == []
    canonical = _canonical(result)
    assert canonical["transcript_segments"][0]["raw_text"] == "Timestamped subtitle only."
    manifest = json.loads(
        (result.project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_versions"]["asr"] == "not_used:user_subtitle"


def test_public_audio_only_uses_model_independent_asr(tmp_path: Path) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    audio = tmp_path / "audio-only.wav"
    subprocess.run(
        [
            _ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(fixtures / "talking-head.mp4"),
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(audio),
        ],
        check=True,
    )
    adapter = ModelIndependentASRAdapter(
        lambda path, **kwargs: [_segment("audio-1", 0, 4000, "Audio-only exact script.")]
    )
    result = run_pipeline(audio, output_root=tmp_path / "out", asr_adapter=adapter)
    canonical = _canonical(result)
    assert canonical["visual_source_available"] is False
    assert canonical["frames"] == []
    assert canonical["transcript_segments"][0]["raw_text"] == "Audio-only exact script."
    assert "Audio-only exact script." in result.markdown_path.read_text(encoding="utf-8")


def test_public_damaged_subtitle_invokes_selective_interval_repair(tmp_path: Path) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    damaged = tmp_path / "damaged.srt"
    damaged.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nReliable exact sentence.\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\n"
        "broken words repeated far too densely across this damaged subtitle interval now now now now\n",
        encoding="utf-8",
    )
    clip_calls: list[Path] = []

    def repair_recognizer(path: Path, **kwargs: object):
        clip_calls.append(path)
        return [_segment("repair-1", 750, 2750, "Correct repaired sentence.")]

    result = run_pipeline(
        fixtures / "talking-head.mp4",
        output_root=tmp_path / "out",
        subtitles=[damaged],
        asr_adapter=ModelIndependentASRAdapter(repair_recognizer),
    )
    canonical = _canonical(result)
    assert len(clip_calls) == 1
    assert clip_calls[0].suffix == ".wav"
    assert canonical["transcript_segments"][0]["raw_text"] == "Reliable exact sentence."
    repaired = canonical["transcript_segments"][1]
    assert repaired["raw_text"].startswith("broken words repeated")
    assert repaired["repaired_text"] == "Correct repaired sentence."
    assert canonical["repairs"][0]["action"] == "replace"
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert markdown.count("Reliable exact sentence.") == 1
    assert markdown.count("Correct repaired sentence.") == 1
