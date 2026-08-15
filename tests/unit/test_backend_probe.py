from __future__ import annotations

from pathlib import Path

import pytest

import video_script_reconstructor.pipeline as pipeline_module
from video_script_reconstructor.backend_probe import probe_optional_backends
from video_script_reconstructor.pipeline import doctor_report


def test_optional_backend_probe_is_read_only_and_reports_exact_capabilities(
    tmp_path: Path,
) -> None:
    report = probe_optional_backends(model_root=tmp_path / "models")
    assert set(report) == {
        "primary_asr",
        "large_v3",
        "tesseract",
        "primary_ocr",
        "neural_diarization",
        "forced_alignment",
        "semantic_vision",
    }
    assert report["large_v3"]["offline_ready"] is False
    assert report["primary_asr"]["offline_ready"] is False
    assert report["primary_ocr"]["offline_ready"] is False
    assert report["neural_diarization"]["offline_ready"] is False
    assert report["large_v3"]["model"]["manifest_present"] is False
    assert report["semantic_vision"]["status"] in {"available", "optional"}


def test_doctor_summaries_name_new_primary_local_backends(tmp_path: Path) -> None:
    report = doctor_report(output_path=tmp_path, offline=True)
    checks = report["checks"]

    assert "faster-whisper large-v3" in checks["speech_recognition"]["value"]["primary"]
    assert checks["speech_recognition"]["value"]["cpu_fallback"] == (
        "enabled when CUDA/cuBLAS cannot load"
    )
    assert "preferred for Filipino" in checks["faster_whisper"]["value"]
    assert checks["ocr"]["value"]["primary"].startswith("PP-OCRv5")
    assert checks["diarization"]["value"]["primary"].startswith("neutral speaker labels")
    assert checks["vision_provider"]["value"]["primary"] == (
        "Codex/subagent file review bundle"
    )
    assert checks["vision_provider"]["value"]["network_required"] is False


def test_explicit_external_large_v3_snapshot_is_visible_to_doctor(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "large-v3"
    snapshot.mkdir()
    for name in (
        "config.json",
        "model.bin",
        "preprocessor_config.json",
        "tokenizer.json",
        "vocabulary.json",
    ):
        (snapshot / name).write_bytes(b"fixture")
    monkeypatch.setenv("VSR_FASTER_WHISPER_LARGE_V3_PATH", str(snapshot))

    report = probe_optional_backends(model_root=tmp_path / "models")
    large_v3 = report["large_v3"]
    assert large_v3["offline_ready"] is bool(large_v3["package_version"])
    assert large_v3["model"]["source"] == "explicit_external"
    assert large_v3["model"]["missing_files"] == []

    doctor = doctor_report(output_path=tmp_path / "out", offline=True)
    speech = doctor["checks"]["speech_recognition"]
    assert speech["value"]["whisper_model_source"] == "explicit_external"
    assert speech["value"]["whisper_large_v3_offline_ready"] is large_v3["offline_ready"]


def test_doctor_reports_bounded_scheduler_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VSR_FRAME_EXTRACT_WORKERS", "3")
    monkeypatch.setenv("VSR_FRAME_ANALYSIS_WORKERS", "5")
    monkeypatch.setenv("VSR_SURVEY_FFMPEG_THREADS", "2")
    monkeypatch.setenv("VSR_OCR_WORKERS", "12")
    monkeypatch.setenv("VSR_OCR_CHECKPOINT_BATCH", "7")
    monkeypatch.setenv("VSR_ASR_CPU_THREADS", "4")
    monkeypatch.setenv("VSR_FASTER_WHISPER_NUM_WORKERS", "1")
    monkeypatch.setenv("VSR_PARALLEL_VISUAL_SURVEY", "1")

    scheduling = doctor_report(output_path=tmp_path, offline=True)["checks"]["scheduling"]
    assert scheduling["status"] == "available"
    assert scheduling["value"] == {
        "frame_extract_workers": 3,
        "frame_analysis_workers": 5,
        "crop_prepare_workers": 5,
        "survey_ffmpeg_threads": 2,
        "ocr_workers": 12,
        "ocr_checkpoint_batch": 7,
        "ocr_batch_size": 256,
        "asr_cpu_threads": 4,
        "asr_num_workers": 1,
        "validator_metadata_workers": 16,
        "parallel_visual_survey": True,
    }


def test_doctor_reports_shared_cache_usage_and_limits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VSR_ASR_SHARED_CACHE_DIR", str(tmp_path / "asr"))
    monkeypatch.setenv("VSR_VISUAL_SHARED_CACHE_DIR", str(tmp_path / "visual"))
    monkeypatch.setenv("VSR_SEMANTIC_SHARED_CACHE_DIR", str(tmp_path / "semantic"))
    monkeypatch.setenv("VSR_ASR_SHARED_CACHE_MAX_BYTES", "1234")
    monkeypatch.setenv("VSR_VISUAL_SHARED_CACHE_MAX_BYTES", "5678")
    monkeypatch.setenv("VSR_OCR_SHARED_CACHE_MAX_BYTES", "9012")

    shared = doctor_report(output_path=tmp_path / "out", offline=True)["checks"]["shared_caches"]
    assert shared["status"] == "available"
    entries = shared["value"]["entries"]
    assert entries["asr"]["limit_bytes"] == 1234
    assert entries["visual_frames"]["limit_bytes"] == 5678
    assert entries["ocr"]["limit_bytes"] == 9012
    assert shared["value"]["total_bytes"] == 0


def test_cache_usage_recurses_with_scandir_and_rejects_symlinks(
    tmp_path: Path, monkeypatch
) -> None:
    cache = tmp_path / "cache"
    nested = cache / "nested"
    nested.mkdir(parents=True)
    (cache / "root.bin").write_bytes(b"root")
    (nested / "child.bin").write_bytes(b"child")
    try:
        (cache / "outside-link").symlink_to(tmp_path / "outside.bin")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")

    def fail_rglob(*_args, **_kwargs):
        raise AssertionError("cache usage must recurse with scandir, not Path.rglob")

    monkeypatch.setattr(pipeline_module.Path, "rglob", fail_rglob)
    usage = pipeline_module._cache_directory_usage(cache)

    assert usage["file_count"] == 2
    assert usage["bytes"] == len(b"root") + len(b"child")
    assert usage["exists"] is True
