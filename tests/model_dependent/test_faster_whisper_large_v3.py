from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

from video_script_reconstructor.model_store import verify_model
from video_script_reconstructor.whisper_adapter import (
    FasterWhisperAdapter,
    transcribe_checkpointed_chunks,
)

pytestmark = [pytest.mark.model_dependent, pytest.mark.faster_whisper_model]


def test_real_large_v3_bounded_interval_model_metadata_and_checkpoint_resume(
    tmp_path: Path,
) -> None:
    installed = verify_model("faster-whisper-large-v3")
    model_value = os.environ.get("VSR_FASTER_WHISPER_LARGE_V3_PATH") or (
        str(installed["directory"]) if installed.get("offline_ready") else None
    )
    if not model_value:
        pytest.skip(
            "large-v3 weights unavailable: set VSR_FASTER_WHISPER_LARGE_V3_PATH "
            "to a local complete model directory"
        )
    model_path = Path(model_value).expanduser().resolve()
    if not model_path.is_dir():
        pytest.skip(
            f"large-v3 weights unavailable: configured directory does not exist: {model_path}"
        )
    if importlib.util.find_spec("faster_whisper") is None:
        pytest.skip("faster-whisper backend unavailable: install the 'asr' optional dependency")
    bundled_audio = (
        Path(__file__).resolve().parents[1] / "fixtures" / "model-dependent" / "large-v3-smoke.wav"
    )
    audio_value = os.environ.get("VSR_FASTER_WHISPER_SMOKE_AUDIO") or (
        str(bundled_audio) if bundled_audio.is_file() else None
    )
    if not audio_value:
        pytest.skip(
            "large-v3 smoke audio unavailable: set VSR_FASTER_WHISPER_SMOKE_AUDIO "
            "to a local file containing speech"
        )
    audio_path = Path(audio_value).expanduser().resolve()
    if not audio_path.is_file():
        pytest.skip(
            f"large-v3 smoke audio unavailable: configured file does not exist: {audio_path}"
        )

    start_ms = int(os.environ.get("VSR_FASTER_WHISPER_INTERVAL_START_MS", "0"))
    end_ms = int(os.environ.get("VSR_FASTER_WHISPER_INTERVAL_END_MS", "5000"))
    if end_ms <= start_ms:
        pytest.skip("configured large-v3 smoke interval must have positive duration")
    gpu_available = shutil.which("nvidia-smi") is not None
    adapter = FasterWhisperAdapter(
        model_path,
        device=os.environ.get("VSR_FASTER_WHISPER_DEVICE", "cuda" if gpu_available else "cpu"),
        compute_type=os.environ.get(
            "VSR_FASTER_WHISPER_COMPUTE_TYPE", "float16" if gpu_available else "int8"
        ),
        allow_model_download=False,
    )
    result = adapter.transcribe(
        audio_path,
        interval_start_ms=start_ms,
        interval_end_ms=end_ms,
        word_timestamps=True,
    )
    assert result.metadata["backend"] == "faster-whisper"
    assert Path(result.metadata["model"]).resolve() == model_path
    assert result.metadata["interval_start_ms"] == start_ms
    assert result.metadata["interval_end_ms"] == end_ms
    assert result.metadata["extracted_start_ms"] <= start_ms
    assert result.metadata["extracted_end_ms"] >= end_ms
    assert len(result.segments) > 0, "configured smoke interval must contain audible speech"
    assert (
        min(segment.start_ms for segment in result.segments)
        >= result.metadata["extracted_start_ms"]
    )
    assert result.metadata["package_versions"]["faster-whisper"] is not None

    span_ms = end_ms - start_ms
    overlap_ms = min(500, max(0, span_ms // 10))
    chunk_ms = max(overlap_ms + 1, max(1000, (span_ms + overlap_ms) // 2))
    first_checkpointed = transcribe_checkpointed_chunks(
        adapter,
        audio_path,
        duration_ms=end_ms,
        interval_start_ms=start_ms,
        interval_end_ms=end_ms,
        checkpoint_dir=tmp_path / "large-v3-checkpoints",
        chunk_ms=chunk_ms,
        overlap_ms=overlap_ms,
    )
    assert first_checkpointed.metadata["processed_chunk_indexes"]
    assert first_checkpointed.metadata["resumed_chunk_indexes"] == []
    assert first_checkpointed.metadata["model"] == str(model_path)
    assert first_checkpointed.metadata["model_metadata"]["backend"] == "faster-whisper"
    assert list((tmp_path / "large-v3-checkpoints").glob("*.json"))

    resumed = transcribe_checkpointed_chunks(
        adapter,
        audio_path,
        duration_ms=end_ms,
        interval_start_ms=start_ms,
        interval_end_ms=end_ms,
        checkpoint_dir=tmp_path / "large-v3-checkpoints",
        chunk_ms=chunk_ms,
        overlap_ms=overlap_ms,
    )
    assert resumed.metadata["processed_chunk_indexes"] == []
    assert resumed.metadata["resumed_chunk_indexes"] == list(
        range(len(first_checkpointed.metadata["chunk_ranges_ms"]))
    )
    assert (
        resumed.metadata["checkpoint_cache_key"]
        == first_checkpointed.metadata["checkpoint_cache_key"]
    )
