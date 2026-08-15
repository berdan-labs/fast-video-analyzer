from __future__ import annotations

import os
from pathlib import Path

import pytest

from video_script_reconstructor.model_store import verify_model
from video_script_reconstructor.qwen_asr_adapter import Qwen3ASRAdapter
from video_script_reconstructor.whisper_adapter import transcribe_checkpointed_chunks

pytestmark = [pytest.mark.model_dependent, pytest.mark.qwen3_asr]


def test_real_qwen3_asr_uses_forced_spans_and_resumable_chunks(tmp_path: Path) -> None:
    asr_status = verify_model("qwen3-asr-1.7b")
    aligner_status = verify_model("qwen3-forced-aligner-0.6b")
    if not asr_status.get("offline_ready") or not aligner_status.get("offline_ready"):
        pytest.skip("Qwen3-ASR or forced-aligner weights are unavailable")
    repository = Path(__file__).resolve().parents[2]
    default_worker = repository / ".artifacts" / "workers" / "qwen-asr" / "Scripts" / "python.exe"
    worker = Path(os.environ.get("VSR_QWEN_SPEECH_PYTHON", default_worker)).resolve()
    if not worker.is_file():
        pytest.skip("Qwen speech worker unavailable: set VSR_QWEN_SPEECH_PYTHON")
    audio = (
        Path(__file__).resolve().parents[1] / "fixtures" / "model-dependent" / "large-v3-smoke.wav"
    )
    adapter = Qwen3ASRAdapter(
        worker_python=worker,
        aligner_name="qwen3-forced-aligner-0.6b",
        timeout_seconds=600,
    )
    result = adapter.transcribe(audio, language="English", word_timestamps=True)

    assert "accuracy first reconstruction" in result.segments[0].normalized_text.casefold()
    assert "verified value" in result.segments[0].normalized_text.casefold()
    assert result.segments[0].words
    assert result.segments[0].timing_provenance == "qwen3_forced_alignment"
    assert result.metadata["model_path_or_revision"] == asr_status["revision"]
    assert result.metadata["aligner_revision"] == aligner_status["revision"]
    assert result.metadata["offline"] is True

    first = transcribe_checkpointed_chunks(
        adapter,
        audio,
        duration_ms=7738,
        checkpoint_dir=tmp_path / "qwen-checkpoints",
        chunk_ms=5000,
        overlap_ms=500,
        language="English",
    )
    assert first.metadata["processed_chunk_indexes"] == [0, 1]
    resumed = transcribe_checkpointed_chunks(
        adapter,
        audio,
        duration_ms=7738,
        checkpoint_dir=tmp_path / "qwen-checkpoints",
        chunk_ms=5000,
        overlap_ms=500,
        language="English",
    )
    assert resumed.metadata["processed_chunk_indexes"] == []
    assert resumed.metadata["resumed_chunk_indexes"] == [0, 1]
    assert resumed.metadata["checkpoint_cache_key"] == first.metadata["checkpoint_cache_key"]
