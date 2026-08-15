from __future__ import annotations

import os
from pathlib import Path

import pytest

from video_script_reconstructor.model_store import verify_model
from video_script_reconstructor.moss_adapter import MossTranscribeDiarizeAdapter

pytestmark = [pytest.mark.model_dependent, pytest.mark.neural_diarization]


def test_real_moss_transcription_emits_anonymous_timed_speaker_turns() -> None:
    status = verify_model("moss-transcribe-diarize-0.9b")
    if not status.get("offline_ready"):
        pytest.skip("MOSS weights unavailable: fetch the managed local model")
    repository = Path(__file__).resolve().parents[2]
    default_worker = repository / ".artifacts" / "workers" / "moss" / "Scripts" / "python.exe"
    worker = Path(os.environ.get("VSR_MOSS_SPEECH_PYTHON", default_worker)).resolve()
    if not worker.is_file():
        pytest.skip("MOSS worker unavailable: set VSR_MOSS_SPEECH_PYTHON")
    audio = (
        Path(__file__).resolve().parents[1] / "fixtures" / "model-dependent" / "large-v3-smoke.wav"
    )
    result = MossTranscribeDiarizeAdapter(worker_python=worker, max_new_tokens=512).transcribe(
        audio, language="English"
    )

    reconstructed = " ".join(segment.normalized_text for segment in result.segments)
    assert "accuracy first reconstruction" in reconstructed.casefold()
    assert "verified value" in reconstructed.casefold()
    assert all(segment.speaker_label for segment in result.segments)
    assert all(segment.start_ms < segment.end_ms for segment in result.segments)
    assert all(
        current.start_ms >= previous.start_ms
        for previous, current in zip(result.segments, result.segments[1:], strict=False)
    )
    assert result.metadata["speaker_labels_are_anonymous"] is True
    assert result.metadata["model_path_or_revision"] == status["revision"]
    assert result.metadata["offline"] is True
