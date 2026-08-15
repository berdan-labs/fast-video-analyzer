from __future__ import annotations

import os
from pathlib import Path

import pytest

from video_script_reconstructor.model_store import verify_model
from video_script_reconstructor.qwen_asr_adapter import Qwen3ForcedAlignmentAdapter

pytestmark = [pytest.mark.model_dependent, pytest.mark.forced_alignment]


def test_real_qwen3_forced_alignment_spans_are_ordered_and_exact() -> None:
    status = verify_model("qwen3-forced-aligner-0.6b")
    if not status.get("offline_ready"):
        pytest.skip("Qwen3 forced-aligner weights unavailable: fetch the managed local model")
    repository = Path(__file__).resolve().parents[2]
    default_worker = repository / ".artifacts" / "workers" / "qwen-asr" / "Scripts" / "python.exe"
    worker = Path(os.environ.get("VSR_QWEN_SPEECH_PYTHON", default_worker)).resolve()
    if not worker.is_file():
        pytest.skip(
            "Qwen speech worker unavailable: set VSR_QWEN_SPEECH_PYTHON to its Python executable"
        )
    audio = (
        Path(__file__).resolve().parents[1] / "fixtures" / "model-dependent" / "large-v3-smoke.wav"
    )
    transcript = (
        "The accuracy first reconstruction keeps every exact word. The verified value is 42."
    )
    result = Qwen3ForcedAlignmentAdapter(worker_python=worker).align(
        audio, transcript, language="English"
    )

    assert " ".join(span.text for span in result.spans) == transcript.replace(".", "").strip()
    assert all(span.start_ms < span.end_ms for span in result.spans)
    assert all(
        current.start_ms >= previous.start_ms
        for previous, current in zip(result.spans, result.spans[1:], strict=False)
    )
    assert result.spans[-1].text == "42"
    assert result.metadata["model_path_or_revision"] == status["revision"]
    assert result.metadata["offline"] is True
