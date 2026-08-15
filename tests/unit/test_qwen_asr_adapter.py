from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from video_script_reconstructor.qwen_asr_adapter import (
    Qwen3ASRAdapter,
    Qwen3ForcedAlignmentAdapter,
)
from video_script_reconstructor.whisper_adapter import ASRDependencyError, ASRError


def _status(name: str, _: Path | None) -> dict[str, object]:
    return {
        "name": name,
        "verified": True,
        "directory": f"C:/models/{name}",
        "revision": f"{name}-revision",
    }


def test_qwen_adapter_normalizes_unaligned_worker_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    monkeypatch.setattr("video_script_reconstructor.qwen_asr_adapter.verify_model", _status)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        request = json.loads(str(kwargs["input"]))
        assert request["model_path"].endswith("qwen3-asr-1.7b")
        assert request["language"] == "Filipino"
        payload = {
            "ok": True,
            "text": "Exact spoken words.",
            "language": "English",
            "duration_ms": 1875,
            "time_stamps": None,
            "device": "cuda:0",
            "dtype": "float16",
            "package_versions": {"qwen-asr": "0.0.6"},
        }
        return subprocess.CompletedProcess(
            args=["worker"],
            returncode=0,
            stdout="noise\nVSR_RESULT\t" + json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = Qwen3ASRAdapter(worker_python=sys.executable)
    result = adapter.transcribe(audio, language="fil")

    assert result.language == "English"
    assert result.segments[0].normalized_text == "Exact spoken words."
    assert result.segments[0].start_ms == 0
    assert result.segments[0].end_ms == 1875
    assert result.segments[0].words == []
    assert result.segments[0].timing_provenance == "qwen3_asr_utterance_bounds_estimated"
    assert result.segments[0].uncertainty_items == ["word_timestamps_unavailable"]
    assert result.metadata["timing_precision"] == "utterance-bounds-estimated"
    assert result.metadata["offline"] is True


def test_qwen_adapter_normalizes_forced_aligner_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    monkeypatch.setattr("video_script_reconstructor.qwen_asr_adapter.verify_model", _status)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        request = json.loads(str(kwargs["input"]))
        assert request["aligner_path"].endswith("qwen3-forced-aligner-0.6b")
        payload = {
            "ok": True,
            "text": "Exact words",
            "language": "English",
            "duration_ms": 2000,
            "time_stamps": [
                {"text": "Exact", "start_time": 0.25, "end_time": 0.7},
                {"text": "words", "start_time": 0.8, "end_time": 1.4},
            ],
        }
        return subprocess.CompletedProcess(
            args=["worker"], returncode=0, stdout="VSR_RESULT\t" + json.dumps(payload), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = Qwen3ASRAdapter(
        worker_python=sys.executable,
        aligner_name="qwen3-forced-aligner-0.6b",
    )
    result = adapter.transcribe(audio)

    segment = result.segments[0]
    assert (segment.start_ms, segment.end_ms) == (250, 1400)
    assert [word["text"] for word in segment.words] == ["Exact", "words"]
    assert segment.timing_provenance == "qwen3_forced_alignment"
    assert segment.uncertainty_items == []
    assert result.metadata["timing_precision"] == "forced-aligned"


def test_qwen_adapter_rejects_unverified_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    monkeypatch.setattr(
        "video_script_reconstructor.qwen_asr_adapter.verify_model",
        lambda *_: {"verified": False, "reason": "manifest is absent"},
    )
    adapter = Qwen3ASRAdapter(worker_python=sys.executable)

    with pytest.raises(ASRDependencyError, match="hash-verified"):
        adapter.transcribe(audio)


def test_qwen_adapter_reports_worker_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    monkeypatch.setattr("video_script_reconstructor.qwen_asr_adapter.verify_model", _status)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["worker"],
            returncode=1,
            stdout='VSR_RESULT\t{"ok":false,"error":"out of memory"}',
            stderr="",
        ),
    )
    adapter = Qwen3ASRAdapter(worker_python=sys.executable)

    with pytest.raises(ASRError, match="out of memory"):
        adapter.transcribe(audio)


def test_qwen_forced_alignment_adapter_validates_spans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    monkeypatch.setattr("video_script_reconstructor.qwen_asr_adapter.verify_model", _status)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        request = json.loads(str(kwargs["input"]))
        assert request["mode"] == "align"
        assert request["text"] == "Exact words"
        payload = {
            "ok": True,
            "time_stamps": [
                {"text": "Exact", "start_time": 0.2, "end_time": 0.7},
                {"text": "words", "start_time": 0.8, "end_time": 1.3},
            ],
        }
        return subprocess.CompletedProcess(
            args=["worker"], returncode=0, stdout="VSR_RESULT\t" + json.dumps(payload), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = Qwen3ForcedAlignmentAdapter(worker_python=sys.executable)
    result = adapter.align(audio, "Exact words", language="English")

    assert [(span.text, span.start_ms, span.end_ms) for span in result.spans] == [
        ("Exact", 200, 700),
        ("words", 800, 1300),
    ]
    assert result.metadata["backend"] == "qwen3-forced-aligner"
    assert result.metadata["offline"] is True
