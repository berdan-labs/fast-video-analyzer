from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from video_script_reconstructor.moss_adapter import MossTranscribeDiarizeAdapter
from video_script_reconstructor.whisper_adapter import ASRDependencyError


def _verified(_: str, __: Path | None) -> dict[str, object]:
    return {
        "verified": True,
        "directory": "C:/models/moss",
        "revision": "pinned-moss-revision",
    }


def test_moss_adapter_preserves_anonymous_speaker_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fixture")
    monkeypatch.setattr("video_script_reconstructor.moss_adapter.verify_model", _verified)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        request = json.loads(str(kwargs["input"]))
        assert request["mode"] == "transcribe-diarize"
        payload = {
            "ok": True,
            "segments": [
                {"start": 0.2, "end": 1.1, "speaker": "S01", "text": "First turn."},
                {"start": 1.3, "end": 2.4, "speaker": "S02", "text": "Second turn."},
            ],
            "prompt": "fixed prompt",
            "raw_output": "[0.2][S01]First turn.[1.1][1.3][S02]Second turn.[2.4]",
            "device": "cuda:0",
            "dtype": "bfloat16",
            "generated_tokens": 32,
            "package_versions": {"moss-transcribe-diarize": "0.1.0"},
        }
        return subprocess.CompletedProcess(
            args=["worker"],
            returncode=0,
            stdout="diagnostic\nVSR_RESULT\t" + json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = MossTranscribeDiarizeAdapter(worker_python=sys.executable).transcribe(audio)

    assert [segment.speaker_label for segment in result.segments] == ["Speaker 1", "Speaker 2"]
    assert [(segment.start_ms, segment.end_ms) for segment in result.segments] == [
        (200, 1100),
        (1300, 2400),
    ]
    assert all(
        segment.uncertainty_items == ["model_generated_speaker_label"]
        for segment in result.segments
    )
    assert result.metadata["speaker_labels_are_anonymous"] is True
    assert result.metadata["speaker_label_mapping"] == {"S01": "Speaker 1", "S02": "Speaker 2"}
    assert result.metadata["model_path_or_revision"] == "pinned-moss-revision"
    assert result.metadata["offline"] is True


def test_moss_adapter_rejects_unverified_custom_code_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fixture")
    monkeypatch.setattr(
        "video_script_reconstructor.moss_adapter.verify_model",
        lambda *_: {"verified": False, "reason": "hash mismatch"},
    )
    adapter = MossTranscribeDiarizeAdapter(worker_python=sys.executable)

    with pytest.raises(ASRDependencyError, match="hash-verified"):
        adapter.transcribe(audio)
