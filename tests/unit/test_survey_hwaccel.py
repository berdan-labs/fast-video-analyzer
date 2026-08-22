"""Focused coverage for the off-by-default survey hwaccel experiment."""

from __future__ import annotations

from pathlib import Path

import pytest

import video_script_reconstructor.pipeline as pipeline_module
from video_script_reconstructor.frame_extract import build_frame_extraction_command
from video_script_reconstructor.pipeline import (
    _survey_hwaccel_effective_mode,
    _survey_hwaccel_telemetry,
    _visual_survey_cache_identity,
)
from video_script_reconstructor.scene_detection import build_combined_survey_command


@pytest.fixture(autouse=True)
def _isolated_hwaccel_state(monkeypatch: pytest.MonkeyPatch):
    """Keep probe memoization and the request env out of other tests."""

    pipeline_module._SURVEY_HWACCEL_PROBES.clear()
    monkeypatch.delenv("VSR_SURVEY_HWACCEL", raising=False)
    yield
    pipeline_module._SURVEY_HWACCEL_PROBES.clear()


def _write_source(tmp_path: Path) -> Path:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"hwaccel survey fixture")
    return source


def test_default_survey_command_is_unchanged_and_cuda_precedes_input(
    tmp_path: Path,
) -> None:
    media = _write_source(tmp_path)

    default_command = build_combined_survey_command(media)
    cuda_command = build_combined_survey_command(media, hwaccel="cuda")

    assert "-hwaccel" not in default_command
    index = cuda_command.index("-hwaccel")
    assert cuda_command[index : index + 2] == ["-hwaccel", "cuda"]
    # ``-hwaccel`` is an input option, so it must land directly before ``-i``.
    assert cuda_command[index + 2] == "-i"
    assert cuda_command[:index] + cuda_command[index + 2 :] == default_command

    frame_default = build_frame_extraction_command(media, 0, tmp_path / "frame.png")
    frame_cuda = build_frame_extraction_command(
        media, 0, tmp_path / "frame.png", hwaccel="cuda"
    )
    frame_index = frame_cuda.index("-hwaccel")
    assert frame_cuda[frame_index : frame_index + 2] == ["-hwaccel", "cuda"]
    assert frame_cuda[frame_index + 2] == "-ss"
    assert frame_cuda[:frame_index] + frame_cuda[frame_index + 2 :] == frame_default


def test_survey_cache_key_partitions_decode_mode(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    project_dir = tmp_path / "project"

    def identity(**kwargs: object) -> str:
        _path, key, _digest, _ffmpeg = _visual_survey_cache_identity(
            source,
            project_dir,
            duration_ms=60_000,
            interval_seconds=2.0,
            strict=False,
            scene_detection=True,
            adaptive_detection=True,
            speech_reference_times_ms=(),
            source_sha256=None,
            **kwargs,  # type: ignore[arg-type]
        )
        return key

    software_default = identity()
    software_explicit = identity(decode_mode=None)
    hardware = identity(decode_mode="cuda")

    # Omitting the component must stay byte-identical to the historical key.
    assert software_explicit == software_default
    assert hardware != software_default


def test_survey_hwaccel_telemetry_reports_unset_and_invalid_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VSR_SURVEY_HWACCEL", raising=False)
    assert _survey_hwaccel_telemetry() == {
        "requested": None,
        "effective": None,
        "status": "disabled",
        "detail": "",
    }

    monkeypatch.setenv("VSR_SURVEY_HWACCEL", "vulkan")
    rejected = _survey_hwaccel_telemetry()
    assert rejected["status"] == "rejected"
    assert rejected["requested"] == "vulkan"
    assert rejected["effective"] is None
    assert rejected["detail"] == "allowed values: cuda"


def test_probe_mismatch_memoizes_and_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(tmp_path)
    calls: list[tuple[str, Path]] = []

    def failing_probe(mode: str, probed_source: Path) -> dict[str, str]:
        calls.append((mode, probed_source))
        return {"status": "mismatch", "detail": "pixels differ"}

    monkeypatch.setattr(pipeline_module, "_probe_survey_hwaccel", failing_probe)
    monkeypatch.setenv("VSR_SURVEY_HWACCEL", "cuda")

    first = _survey_hwaccel_effective_mode(source, source_sha256="abc123")
    second = _survey_hwaccel_effective_mode(source, source_sha256="abc123")

    assert first is None
    assert second is None
    assert calls == [("cuda", source)]

    telemetry = _survey_hwaccel_telemetry()
    assert telemetry["status"] == "mismatch"
    assert telemetry["effective"] is None
