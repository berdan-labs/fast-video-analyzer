from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_script_reconstructor.errors import InputError, SecurityError
from video_script_reconstructor.ingest import (
    SourceKind,
    classify_local_source,
    ingest_local_source,
)
from video_script_reconstructor.media_probe import build_ffprobe_command, parse_ffprobe_json


def test_parse_ffprobe_preserves_stream_timing_rotation_and_vfr(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"media")
    payload = {
        "format": {
            "format_name": "mov,mp4",
            "duration": "2.501",
            "size": "5",
            "tags": {"title": "x"},
        },
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "duration": "2.5",
                "start_time": "0.04",
                "time_base": "1/90000",
                "width": 1920,
                "height": 1080,
                "sample_aspect_ratio": "1:1",
                "r_frame_rate": "30/1",
                "avg_frame_rate": "2997/100",
                "tags": {"language": "eng"},
                "side_data_list": [{"rotation": -90}],
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "time_base": "1/48000",
                "channels": 2,
                "sample_rate": "48000",
            },
        ],
        "chapters": [{"id": 4, "start_time": "0", "end_time": "2.5", "tags": {"title": "Intro"}}],
    }
    result = parse_ffprobe_json(json.dumps(payload), source_path=source)
    assert result.duration_ms == 2501
    assert result.video_streams[0].time_base == "1/90000"
    assert result.video_streams[0].rotation == -90
    assert result.variable_frame_rate is True
    assert result.chapters[0].end_ms == 2500


def test_ffprobe_command_keeps_hostile_path_as_single_argument(tmp_path: Path) -> None:
    source = tmp_path / "video; whoami $(ignored).mp4"
    command = build_ffprobe_command(source, ffprobe_bin="ffprobe-custom")
    assert command[0] == "ffprobe-custom"
    assert command[-1] == str(source)
    assert command.count(str(source)) == 1


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("x.MKV", SourceKind.VIDEO),
        ("x.flac", SourceKind.AUDIO),
        ("x.SSA", SourceKind.SUBTITLE),
        ("x.json", SourceKind.TIMESTAMPED_TRANSCRIPT),
        ("x.txt", SourceKind.PLAIN_TRANSCRIPT),
    ],
)
def test_classify_supported_sources_case_insensitively(name: str, kind: SourceKind) -> None:
    assert classify_local_source(name) is kind


def test_ingest_plain_transcript_is_content_addressed_without_media_probe(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("spoken evidence", encoding="utf-8")
    first = ingest_local_source(source)
    second = ingest_local_source(source)
    assert first.media_id == second.media_id
    assert first.media_id.startswith("M")
    assert first.probe is None
    assert first.content_sha256 == second.content_sha256


def test_ingest_rejects_url_and_symlink(tmp_path: Path) -> None:
    with pytest.raises(SecurityError):
        ingest_local_source("https://example.com/video.mp4")
    with pytest.raises(InputError):
        classify_local_source("video.exe")
