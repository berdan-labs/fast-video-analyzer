from __future__ import annotations

import json
from pathlib import Path

from video_script_reconstructor.ocr_refresh import (
    _packet_ocr_projection_from_canonical,
    _refresh_workers,
    _replace_packet_ocr,
)


def test_refresh_workers_is_bounded_and_env_overridable(monkeypatch) -> None:
    monkeypatch.setenv("VSR_OCR_REFRESH_WORKERS", "4")
    assert _refresh_workers(None) == 4
    assert _refresh_workers(2) == 2


def test_replace_packet_ocr_updates_only_changed_packet_context(tmp_path: Path) -> None:
    packet_dir = tmp_path / ".state" / "vision" / "packets"
    packet_dir.mkdir(parents=True)
    unchanged = {
        "schema_name": "video-script-reconstructor.vision-packet",
        "raw_ocr": [{"observation_id": "O000001", "normalized_interpretation": "same"}],
    }
    changed = {
        "schema_name": "video-script-reconstructor.vision-packet",
        "raw_ocr": [{"observation_id": "O000002", "normalized_interpretation": "old"}],
    }
    (packet_dir / "V000001.json").write_text(
        json.dumps(unchanged), encoding="utf-8"
    )
    (packet_dir / "V000002.json").write_text(json.dumps(changed), encoding="utf-8")

    updated = _replace_packet_ocr(
        tmp_path,
        {
            "O000001": {"observation_id": "O000001", "normalized_interpretation": "same"},
            "O000002": {"observation_id": "O000002", "normalized_interpretation": "new"},
        },
    )

    assert updated == 1
    assert json.loads((packet_dir / "V000001.json").read_text(encoding="utf-8")) == unchanged
    assert json.loads((packet_dir / "V000002.json").read_text(encoding="utf-8"))["raw_ocr"][0][
        "normalized_interpretation"
    ] == "new"


def test_canonical_ocr_projection_restores_packet_shape() -> None:
    projection = _packet_ocr_projection_from_canonical(
        {
            "observation_id": "O000001",
            "frame_id": "F000001",
            "raw_engine_text": 'a "quoted" token',
            "normalized_interpretation": 'a "quoted" token',
            "confidence": 0.8,
            "uncertain_characters": ["9 (engine confidence below threshold; confidence=40)"],
            "engine": "tesseract",
        }
    )

    assert set(projection) == {
        "observation_id",
        "frame_id",
        "raw_engine_text",
        "normalized_interpretation",
        "confidence",
        "uncertain_characters",
    }
    assert projection["uncertain_characters"][0]["text"].startswith("9 ")
