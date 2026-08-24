from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from video_script_reconstructor.pipeline import _timeline_record, _write_timeline_records
from video_script_reconstructor.security import atomic_write_json
from video_script_reconstructor.timeline import TimelineItem


@dataclass(slots=True)
class _NestedRecord:
    value: int


def _item(payload: Any) -> TimelineItem:
    return TimelineItem(
        timeline_id="timeline_probe00000000000",
        kind="chapter",
        source_id="probe_source",
        start_ms=0,
        end_ms=1,
        timing_provenance="source_timing",
        payload=payload,
        source_order=0,
    )


def _document(items: list[TimelineItem], materialize: Any) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "items": [materialize(item) for item in items],
        "validation": {
            "valid": True,
            "errors": [],
            "warnings": [],
            "timed_count": len(items),
            "untimed_count": 0,
        },
    }


def _validation(item_count: int) -> dict[str, Any]:
    return {
        "valid": True,
        "errors": [],
        "warnings": [],
        "timed_count": item_count,
        "untimed_count": 0,
    }


def test_model_dump_keeps_recursive_isolation() -> None:
    item = _item({"block_ids": ["B1"], "metadata": {"score": 1}})

    dumped = item.model_dump()
    dumped["payload"]["block_ids"].append("B2")
    dumped["payload"]["metadata"]["score"] = 2

    assert item.payload == {"block_ids": ["B1"], "metadata": {"score": 1}}


def test_plain_json_fast_path_preserves_values_and_record_ownership() -> None:
    item = _item({"block_ids": ["B1"], "metadata": {"score": 1}, "visible": True})

    record = _timeline_record(item)

    assert record == item.model_dump()
    assert record["payload"] is not item.payload
    assert record["payload"]["block_ids"] is item.payload["block_ids"]
    record["start_ms"] = 99
    record["payload"] = None
    assert item.start_ms == 0
    assert item.payload == {"block_ids": ["B1"], "metadata": {"score": 1}, "visible": True}


def test_non_dict_payload_keeps_model_dump_isolation() -> None:
    item = _item(["value", [1, 2]])

    record = _timeline_record(item)
    record["payload"][1].append(3)

    assert item.payload == ["value", [1, 2]]


def test_exotic_payload_falls_back_to_isolated_model_dump(tmp_path: Path) -> None:
    item = _item({"nested": _NestedRecord(7), "values": [1, 2]})

    records = _write_timeline_records(tmp_path / "timeline.json", [item], _validation(1))
    record = records[0]

    assert record == item.model_dump()
    assert record["payload"] is not item.payload
    record["payload"]["nested"]["value"] = 99
    record["payload"]["values"].append(3)
    assert item.payload["nested"].value == 7
    assert item.payload["values"] == [1, 2]


def test_failed_fallback_leaves_no_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "timeline.json"

    with pytest.raises(TypeError):
        _write_timeline_records(path, [_item({"values": {1, 2}})], _validation(1))

    assert not path.exists()


def test_timeline_document_bytes_match_model_dump_reference(tmp_path: Path) -> None:
    items = [
        _item({"block_ids": ["B1"], "metadata": {"score": 0.5}}),
        _item({"nested": _NestedRecord(7), "api_key": "secret-value"}),
    ]
    optimized_path = tmp_path / "optimized.json"
    reference_path = tmp_path / "reference.json"

    optimized_records = _write_timeline_records(
        optimized_path,
        items,
        _validation(len(items)),
    )
    atomic_write_json(reference_path, _document(items, lambda item: item.model_dump()), compact=True)

    optimized = optimized_path.read_bytes()
    assert optimized == reference_path.read_bytes()
    assert optimized_records == [item.model_dump() for item in items]
    assert optimized.endswith(b"\n") and optimized.count(b"\n") == 1
    decoded = json.loads(optimized)
    assert decoded["items"][0]["payload"] == items[0].payload
    assert decoded["items"][1]["payload"]["nested"] == {"value": 7}
    assert decoded["items"][1]["payload"]["api_key"] == "[REDACTED]"
    assert b"secret-value" not in optimized
