from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _profile_module() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "profile_visual_stage.py"
    spec = importlib.util.spec_from_file_location("vsr_profile_visual_stage", path)
    if spec is None or spec.loader is None:
        raise AssertionError("visual profile script could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visual_profile_threads_one_source_digest_through_visual_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _profile_module()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"profile source")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        module,
        "probe_media",
        lambda _source: SimpleNamespace(duration_ms=1_000),
    )

    def fake_extract(*_args: Any, **kwargs: Any) -> tuple[list[Any], ...]:
        captured.update(kwargs)
        return ([], [], [], [], [], [])

    monkeypatch.setattr(module, "_extract_visual_evidence", fake_extract)
    hash_calls: list[Path] = []

    def fake_sha256(path: Path) -> str:
        hash_calls.append(path)
        return "a" * 64

    monkeypatch.setattr(module, "sha256_file", fake_sha256)

    report = module.profile(source, output=tmp_path / "output")

    assert report["full_frame_count"] == 0
    assert hash_calls == [source.resolve()]
    assert captured["source_sha256"] == "a" * 64
