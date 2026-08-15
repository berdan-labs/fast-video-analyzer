from __future__ import annotations

from pathlib import Path

from video_script_reconstructor.worker_store import (
    WORKER_SPECS,
    list_workers,
    verify_worker,
    worker_python,
)


def test_worker_specs_pin_heavy_runtimes_and_keep_models_separate() -> None:
    assert set(WORKER_SPECS) == {"qwen-speech", "moss-speech", "paddle-ocr"}
    flattened = {
        name: " ".join(part for step in spec.install_steps for part in step)
        for name, spec in WORKER_SPECS.items()
    }
    assert "qwen-asr==0.0.6" in flattened["qwen-speech"]
    assert "0e3d1403fd8f1f1c674e883ece96b9f630794ebe" in flattened["moss-speech"]
    assert "paddleocr==3.7.0" in flattened["paddle-ocr"]
    assert all("model" not in command.casefold() for command in flattened.values())


def test_missing_workers_report_actionable_status_without_installing(tmp_path: Path) -> None:
    statuses = list_workers(tmp_path)
    assert len(statuses) == 3
    assert all(status["available"] is False for status in statuses)
    assert all("manifest is absent" in str(status["reason"]) for status in statuses)
    assert verify_worker("qwen-speech", tmp_path)["verified"] is False
    assert worker_python("qwen-speech", tmp_path).name in {"python", "python.exe"}


def test_worker_listing_reuses_short_stat_bound_probe_cache(
    monkeypatch, tmp_path: Path
) -> None:
    import video_script_reconstructor.worker_store as worker_store_module

    calls: list[str] = []

    def fake_verify(name: str, root: Path | None = None) -> dict[str, object]:
        calls.append(name)
        return {"name": name, "available": False, "verified": False}

    monkeypatch.setattr(worker_store_module, "verify_worker", fake_verify)
    first = list_workers(tmp_path)
    second = list_workers(tmp_path)

    assert len(first) == len(second) == 3
    assert calls == ["qwen-speech", "moss-speech", "paddle-ocr"]

    python = worker_python("qwen-speech", tmp_path)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"changed")
    list_workers(tmp_path)
    assert calls.count("qwen-speech") == 2
