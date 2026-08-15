from __future__ import annotations

import json
from pathlib import Path

import pytest

import video_script_reconstructor.model_store as model_store_module
from video_script_reconstructor.errors import ValidationFailure
from video_script_reconstructor.model_store import (
    MODEL_SPECS,
    list_models,
    model_report,
    remove_model,
    verify_model,
)
from video_script_reconstructor.security import sha256_file


def _install_fake_model(root: Path, name: str) -> Path:
    spec = MODEL_SPECS[name]
    directory = root / name
    directory.mkdir(parents=True)
    for relative in spec.required_files:
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}", encoding="utf-8")
    files = {
        path.relative_to(directory).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in directory.rglob("*")
        if path.is_file()
    }
    (directory / "model-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": name,
                "backend": spec.backend,
                "source": spec.source,
                "revision": "fixture-revision",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_model_status_is_explicit_when_weights_are_absent(tmp_path: Path) -> None:
    status = list_models(tmp_path)
    assert {item["name"] for item in status} == set(MODEL_SPECS)
    assert all(item["verified"] is False for item in status)
    assert all(item["offline_ready"] is False for item in status)


def test_model_manifest_verification_detects_mutation(tmp_path: Path) -> None:
    name = "faster-whisper-large-v3"
    directory = _install_fake_model(tmp_path, name)
    assert verify_model(name, tmp_path)["offline_ready"] is True
    (directory / "config.json").write_text("mutated", encoding="utf-8")
    result = verify_model(name, tmp_path)
    assert result["verified"] is False
    assert "config.json" in result["mismatched_files"]


def test_model_inventories_use_scandir_and_skip_verification_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "model"
    nested = directory / "nested"
    nested.mkdir(parents=True)
    (directory / "root.bin").write_bytes(b"root")
    (nested / "child.bin").write_bytes(b"child")
    cache = directory / ".cache"
    cache.mkdir()
    (cache / "receipt.json").write_bytes(b"ignored")

    def fail_rglob(*_args, **_kwargs):
        raise AssertionError("model inventory must recurse with scandir")

    monkeypatch.setattr(model_store_module.Path, "rglob", fail_rglob)
    manifest_files = model_store_module._manifest_files(directory)

    assert set(manifest_files) == {"root.bin", "nested/child.bin"}
    assert model_store_module._model_directory_usage(directory) == (3, len(b"root") + len(b"child") + len(b"ignored"))


def test_model_usage_inventory_rejects_symlinked_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "model"
    directory.mkdir()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    try:
        (directory / "link.bin").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(ValidationFailure, match="symlink"):
        model_store_module._model_directory_usage(directory)


def test_unchanged_model_uses_stat_bound_receipt_and_full_override_rehashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name = "faster-whisper-large-v3"
    _install_fake_model(tmp_path, name)
    original_hash = model_store_module.sha256_file
    hashed_paths: list[Path] = []

    def counted_hash(path: Path) -> str:
        hashed_paths.append(path)
        return original_hash(path)

    monkeypatch.setattr(model_store_module, "sha256_file", counted_hash)
    first = verify_model(name, tmp_path)
    first_count = len(hashed_paths)
    hashed_paths.clear()
    second = verify_model(name, tmp_path)
    cached_count = len(hashed_paths)
    hashed_paths.clear()
    forced = verify_model(name, tmp_path, force_full=True)

    assert first["verification_source"] == "full-hash"
    assert second["verification_source"] == "in-process-stat-cache"
    assert forced["verification_source"] == "full-hash"
    assert first_count > cached_count
    assert cached_count == 1  # Only the tiny manifest digest is recomputed.
    assert len(hashed_paths) > cached_count


def test_model_verification_cache_invalidates_on_manifest_or_file_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name = "faster-whisper-large-v3"
    directory = _install_fake_model(tmp_path, name)
    verify_model(name, tmp_path)

    original_hash = model_store_module.sha256_file
    hashed_paths: list[Path] = []

    def counted_hash(path: Path) -> str:
        hashed_paths.append(path)
        return original_hash(path)

    monkeypatch.setattr(model_store_module, "sha256_file", counted_hash)
    directory.joinpath("config.json").write_text("changed", encoding="utf-8")
    result = verify_model(name, tmp_path)

    assert result["verified"] is False
    assert "config.json" in result["mismatched_files"]
    assert hashed_paths


def test_model_removal_requires_matching_manifest(tmp_path: Path) -> None:
    name = "speechbrain-ecapa-voxceleb"
    directory = _install_fake_model(tmp_path, name)
    result = remove_model(name, tmp_path)
    assert result["removed"] is True
    assert not directory.exists()
    unowned = tmp_path / name
    unowned.mkdir()
    with pytest.raises(ValidationFailure, match="unverified directory"):
        remove_model(name, tmp_path)


def test_model_report_rolls_up_disk_footprint_without_marking_removal_safe(
    tmp_path: Path,
) -> None:
    name = "faster-whisper-large-v3"
    directory = _install_fake_model(tmp_path, name)
    (directory / "extra-cache.bin").write_bytes(b"cache")
    unverified = tmp_path / "speechbrain-ecapa-voxceleb"
    unverified.mkdir()
    (unverified / "unknown.bin").write_bytes(b"unowned")

    report = model_report(tmp_path)

    assert report["model_count"] == len(MODEL_SPECS)
    assert report["present_model_count"] == 2
    assert report["verified_model_count"] == 1
    item = next(row for row in report["models"] if row["name"] == name)
    assert item["disk_file_count"] == len(MODEL_SPECS[name].required_files) + 3
    assert item["disk_bytes"] >= item["manifest_file_bytes"]
    assert item["removal_requires_explicit_confirmation"] is True
    assert item["removal_blocked_until_verified"] is False
    assert "models remove faster-whisper-large-v3" in item["remove_command"]
    unverified_item = next(
        row for row in report["models"] if row["name"] == "speechbrain-ecapa-voxceleb"
    )
    assert unverified_item["removal_blocked_until_verified"] is True
    assert unverified_item["remove_command"] is None
    assert report["total_bytes"] == sum(row["disk_bytes"] for row in report["models"])
    assert report["unverified_bytes"] == report["total_bytes"] - item["disk_bytes"]
    legacy = next(row for row in report["models"] if row["name"] == "qwen2.5-vl-3b-q4")
    assert legacy["lifecycle"] == "legacy"
    assert legacy["replacement"] == "qwen3-vl-4b-q4"
    assert legacy["cleanup_recommendation"].startswith("Review before removal")
    assert report["legacy_model_count"] == 1
    assert report["present_legacy_model_count"] == 0


def test_model_report_can_probe_worker_readiness_explicitly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import video_script_reconstructor.worker_store as worker_store_module

    monkeypatch.setattr(
        worker_store_module,
        "list_workers",
        lambda: [
            {"name": "qwen-speech", "verified": True},
            {"name": "moss-speech", "verified": False},
            {"name": "paddle-ocr", "verified": False},
        ],
    )

    _install_fake_model(tmp_path, "qwen3-asr-1.7b")
    _install_fake_model(tmp_path, "moss-transcribe-diarize-0.9b")
    report = model_report(tmp_path, include_workers=True)

    assert report["workers_probed"] is True
    qwen = next(row for row in report["models"] if row["name"] == "qwen3-asr-1.7b")
    moss = next(row for row in report["models"] if row["name"] == "moss-transcribe-diarize-0.9b")
    whisper = next(row for row in report["models"] if row["name"] == "faster-whisper-large-v3")
    assert qwen["runtime_status"] == "verified"
    assert moss["runtime_status"] == "unavailable"
    assert qwen["runtime_unavailable"] is False
    assert moss["runtime_unavailable"] is True
    assert moss["cleanup_recommendation"].startswith("Worker unavailable")
    assert whisper["runtime_status"] == "self-contained"
    assert moss["runtime_reason"] is None
    assert report["runtime_unavailable_model_count"] == 1
    assert report["runtime_unavailable_bytes"] == moss["disk_bytes"]
