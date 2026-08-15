from pathlib import Path

import video_script_reconstructor.resource_usage as resource_usage_module
from video_script_reconstructor.resource_usage import (
    directory_usage,
    process_memory_usage,
    resource_snapshot,
)


def test_resource_snapshot_counts_generated_files_without_following_links(tmp_path: Path) -> None:
    (tmp_path / "one.bin").write_bytes(b"123")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "two.bin").write_bytes(b"4567")

    usage = directory_usage(tmp_path)
    snapshot = resource_snapshot(tmp_path)

    assert usage == {"file_count": 2, "bytes": 7}
    assert snapshot["output"] == {
        "file_count": 2,
        "bytes": 7,
        "reclaimable_file_count": 0,
        "reclaimable_bytes": 0,
    }
    assert set(process_memory_usage()) == {"current_rss_bytes", "peak_rss_bytes"}


def test_directory_usage_avoids_recursive_path_metadata_scan(
    tmp_path: Path, monkeypatch
) -> None:
    """Telemetry inventory should use one ``scandir`` pass, not path walkers."""

    (tmp_path / "one.bin").write_bytes(b"123")
    cache = tmp_path / ".state" / "cache"
    cache.mkdir(parents=True)
    (cache / "cached.bin").write_bytes(b"4567")
    ocr = tmp_path / ".state" / "checkpoints" / "ocr"
    ocr.mkdir(parents=True)
    (ocr / "ocr.bin").write_bytes(b"89")

    def forbidden_rglob(*_args, **_kwargs):
        raise AssertionError("directory usage should not recursively enumerate with Path.rglob")

    def forbidden_walk(*_args, **_kwargs):
        raise AssertionError("directory usage should recurse with scandir, not os.walk")

    monkeypatch.setattr(resource_usage_module.Path, "rglob", forbidden_rglob)
    monkeypatch.setattr(resource_usage_module.os, "walk", forbidden_walk)

    assert directory_usage(tmp_path, include_reclaimable=True) == {
        "file_count": 3,
        "bytes": 9,
        "reclaimable_file_count": 2,
        "reclaimable_bytes": 6,
    }
