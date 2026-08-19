"""Regression tests for the standalone repository continuity backup helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "backup_repo.py"
SPEC = importlib.util.spec_from_file_location("backup_repo", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
backup_repo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup_repo)


def test_checksum_manifest_detects_changed_and_unexpected_files(tmp_path: Path) -> None:
    backup = tmp_path / "backup"
    backup.mkdir()
    tracked = backup / "github-metadata" / "repository.json"
    tracked.parent.mkdir()
    tracked.write_text('{"name": "fast-video-analyzer"}\n', encoding="utf-8")

    backup_repo.write_checksums(backup)
    backup_repo.validate_checksums(backup)

    tracked.write_text('{"name": "changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        backup_repo.validate_checksums(backup)

    backup_repo.write_checksums(backup)
    (backup / "unexpected.txt").write_text("not in manifest\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not match SHA256SUMS"):
        backup_repo.validate_checksums(backup)


@pytest.mark.parametrize(
    "value",
    ("../escape", "/absolute/path", "C:/absolute/path", r"\\server\share\path"),
)
def test_safe_relative_path_rejects_escape_attempts(value: str) -> None:
    with pytest.raises(ValueError, match="unsafe relative path"):
        backup_repo._safe_relative_path(value)


def test_backup_destination_cannot_be_inside_checkout() -> None:
    with pytest.raises(ValueError, match="outside the repository checkout"):
        backup_repo._require_external_directory(backup_repo.ROOT / "backups")
