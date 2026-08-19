"""Create, verify, and restore a non-secret repository continuity backup.

The backup is intentionally made outside the checkout. It contains a bare Git
mirror plus non-secret GitHub control-plane metadata. It never exports Actions,
environment, or PyPI secrets; those must be recreated through their owning
service during a recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "berdan-labs/fast-video-analyzer"
API_VERSION = "X-GitHub-Api-Version: 2022-11-28"

METADATA_ENDPOINTS = {
    "repository": ("repos/{repository}", False),
    "main-branch-protection": (
        "repos/{repository}/branches/main/protection",
        False,
    ),
    "actions-permissions": ("repos/{repository}/actions/permissions", False),
    "environments": ("repos/{repository}/environments", False),
    "rulesets": ("repos/{repository}/rulesets", False),
    "workflows": ("repos/{repository}/actions/workflows", False),
    "collaborators": ("repos/{repository}/collaborators", True),
    "deploy-keys": ("repos/{repository}/keys", True),
    "labels": ("repos/{repository}/labels", True),
    "milestones": ("repos/{repository}/milestones", True),
    "releases": ("repos/{repository}/releases", True),
}


def _run(command: Sequence[str], *, cwd: Path | None = None) -> str:
    """Run a prerequisite command and turn failures into concise diagnostics."""

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"required command is unavailable: {command[0]}") from exc
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({' '.join(command)}): {detail}")
    return completed.stdout


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or path.anchor
        or windows_path.drive
        or windows_path.root
        or ".." in path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(f"unsafe relative path in backup metadata: {value!r}")
    return path


def _require_external_directory(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise ValueError("backup destination must be outside the repository checkout")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _backup_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")


def write_checksums(root: Path) -> None:
    """Write a complete checksum manifest without checksumming itself."""

    lines = [
        f"{_file_hash(path)}  {path.relative_to(root).as_posix()}" for path in _backup_files(root)
    ]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_checksums(root: Path) -> None:
    """Reject altered, omitted, or unexpected files in a continuity backup."""

    checksum_file = root / "SHA256SUMS"
    if not checksum_file.is_file():
        raise ValueError("backup is missing SHA256SUMS")

    expected: dict[Path, str] = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, separator, raw_path = line.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError("backup has a malformed SHA256SUMS entry")
        relative_path = _safe_relative_path(raw_path)
        if relative_path in expected:
            raise ValueError("backup has duplicate SHA256SUMS entries")
        expected[relative_path] = digest

    actual = {path.relative_to(root) for path in _backup_files(root)}
    if set(expected) != actual:
        raise ValueError("backup files do not match SHA256SUMS")

    for relative_path, expected_digest in expected.items():
        path = root / relative_path
        if _file_hash(path) != expected_digest:
            raise ValueError(f"checksum mismatch for {relative_path.as_posix()}")


def _fetch_metadata(repository: str, endpoint: str, *, paginate: bool) -> Any:
    command = ["gh", "api", "-H", API_VERSION]
    if paginate:
        command.extend(("--paginate", "--slurp"))
    command.append(endpoint.format(repository=repository))
    try:
        return json.loads(_run(command))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GitHub returned invalid JSON for {endpoint}") from exc


def _release_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _release_records(item)


def _download_release_assets(
    repository: str,
    releases: Any,
    destination: Path,
) -> None:
    for release in _release_records(releases):
        tag = release.get("tag_name")
        assets = release.get("assets")
        if not isinstance(tag, str) or not isinstance(assets, list) or not assets:
            continue
        target = destination / tag
        target.mkdir(parents=True, exist_ok=False)
        _run(
            [
                "gh",
                "release",
                "download",
                tag,
                "--repo",
                repository,
                "--dir",
                str(target),
            ]
        )


def create_backup(
    destination: Path,
    repository: str,
    *,
    include_release_assets: bool,
) -> Path:
    """Create a timestamped, non-overwriting backup in ``destination``."""

    destination = _require_external_directory(destination)
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_root = destination / f"{repository.replace('/', '-')}-{timestamp}"
    if backup_root.exists():
        raise ValueError(f"refusing to overwrite existing backup: {backup_root}")
    backup_root.mkdir()

    mirror_path = backup_root / "repository.git"
    _run(["git", "clone", "--mirror", f"https://github.com/{repository}.git", str(mirror_path)])
    main_commit = _run(
        ["git", "--git-dir", str(mirror_path), "rev-parse", "refs/heads/main"]
    ).strip()

    metadata_root = backup_root / "github-metadata"
    metadata_root.mkdir()
    metadata: dict[str, Any] = {}
    for name, (endpoint, paginate) in METADATA_ENDPOINTS.items():
        value = _fetch_metadata(repository, endpoint, paginate=paginate)
        metadata[name] = value
        _write_json(metadata_root / f"{name}.json", value)

    if include_release_assets:
        _download_release_assets(
            repository,
            metadata["releases"],
            backup_root / "release-assets",
        )

    (backup_root / "README.txt").write_text(
        "This backup contains a bare Git mirror and non-secret GitHub metadata.\n"
        "It intentionally excludes Actions and environment secrets, tokens, and\n"
        "webhook configuration. See docs/backup-and-restore.md for recovery steps.\n",
        encoding="utf-8",
    )
    _write_json(
        backup_root / "backup-manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "include_release_assets": include_release_assets,
            "main_commit": main_commit,
            "metadata_directory": "github-metadata",
            "mirror_directory": "repository.git",
            "repository": repository,
        },
    )
    write_checksums(backup_root)
    return backup_root


def validate_backup(backup_root: Path) -> dict[str, Any]:
    """Validate hashes and ensure the mirrored main ref matches the manifest."""

    backup_root = backup_root.resolve()
    if not backup_root.is_dir():
        raise ValueError(f"backup directory does not exist: {backup_root}")
    validate_checksums(backup_root)
    try:
        manifest = json.loads((backup_root / "backup-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("backup has an unreadable manifest") from exc

    mirror_directory = manifest.get("mirror_directory")
    main_commit = manifest.get("main_commit")
    if not isinstance(mirror_directory, str) or not isinstance(main_commit, str):
        raise ValueError("backup manifest has no valid mirror/main commit")
    mirror_path = backup_root / _safe_relative_path(mirror_directory)
    mirrored_commit = _run(
        ["git", "--git-dir", str(mirror_path), "rev-parse", "refs/heads/main"]
    ).strip()
    if mirrored_commit != main_commit:
        raise ValueError("backup mirror main ref does not match the manifest")
    return manifest


def restore_backup(
    backup_root: Path,
    destination: Path,
    *,
    run_contract: bool,
) -> Path:
    """Restore a validated mirror into a fresh checkout without modifying it."""

    manifest = validate_backup(backup_root)
    destination = destination.resolve()
    try:
        destination.relative_to(backup_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("restore destination must be outside the backup directory")
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing restore destination: {destination}")

    mirror_path = backup_root.resolve() / _safe_relative_path(manifest["mirror_directory"])
    _run(["git", "clone", str(mirror_path), str(destination)])
    restored_commit = _run(["git", "-C", str(destination), "rev-parse", "HEAD"]).strip()
    if restored_commit != manifest["main_commit"]:
        raise ValueError("restored checkout does not match the backup manifest")
    if run_contract:
        _run(["uv", "sync", "--locked", "--extra", "dev"], cwd=destination)
        _run(["uv", "run", "python", "scripts/verify_repo.py"], cwd=destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="create a timestamped backup")
    create.add_argument("--destination", required=True, type=Path)
    create.add_argument("--repository", default=DEFAULT_REPOSITORY)
    create.add_argument("--include-release-assets", action="store_true")

    verify = commands.add_parser("verify", help="verify an existing backup")
    verify.add_argument("--backup", required=True, type=Path)

    restore = commands.add_parser("restore", help="restore into a fresh checkout")
    restore.add_argument("--backup", required=True, type=Path)
    restore.add_argument("--destination", required=True, type=Path)
    restore.add_argument("--run-contract", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "create":
            backup = create_backup(
                args.destination,
                args.repository,
                include_release_assets=args.include_release_assets,
            )
            print(f"Backup created: {backup}")
        elif args.command == "verify":
            manifest = validate_backup(args.backup)
            print(f"Backup verified: {args.backup.resolve()} ({manifest['main_commit']})")
        else:
            restored = restore_backup(
                args.backup,
                args.destination,
                run_contract=args.run_contract,
            )
            print(f"Backup restored: {restored}")
    except (RuntimeError, ValueError) as exc:
        print(f"backup error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
