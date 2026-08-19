# Contributing to Fast Video Analyzer

Thanks for helping improve Fast Video Analyzer. Keep user-facing documentation,
examples, and integrations aligned with the `fast-video-analyzer` command.

## Development setup

```powershell
uv sync --locked --extra dev
```

The committed `uv.lock` is authoritative for development and CI. Python 3.12
is the default maintainer version; the CI matrix also verifies Python 3.10 and
3.11. FFmpeg and FFprobe must be available on `PATH` for media and end-to-end
checks.

Run the project checks before opening a pull request:

```powershell
uv run python scripts/verify_repo.py
uv run ruff format --check scripts/verify_repo.py
uv run ruff check src tests scripts
uv run mypy src/video_script_reconstructor
uv run pytest tests/unit tests/integration -q
uv run pytest tests/e2e tests/mutation tests/packaging -q
```

Keep generated reports, benchmark roots, model weights, and local logs outside
the repository. Tests should use `tmp_path` or the maintained fixtures under
`tests/fixtures/generated`.

## Pull requests

Describe user-visible behavior, include a focused regression test, and call out
changes to serialized project schemas or image metadata. Do not commit source
media, credentials, model weights, or generated `.state` directories.

Every change should be made on a focused branch and submitted through a pull
request. Changes to `.github/`, security/provider code, pipeline orchestration,
schemas, package metadata, lockfiles, or public output contracts require explicit
owner review through CODEOWNERS.
