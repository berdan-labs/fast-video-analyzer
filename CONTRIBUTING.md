# Contributing to Fast Video Analyzer

Thanks for helping improve Fast Video Analyzer. Keep user-facing documentation,
examples, and integrations aligned with the `fast-video-analyzer` command.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the project checks before opening a pull request:

```powershell
python -m pytest tests/unit -q
python -m pytest tests/integration -q
ruff check src tests
mypy src/video_script_reconstructor
```

Keep generated reports, benchmark roots, model weights, and local logs outside
the repository. Tests should use `tmp_path` or the maintained fixtures under
`tests/fixtures/generated`.

## Pull requests

Describe user-visible behavior, include a focused regression test, and call out
changes to serialized project schemas or image metadata. Do not commit source
media, credentials, model weights, or generated `.state` directories.
