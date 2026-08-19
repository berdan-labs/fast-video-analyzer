## What changed

Describe the user-visible or maintainer-visible change and why it is needed.

## Verification

- [ ] `uv sync --locked --extra dev`
- [ ] `uv run ruff format --check scripts/verify_repo.py` (and any changed files)
- [ ] `uv run ruff check src tests scripts`
- [ ] `uv run mypy src/video_script_reconstructor`
- [ ] Relevant unit/integration/e2e/mutation/packaging checks passed
- [ ] `uv run python scripts/verify_repo.py`

## Contract and safety review

- [ ] Public CLI/API behavior and compatibility impact are documented.
- [ ] Serialized schema or `.state` migration impact is documented.
- [ ] Offline-by-default and external-provider boundaries are preserved.
- [ ] No credentials, source media, model weights, generated reports, or `.state` directories are included.
- [ ] User-facing documentation, examples, and changelog entries were updated when needed.

## Related issue

Fixes #
