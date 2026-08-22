# Maintainer and agent handoff contract

This file is the first stop for any human or coding agent working in this
repository. It is an operating contract, not a project plan. The current Git
history, CI status, tests, and reviewed pull requests are authoritative; old
AI notes are not.

## Before changing anything

1. Read this file and `CONTRIBUTING.md`.
2. Inspect `git status --short`, the current branch, recent commits, and open
   pull requests before touching files.
3. Confirm ownership of the checkout. If another developer or agent has
   uncommitted work, do not reset, discard, rebase, or overwrite it. Use a
   separate branch/worktree or wait for an explicit handoff.
4. Treat media, model weights, credentials, machine-specific paths, raw
   benchmarks, and active plans as owner-local unless a maintainer deliberately
   promotes a small, reproducible conclusion into the repository.

## Working rules

- Make one focused change at a time on a named branch; never push directly to
  the default branch.
- Preserve backward-compatible defaults. Opt-in experiments need an explicit
  gate, a bounded fallback, cache/provenance separation, and a focused test.
- Do not invent benchmark results. Record commands, environment, input digest,
  quality checks, failures, and the next decision in an owner-local handoff.
- Do not put absolute local paths or private media references in tracked files.
- Keep generated state, logs, exports, and AI transcripts outside the checkout.
- Before handoff, leave the tree in a recoverable state and report: branch,
  commit, changed files, tests run, known risks, and the single next action.

## Validation and handoff

At minimum, run the checks relevant to the change, then:

```powershell
git diff --check
uv run python scripts/verify_repo.py
uv run ruff check src tests scripts
```

For orchestration, decoder, schema, or output changes, also run the focused
unit/integration tests and inspect the diff. A pull request is the review and
handoff boundary: summarize behavior, evidence, compatibility, and rollback.

When switching agents, the incoming agent must re-read this file, inspect the
working tree, and continue from evidence rather than replaying an old plan.
