# Maintainer backlog

This is the living implementation checklist for the repository owner operating
system. Checked items are present on `main`; unchecked items require an owner
decision, an external account, or a future focused change. Review this file
weekly and update it through a pull request.

Last reviewed: 2026-08-24

Review note (2026-08-24): No open issues or pull requests; `main` CI and
CodeQL passed after the exact-pixel deduplication optimization and its changelog
entry (`0a11f75`, `b94be91`); security and dependency alerts remain clear; the
five-hour performance qualification remains pending an authorized stronger
host. Owner-only benchmark receipts and hardware fingerprints remain outside
Git.

## Completed

- [x] Establish `uv.lock` as the authoritative development and CI lockfile.
- [x] Add Python 3.10-3.12 quality checks and mandatory acceptance suites.
- [x] Add timeouts, concurrency cancellation, pinned Actions, and a stable
  aggregate `required` status check.
- [x] Add CODEOWNERS, issue forms, PR checklist, support policy, conduct policy,
  changelog, roadmap, operations guide, and runbooks.
- [x] Replace unsafe issue triage with least-privilege GitHub CLI operations.
- [x] Add CodeQL, dependency auditing, Dependabot, stale maintenance, and a
  guarded release workflow.
- [x] Enable squash-only merges, merged-branch deletion, Dependabot security
  updates, private vulnerability reporting, and Actions SHA pinning.
- [x] Protect `main` from force-push/deletion and require the aggregate CI gate.
- [x] Validate clean wheel installation, CLI help, and offline doctor behavior.
- [x] Close the pre-existing triage test issue after verifying the workflow.
- [x] Add scheduled Ubuntu, Windows, and macOS CLI smoke coverage.
- [x] Enforce a documented allowlist of SHA-pinned GitHub Actions.
- [x] Add a sanitized diagnostic-bundle command with privacy tests.
- [x] Generate a CycloneDX runtime dependency SBOM and attach it to releases.
- [x] Expand public CLI examples and compatibility tests for all entrypoint aliases.
- [x] Adopt an explicit single-administrator policy with pull-request-only
  changes and zero required approvals; a second administrator is not required.
- [x] Keep GitHub Discussions disabled because issue forms and `SUPPORT.md`
  already provide the public support path.
- [x] Disable the unused GitHub Wiki so repository documentation has one
  reviewable, version-controlled home.
- [x] Decide not to use GitHub Projects or release milestones until concurrent
  work makes their maintenance value exceed their overhead.
- [x] Add a non-secret backup export, integrity verification, and restore-drill
  procedure.
- [x] Define an evidence-first owner-operations procedure that encodes the
  single-administrator policy, safe mutation boundaries, and completion audit;
  detailed agent prompt remains owner-local.
- [x] Separate durable public repository truth from owner-local planning,
  private triage, raw evaluation output, and machine-specific working files.
- [x] Make long-running native ASR calls observable with owner-local heartbeat
  receipts without adding automatic interruption, retry, or false-success
  behavior.

## Owner decisions and external setup

- [x] Register PyPI's pending Trusted Publisher for `fast-video-analyzer`;
  PyPI now shows the exact `berdan-labs` / `fast-video-analyzer` / `release.yml`
  / `pypi` entry, and GitHub's branch-restricted environment and OIDC workflow
  are ready.
- [x] Publish the first tagged release through the pending publisher, then
  verify the active PyPI publisher, package page, clean installation, and
  release URL (`v0.1.0`, GitHub Actions run `32341261708`).
- [x] Publish `v0.2.0` through the active Trusted Publisher, attach the exact
  sanitized model-audit evidence and SBOM to GitHub Releases, verify release
  attestations, and complete a clean PyPI install/dogfood journey (workflow
  `32371441983`).
- [x] Create and verify an off-GitHub encrypted mirror and a successful fresh
  restore drill; exact locations, timestamps, commits, and filesystem details
  remain in owner-local continuity records rather than this public backlog.
- [ ] Export and retain the current EFS certificate/key, or deliberately
  configure and test a recovery-agent certificate, in a separately protected
  owner-controlled location. Never place recovery material in Git.

## Planned engineering follow-ups

- [x] Add a documented backup export and quarterly restore drill.
- [x] Measure and document the no-copy review handoff from `review_required` to
  attributable final sign-off; keep the owner-local timing evidence outside Git.
- [x] Add opt-in path-free `doctor --summary` and copyable `plan --summary`
  output after measuring first-run setup friction.

## Recurring cadence

- [ ] Weekly: triage issues/PRs, inspect failed workflows, and review dependency
  update noise.
- [ ] Monthly: review access, Actions policy, dependencies, security findings,
  release readiness, and backup freshness.
- [ ] Quarterly: perform a restore drill, threat-model review, and maintainer
  succession check.
