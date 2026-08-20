# Maintainer backlog

This is the living implementation checklist for the repository owner operating
system. Checked items are present on `main`; unchecked items require an owner
decision, an external account, or a future focused change. Review this file
weekly and update it through a pull request.

Last reviewed: 2026-08-20

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
- [x] Add an evidence-first owner-operations prompt that encodes the
  single-administrator policy, safe mutation boundaries, and completion audit.

## Owner decisions and external setup

- [ ] Register PyPI's pending (or existing-project) Trusted Publisher for
  `fast-video-analyzer`; GitHub's branch-restricted `pypi` environment and OIDC
  workflow are ready, but the PyPI account step cannot be performed here.
- [ ] Choose encrypted external backup storage, create the first off-GitHub
  mirror, and record a successful restore drill.

## Planned engineering follow-ups

- [x] Add a documented backup export and quarterly restore drill.

## Recurring cadence

- [ ] Weekly: triage issues/PRs, inspect failed workflows, and review dependency
  update noise.
- [ ] Monthly: review access, Actions policy, dependencies, security findings,
  release readiness, and backup freshness.
- [ ] Quarterly: perform a restore drill, threat-model review, and maintainer
  succession check.
