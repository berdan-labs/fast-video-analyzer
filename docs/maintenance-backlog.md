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

## Owner decisions and external setup

- [ ] Appoint a second trusted human administrator with MFA and offline recovery
  codes. Only then require an approval and CODEOWNERS review on `main`.
- [ ] Configure the protected `pypi` environment and PyPI Trusted Publishing/OIDC
  before enabling a real package publication.
- [ ] Decide whether GitHub Discussions should be enabled for support and
  announcements.
- [ ] Decide whether the currently empty Wiki should remain enabled; disable it
  if it will not be maintained.
- [ ] Establish an encrypted off-GitHub mirror and perform a restore drill into
  a fresh location.
- [ ] Decide whether a GitHub Project and release milestones add enough value to
  justify their ongoing maintenance.

## Planned engineering follow-ups

- [ ] Add a documented backup export and quarterly restore drill.
- [ ] Add a documented action allowlist after inventorying the exact actions used
  by the workflows and Dependabot.

## Recurring cadence

- [ ] Weekly: triage issues/PRs, inspect failed workflows, and review dependency
  update noise.
- [ ] Monthly: review access, Actions policy, dependencies, security findings,
  release readiness, and backup freshness.
- [ ] Quarterly: perform a restore drill, threat-model review, and maintainer
  succession check.
