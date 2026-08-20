# Roadmap

This roadmap summarizes product direction, not delivery dates. The executable
engineering sequence and acceptance gates live in
[docs/development-plan.md](docs/development-plan.md). The repository operating
checklist lives in [docs/maintenance-backlog.md](docs/maintenance-backlog.md).

## Current baseline

- `v0.1.0` is published on GitHub and PyPI through Trusted Publishing.
- The public pipeline produces one chronological evidence-linked Markdown
  project and retains strict validation, provenance, and review gates.
- Protected `main`, mandatory acceptance suites, CodeQL, cross-platform smoke
  tests, release provenance, and encrypted restore-tested backups are active.
- The repository intentionally uses one administrator and zero required
  approvals; no second administrator is planned.

## Now

- Make PyPI installation and the first successful user journey authoritative
  and continuously tested.
- Use one source of truth for package, CLI, and release versions.
- Define stable, provisional, and internal CLI/API/schema contracts.
- Build a manifest-driven evaluation corpus and quality/performance baseline.
- Establish a controlled release-candidate audit for real ASR, OCR,
  diarization, alignment, and semantic-review capabilities.

## Next

- Improve transcript, OCR, visual-event, and semantic-review quality based on
  measured corpus failures.
- Enforce performance, memory, storage, and resume-regression budgets.
- Extract bounded pipeline stages behind characterization tests instead of
  performing a broad rewrite.
- Improve capability setup and human review throughput while preserving the
  local-first evidence contract.
- Publish a small stable Python API before adding new integrations.

## Later

- Add thin agent/MCP integrations over stable public contracts when a real user
  journey justifies them.
- Consider a local review UI only if measured reviewer effort shows that the
  CLI and file-bundle workflow is the limiting factor.
- Stabilize schemas, migrations, platform/input support, and compatibility for
  a `1.0.0` release.

## Not now

- A broad pipeline rewrite.
- A hosted service or telemetry-by-default execution path.
- A second administrator or impossible review quorum.
- Automatic `fully_verified` status or automatic destructive cleanup.
- Mandatory model downloads on ordinary pull requests.
- New aliases, integrations, or UI surfaces without a measured user need.
