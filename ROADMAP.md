# Roadmap

This roadmap summarizes durable product direction, not delivery dates or a
private owner work log. Detailed sequencing, experiments, raw benchmark
results, and owner-only decisions stay outside the Git working tree. Public
work that needs contributor context belongs in a focused issue or pull request.
The repository operating checklist lives in
[docs/maintenance-backlog.md](docs/maintenance-backlog.md).

## Current baseline

- `v0.1.0` is published on GitHub and PyPI through Trusted Publishing.
- The public pipeline produces one chronological evidence-linked Markdown
  project and retains strict validation, provenance, and review gates.
- Protected `main`, mandatory acceptance suites, CodeQL, cross-platform smoke
  tests, release provenance, and encrypted restore-tested backups are active.
- The repository intentionally uses one administrator and zero required
  approvals; no second administrator is planned.

## Now

- Keep the PyPI-first user journey, one version authority, and public
  CLI/API/schema contracts green as the product changes.
- Expand the manifest-driven evaluation corpus from the generated seed into
  licensed or owner-controlled cases without committing private media or raw
  benchmark output.
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
- Private planning notes, AI transcripts, raw benchmark output, or machine-
  specific working files committed as if they were product documentation.
