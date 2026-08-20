# Changelog

All notable changes to Fast Video Analyzer are documented here. The project
uses a Keep-a-Changelog-style format and follows SemVer-compatible release
intent.

## [Unreleased]

### Added

- Deterministic corpus coverage for standalone audio with a subtitle sidecar
  and transcript-only SRT input. These cases validate modality routing and
  parsing; they do not claim real-model accuracy.

### Changed

- Long-running native ASR chunks now emit owner-local progress heartbeats so a
  blocked decoder is distinguishable from a dead process; heartbeats never
  imply chunk completion or alter transcript output.

## [0.2.0] - 2026-08-20

### Added

- A small synchronous typed Python facade for planning, running, validating,
  and inspecting review items without exposing pipeline internals.
- Installed-wheel coverage proving the facade and `py.typed` marker work
  outside the repository checkout.

### Changed

- The public Python contract now documents the stable `video_script_reconstructor.api`
  import surface, immutable result snapshots, offline defaults, and exception
  semantics.
- The release candidate is backed by the controlled seven-lane model audit and
  five-case deterministic corpus gate; the exact sanitized evidence will be
  attached to the tagged GitHub release.

## [0.1.1] - 2026-08-20

### Added

- Opt-in `doctor --summary` and `plan --summary` commands for concise,
  actionable capability setup and copyable run/validate/review handoff.
- A no-copy review workflow runbook covering review listing, bounded bundles,
  response application, validation, and finalization.
- Deterministic corpus/performance manifests and a controlled release-candidate
  model-audit procedure whose private reports remain outside the repository.
- Durable interruption recovery records for transcript, finalization, semantic
  ledger, and visual-evidence stages.

### Changed

- Expanded caption and hostile-subtitle coverage, including WebVTT and ASS
  candidates, without weakening evidence or review requirements.
- Clarified the public contract and owner-local publication boundary in the
  README and maintainer documentation.

## [0.1.0] - 2026-08-20

### Added

- Maintainer operating documentation, ownership rules, issue forms, and
  pull-request safety checks.
- Locked CI gates for quality and the mandatory acceptance suites.
- CodeQL, dependency auditing, Dependabot configuration, and a protected
  release workflow.

### Changed

- CI now uses the committed `uv.lock` and cancels superseded runs.
- New-issue triage uses the built-in GitHub CLI with least-privilege workflow
  permissions.
