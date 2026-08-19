# Changelog

All notable changes to Fast Video Analyzer are documented here. The project
uses a Keep-a-Changelog-style format and follows SemVer-compatible release
intent.

## [Unreleased]

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
