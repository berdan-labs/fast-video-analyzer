# Roadmap

This roadmap records direction, not a promise of delivery dates.

The executable owner checklist lives in
[docs/maintenance-backlog.md](docs/maintenance-backlog.md).

## Now

- Make locked CI, release, security, and maintainer runbooks reliable.
- Protect `main` after the required aggregate CI check has proved stable.
- Establish a second recovery-capable human maintainer.

## Planned

- Add Windows and macOS smoke coverage on a scheduled cadence.
- Publish a first release through a verified Trusted Publishing environment.
- Add a documented backup export and quarterly restore drill.
- Expand examples and contract tests for public CLI aliases and schema changes.

## Exploring

- GitHub Discussions for user questions and announcements.
- A lightweight project board and quarterly release milestones.
- Signed release artifacts and a generated SBOM attached to each release.

## Not now

- A broad pipeline rewrite without a contract-preserving migration plan.
- Automatic merging or publishing without a human approval boundary.
- Mandatory model downloads on ordinary pull requests.
