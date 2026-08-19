# Maintainers

## Current owner

- GitHub: [@vinnce143](https://github.com/vinnce143)
- Repository administration, security response, release approval, and
  emergency branch-protection bypass remain human responsibilities.

## Maintainer model

Until a second trusted maintainer is appointed, automation may label and
acknowledge issues but must not approve its own changes, merge pull requests,
publish packages, alter access, or bypass required checks.

The next continuity milestone is a recovery-capable second human administrator
with hardware-key or passkey MFA and offline recovery codes. Do not place
personal tokens or credentials in workflows.

## High-risk paths

Changes to security boundaries, provider integrations, pipeline orchestration,
serialized schemas, package metadata, lockfiles, and GitHub workflows require
explicit owner review through CODEOWNERS.
