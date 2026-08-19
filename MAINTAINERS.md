# Maintainers

## Current owner

- GitHub owner: [@berdan-labs](https://github.com/berdan-labs)
- Repository administration, security response, release approval, and
  emergency branch-protection bypass remain human responsibilities.

## Maintainer model

This is an intentional single-administrator repository. Automation may label and
acknowledge issues but must not approve its own changes, merge pull requests,
publish packages, alter access, or bypass required checks. The human owner uses
pull requests and required CI, keeps hardware-key or passkey MFA and offline
recovery codes, and maintains an encrypted off-GitHub backup. Do not place
personal tokens or credentials in workflows.

## High-risk paths

Changes to security boundaries, provider integrations, pipeline orchestration,
serialized schemas, package metadata, lockfiles, and GitHub workflows require
explicit owner review through CODEOWNERS.
