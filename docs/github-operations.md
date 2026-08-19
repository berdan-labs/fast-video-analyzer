# GitHub operations

This document is the desired control-plane state for
`berdan-labs/fast-video-analyzer`. GitHub settings are changed by a human
administrator and should be verified with read-only `gh` commands after each
change.

## Branch policy

`main` requires a pull request, prevents force-pushes and deletion, requires
resolved conversations, and requires the stable aggregate `required` CI check.
The deliberate single-administrator policy requires zero approving reviews: a
nonzero quorum would be impossible to satisfy. CODEOWNERS remains an ownership
map and an explicit owner-review prompt for high-risk paths, not an automated
second-person gate. Keep a documented break-glass path: incident issue, minimal
signed fix, and a follow-up review/postmortem.

Use squash merging and delete merged head branches. Do not enable an
unfulfillable approval or CODEOWNERS-review requirement while the repository is
intentionally single-admin.

## Read-only heartbeat

```powershell
gh repo view berdan-labs/fast-video-analyzer --json nameWithOwner,visibility,defaultBranchRef,viewerPermission,viewerCanAdminister
gh run list --repo berdan-labs/fast-video-analyzer --limit 20
gh issue list --repo berdan-labs/fast-video-analyzer --state open --limit 50
gh pr list --repo berdan-labs/fast-video-analyzer --state open --limit 50
gh release list --repo berdan-labs/fast-video-analyzer --limit 20
gh api repos/berdan-labs/fast-video-analyzer/rulesets
gh api repos/berdan-labs/fast-video-analyzer/actions/permissions
gh api repos/berdan-labs/fast-video-analyzer/branches/main/protection
```

Endpoints that return `404`, `403`, or a scope error are evidence of an access
or configuration gap, not permission to broaden credentials casually.

## Access and continuity

Use hardware-key or passkey MFA, offline recovery codes, and least-privilege
fine-grained credentials. This repository deliberately has one human
administrator; that owner accepts the continuity risk and must keep the recovery
codes and an encrypted off-GitHub backup separately. Review collaborators,
deploy keys, Apps, webhooks, environments, and trusted publishers quarterly.
Never store personal tokens in workflows.
