# GitHub operations

This document is the desired control-plane state for
`berdan-labs/fast-video-analyzer`. GitHub settings are changed by a human
administrator and should be verified with read-only `gh` commands after each
change.

## Branch policy

`main` should require a pull request, prevent force-pushes and deletion, require
resolved conversations, and require the stable aggregate `required` CI check.
Once a second trusted maintainer exists, require one approval and CODEOWNERS
review for high-risk paths. Keep a documented break-glass path: incident issue,
minimal signed fix, and a follow-up review/postmortem.

Use squash merging and delete merged head branches. Do not enable strict review
requirements before CI is reliable and a recovery-capable second human exists.

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

Use hardware-key or passkey MFA, offline recovery codes, least-privilege
fine-grained credentials, and at least two trusted human administrators before
delegating critical access. Review collaborators, deploy keys, Apps, webhooks,
environments, and trusted publishers quarterly. Never store personal tokens in
workflows.
