# Backup and restore drill

The repository has one operational owner, so continuity depends on a tested
off-GitHub backup rather than an unfulfillable reviewer or administrator quorum.
Keep the backup parent directory in encrypted storage that is separate from the
repository checkout and the primary GitHub account.

## What the helper exports

`scripts/backup_repo.py` creates a timestamped, non-overwriting directory with:

- a bare Git mirror, including all reachable branches and tags;
- non-secret GitHub control-plane metadata: repository settings, main branch
  protection, Actions permissions, environments, rulesets, workflows,
  collaborators, deploy keys, labels, milestones, and release metadata;
- optional release assets; and
- a SHA-256 manifest that detects changed, missing, or unexpected files.

It deliberately does not export Actions or environment secret values, tokens,
webhook configuration, private deploy-key material, or PyPI configuration.
Those must be recreated through their owning service after recovery.

## Create a backup

From a clean checkout, choose an encrypted directory outside the repository and
run:

```powershell
uv run python scripts/backup_repo.py create `
  --destination "D:\Encrypted-Backups\fast-video-analyzer" `
  --include-release-assets
```

The script refuses a destination inside the checkout and never overwrites an
existing timestamped backup. Keep the resulting directory together with the
location's encryption/recovery instructions; do not commit it or place it in a
shared public drive.

## Quarterly restore drill

Perform this at least quarterly and after any material GitHub control-plane
change:

```powershell
uv run python scripts/backup_repo.py verify `
  --backup "D:\Encrypted-Backups\fast-video-analyzer\berdan-labs-fast-video-analyzer-YYYYMMDDTHHMMSSZ"

uv run python scripts/backup_repo.py restore `
  --backup "D:\Encrypted-Backups\fast-video-analyzer\berdan-labs-fast-video-analyzer-YYYYMMDDTHHMMSSZ" `
  --destination "D:\Restore-Drills\fast-video-analyzer" `
  --run-contract
```

The restore destination must be new and outside the backup directory. The
`--run-contract` option recreates the locked development environment and runs
the repository maintenance verifier. In the restored checkout, also run the
normal release-ready test commands from [releasing.md](releasing.md), inspect
the GitHub metadata snapshot, and record the date, restored commit, verifier
result, and any manual recovery gaps in the maintenance issue or PR.

## Recovery boundaries

The Git mirror can recreate source and tags. The metadata snapshot makes the
GitHub configuration auditable, but does not reapply it automatically. During a
real recovery, explicitly recreate branch protection, Actions permissions,
environments, secrets, webhook/App integrations, and the PyPI Trusted Publisher
only after reviewing the snapshot. Rotate any credentials involved in the
incident instead of restoring them from an old location.
