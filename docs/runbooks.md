# Operational runbooks

Every incident record should capture symptoms, safe diagnostics, data that may
be shared, recovery steps, escalation criteria, and verification of recovery.

## CI is stuck or flaky

Inspect `gh run list`, identify whether the failure is code, runner, network,
dependency, or timeout related, and preserve the run URL. Do not repeatedly
rerun an unknown failure. Fix the underlying workflow or dependency and verify
the aggregate `required` check before changing branch protection.

## Offline mode attempts network access

Stop the run, preserve sanitized logs, and inspect provider selection and
configuration. Confirm that external providers and model downloads are explicit
opt-ins. Do not upload source media or complete generated project directories.

To create a bounded support artifact, run:

```bash
fast-video-analyzer diagnostic-bundle --output fast-video-analyzer-diagnostic.zip
```

Inspect the archive before sharing it. The command deliberately includes only
sanitized runtime and offline-doctor metadata; it excludes media, transcripts,
screenshots, generated projects, credentials, and host filesystem paths.

## Corrupt state, cache, or insufficient disk

Work on a copy of the project output, inspect manifests and checksums, and use
the documented retention/cleanup commands. Never delete user media or `.state`
data without an explicit owner decision and a recoverable backup.

## Leaked credential or sensitive report

Treat the credential as compromised, revoke/rotate it through the owning
service, preserve only sanitized evidence, and use a private security advisory.
Do not paste the secret into an issue, commit, chat, or diagnostic bundle.

## Bad release

Stop publication, record the tag and artifact checksums, determine whether the
package must be yanked or superseded, and publish a corrected release only after
the clean-install and CLI smoke checks pass.

## Lost GitHub access

Use the owner's offline recovery codes. Do not create shared accounts or put
recovery credentials in the repository. After recovery, audit collaborators,
Apps, deploy keys, webhooks, environments, tokens, and branch protections.

## Restore drill

Quarterly, create and verify an encrypted off-GitHub backup, then restore it to
a fresh location with the commands in
[backup-and-restore.md](backup-and-restore.md). Confirm the restored commit,
maintenance contract, release documentation, and GitHub metadata snapshot. Any
secret, App, webhook, or PyPI configuration gap must be recorded and recreated
through its owning service rather than copied from the backup.
