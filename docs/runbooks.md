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

Use offline recovery codes and the second human administrator or organization
owner. Do not create shared accounts or put recovery credentials in the
repository. After recovery, audit collaborators, Apps, deploy keys, webhooks,
environments, tokens, and branch protections.

## Restore drill

Quarterly, restore the Git mirror, releases, documentation, and GitHub metadata
into a fresh location. Confirm that a different maintainer can run the checks,
understand the operating docs, and produce a release without private context.
