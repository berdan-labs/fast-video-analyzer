# Security policy

Please do not disclose credentials, private media, model weights, or complete
generated project directories in a public issue. Send security-sensitive
reports privately through the repository's GitHub security advisory workflow:

<https://github.com/berdan-labs/fast-video-analyzer/security/advisories/new>

If private vulnerability reporting is unavailable, contact the repository owner
through GitHub before publishing technical details. Include a minimal
reproduction and impact description, but redact credentials, source media,
transcripts, screenshots, model weights, and generated project directories.

The maintainer will acknowledge a report when practical, assess the affected
versions and trust boundary, coordinate a fix privately, and publish a release
or mitigation once it is safe to disclose. Do not use public issues for leaked
secrets or suspected supply-chain compromise.

Fast Video Analyzer is designed for local, auditable processing. Remote media
downloads, external AI providers, and model installation are explicit opt-in
actions; reports should be inspected before sharing because transcripts and
screenshots can contain sensitive information.

Supported releases are the latest `main` commit and the newest tagged release.
The project is local-first: remote media downloads, external AI providers, and
model installation are explicit opt-in actions. Reports and screenshots may
contain sensitive information and should be inspected before sharing.
