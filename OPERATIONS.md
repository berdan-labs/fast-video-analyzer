# Operations

Fast Video Analyzer is local-first and treats subtitles, OCR text, media
metadata, generated Markdown, screenshots, model files, and external provider
responses as untrusted or potentially sensitive data.

## Operating principles

- Every change starts with an issue or documented maintenance task and lands
  through a pull request.
- The default branch is protected only after required CI checks are reliable.
- Automation can inspect, label, test, and draft; a human owner approves access,
  security, releases, publication, deletion, and emergency bypasses.
- Generated reports, model weights, source media, credentials, and `.state`
  directories stay outside version control.
- Releases must be traceable to a protected version tag and verified from a
  clean installation.

See [docs/owner-operations-prompt.md](docs/owner-operations-prompt.md),
[docs/github-operations.md](docs/github-operations.md),
[docs/releasing.md](docs/releasing.md), and [docs/runbooks.md](docs/runbooks.md)
for the operational procedures.
