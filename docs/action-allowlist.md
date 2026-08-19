# GitHub Actions allowlist

This is the repository's explicit action supply-chain boundary. Every action
used by a workflow must be one of the repositories below and must be pinned to
a full 40-character commit SHA. `scripts/verify_repo.py` enforces both rules;
the GitHub repository setting independently requires SHA pinning as defense in
depth.

| Action repository | Current pinned commit | Purpose |
| --- | --- | --- |
| `actions/attest-build-provenance` | `e8998f949152b193b063cb0ec769d69d929409be` | Release provenance attestations |
| `actions/checkout` | `3d3c42e5aac5ba805825da76410c181273ba90b1` | Read-only source checkout |
| `actions/download-artifact` | `3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c` | Release artifact handoff |
| `actions/setup-python` | `5fda3b95a4ea91299a34e894583c3862153e4b97` | Supported Python runtimes |
| `actions/stale` | `4391f3da665fdf50b6810c1a66712fb9ba21aa93` | Stale issue/PR maintenance |
| `actions/upload-artifact` | `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a` | Release artifact handoff |
| `astral-sh/setup-uv` | `37802adc94f370d6bfd71619e3f0bf239e1f3b78` | Locked Python environment setup |
| `github/codeql-action/analyze` | `ff2f1c621b7f889edc0d3c761ac2e6a3f8cdb0dd` | CodeQL v4 analysis |
| `github/codeql-action/init` | `ff2f1c621b7f889edc0d3c761ac2e6a3f8cdb0dd` | CodeQL v4 initialization |
| `pypa/gh-action-pypi-publish` | `dc37677b2e1c63e2034f94d8a5b11f265b73ba33` | Human-gated PyPI publishing |

Dependabot may update a pinned commit when upstream releases a new version;
the workflow and this inventory must change together in the same pull request.
The verifier requires the repository to remain in this allowlist and requires
the workflow reference to remain a full SHA.

## Change procedure

1. Confirm the upstream action, release notes, permissions, and intended input
   behavior.
2. Update the workflow to a full commit SHA and retain a human-readable tag
   comment.
3. Update this inventory in the same pull request.
4. Require the protected `required` check and inspect the resulting run before
   merge.

No workflow uses an unpinned third-party action. The pinned FFmpeg release in
CI is downloaded as a checksum-verified artifact by shell commands, not as a
GitHub Action.
