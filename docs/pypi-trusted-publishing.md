# PyPI Trusted Publishing setup

The repository-side release job is ready for PyPI Trusted Publishing without a
long-lived token. It uses the `pypi` GitHub environment, job-scoped
`id-token: write`, and the pinned PyPA publishing action. The `pypi`
environment is restricted to protected branches; it has no reviewer rule because
this is deliberately a single-administrator repository.

## One-time PyPI account step

An owner of the PyPI account must first enable PyPI two-factor authentication
and safely retain the recovery codes. PyPI blocks the Publishing settings page
until 2FA is enabled; do not automate or handle the authenticator secret,
security key, or recovery codes. After the owner completes that security step,
register a GitHub Actions Trusted Publisher.
For a new project, use PyPI's **pending publisher** flow; for an existing
project, add the same publisher from that project's Publishing settings.

Use these exact values:

| Field | Value |
| --- | --- |
| PyPI project | `fast-video-analyzer` |
| GitHub owner | `berdan-labs` |
| Repository | `fast-video-analyzer` |
| Workflow file | `release.yml` |
| Environment | `pypi` |

## Current status

As of 2026-08-20, the pending publisher was registered with the exact values
above and the first OIDC publication completed successfully. The publisher is
now active for `fast-video-analyzer`; the package API reports version `0.1.0`,
and a clean virtual environment installed that exact version from PyPI and
passed `fast-video-analyzer --help` plus `doctor --offline`.

Evidence for the first publication:

- GitHub Actions run: `32341261708`
- GitHub Release: <https://github.com/berdan-labs/fast-video-analyzer/releases/tag/v0.1.0>
- PyPI release: <https://pypi.org/project/fast-video-analyzer/0.1.0/>
- PyPI JSON API: <https://pypi.org/pypi/fast-video-analyzer/0.1.0/json>

The successful OIDC upload is the authoritative activation evidence; the
account settings page may still require an interactive password confirmation
to display the publisher record.

PyPI pending publishers do not reserve the project name until the first
successful upload. Register it immediately before the first release and publish
without delay. Do not create or store a `PYPI_TOKEN` secret.

## First publication

1. Merge the release PR containing the version and changelog update.
2. Create and push the matching protected `vX.Y.Z` tag. The release workflow
   builds that tag, validates the distributions, creates the GitHub Release,
   attaches the SBOM, and creates provenance.
3. Confirm the GitHub Release artifact and SBOM before publication.
4. Use **Run workflow** for `Release` on protected `main`, enter that existing
   tag, and set **publish** to true. The build job checks out the supplied tag;
   it does not build the branch selected in the workflow-dispatch UI.
5. Confirm the new PyPI release, install the published wheel in a clean
   environment, and record the release URL and verification result.

If the upload is rejected, stop and compare the owner, repository, workflow
filename, environment, tag, and package name with the PyPI Trusted Publisher
record. Do not weaken the workflow or add a long-lived token to work around an
OIDC mismatch.
