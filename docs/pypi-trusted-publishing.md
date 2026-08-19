# PyPI Trusted Publishing setup

The repository-side release job is ready for PyPI Trusted Publishing without a
long-lived token. It uses the `pypi` GitHub environment, job-scoped
`id-token: write`, and the pinned PyPA publishing action. The `pypi`
environment is restricted to protected branches; it has no reviewer rule because
this is deliberately a single-administrator repository.

## One-time PyPI account step

An owner of the PyPI account must register a GitHub Actions Trusted Publisher.
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
