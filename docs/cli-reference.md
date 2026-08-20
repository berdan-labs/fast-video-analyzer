# CLI compatibility reference

`fast-video-analyzer` is the canonical console entrypoint. These two historical
names remain supported for existing scripts and automation:

For the complete status inventory—including exit codes, result fields, Python
imports, schemas, environment variables, and deprecation policy—see
[public-contracts.md](public-contracts.md). This page stays focused on
entrypoint and alias compatibility.

| Entry point | Compatibility status |
| --- | --- |
| `fast-video-analyzer` | Canonical |
| `long-video-analyzer` | Supported compatibility alias |
| `video-script-reconstructor` | Supported compatibility alias |

All three names expose the same commands, exit codes, JSON output, and offline
policy. For example:

```bash
fast-video-analyzer doctor --offline
long-video-analyzer doctor --offline
video-script-reconstructor doctor --offline
```

Nested aliases are retained where a historical workflow used a longer command:

- `review bundle create-all` (canonical)
- `review bundle batch-create` (compatibility alias)
- `review bundle create-batch` (compatibility alias)

The installed-wheel packaging suite invokes every top-level entrypoint outside
the repository checkout, and unit tests parse every nested alias. Add a new
alias only when it is documented here and covered by those tests.
