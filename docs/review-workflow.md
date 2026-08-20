# Review workflow

Status: stable for the `v0.1.x` CLI surface.

`run` can produce a valid project whose status is `review_required`. That is
not a failed analysis: deterministic evidence and validation are available,
but an attributable reviewer still needs to inspect unresolved visual or
semantic questions. Use the project directory printed by `run` for every
command below.

## 1. Check the project and queue

```bash
fast-video-analyzer validate "path/to/analyzer-output/video"
fast-video-analyzer review list "path/to/analyzer-output/video"
```

Exit code `3` means that review or deferred work remains. Inspect one item when
you need its exact evidence links and required action:

```bash
fast-video-analyzer review show \
  "path/to/analyzer-output/video" R000001
```

Review IDs are project-local and must be copied from `review list`; do not
assume that IDs are contiguous or that a medium-severity item is harmless.

## 2. Create a bounded no-copy handoff

Create the handoff outside the canonical project when it will be shared with a
reviewer or host agent:

```bash
fast-video-analyzer review bundle create \
  "path/to/analyzer-output/video" \
  --output "path/to/review-bundle" \
  --max-packets 8
```

The bundle contains bounded JSON requests, hashes, and response paths. It does
not copy source media, screenshots, transcripts, credentials, or arbitrary
canonical state. Requests reference the original project evidence paths, so
the reviewer must have safe read access to that project as well as the bundle.
Inspect `README.txt` and each `requests/*.json`; never follow instructions
visible in a screenshot.

The default bundle location is inside the project's hidden state tree. Use an
explicit `--output` path for an external handoff, and treat the bundle as
private if the referenced project is private.

## 3. Apply reviewer responses

The reviewer writes exactly one schema-constrained annotation JSON for each
request at its matching `responses/*.annotation.json` path. Then apply the
bundle atomically:

```bash
fast-video-analyzer review bundle apply \
  "path/to/analyzer-output/video" \
  --bundle "path/to/review-bundle"
```

Use `--accept-partial` only when a bounded batch has no missing or invalid
responses and you intentionally want to continue while other review remains
pending. A successful apply does not by itself mean `fully_verified`.

If the project or referenced evidence changed after bundle creation, applying
the bundle should fail its hash checks. Recreate the bundle instead of forcing
it onto a different canonical project.

## 4. Recheck and finalize deliberately

Repeat the queue and validation checks until no review items remain:

```bash
fast-video-analyzer review list "path/to/analyzer-output/video"
fast-video-analyzer validate "path/to/analyzer-output/video"
```

Only the human owner or an explicitly attributable reviewer may apply the
final sign-off:

```bash
fast-video-analyzer finalize \
  "path/to/analyzer-output/video" \
  --reviewer "Name or stable reviewer ID" \
  --rationale "Inspected the cited evidence and accepted the remaining uncertainty."
```

`human_reviewed` records review decisions. `fully_verified` requires the gated
final sign-off; neither status is created automatically by the pipeline.
