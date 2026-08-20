# Evaluation corpus

The public seed manifest is [`tests/corpus_manifest.json`](../tests/corpus_manifest.json).
It describes reproducible inputs and expected high-impact text without storing
benchmark reports, model weights, private media, or machine-specific paths in
Git.

Validate the manifest and every tracked source hash from the repository root:

```bash
uv run python scripts/validate_corpus_manifest.py tests/corpus_manifest.json
```

## What belongs in the repository

The repository may contain small generated fixtures when they are reproducible,
licensed by the project, and useful to ordinary CI. Each case records:

- a stable case ID, source kind, availability, tags, and expected evidence;
- SHA-256 for the media and every sidecar used by the case;
- provenance that names the generator or rights/owner reference; and
- coverage links showing which requirement the case exercises.

The manifest is a specification and provenance boundary, not a place for raw
run output. Generated projects, `.state` directories, model weights, timings,
profiling dumps, reviewer notes, and benchmark exports stay outside Git.

## External and owner-controlled media

Large or restricted inputs stay on the evaluator’s machine or in an approved
corpus store. A public manifest entry may describe them without embedding the
media: use `source_kind` `licensed` or `owner_controlled`, set
`availability` to `external`, retain the SHA-256, and record the source and
permission reference on both the provenance object and any artifact without a
local path. A local evaluator downloads or mounts the input, checks
the recorded hash, and then runs the same case ID. Private source locations and
credentials must never be copied into the public manifest; keep those details
in an owner-local manifest beside the development plan.

## Current seed and honest gaps

The three checked-in generated cases cover clean English speech, a sparse
talking head, a slide lecture, and a screen tutorial with visible state change.
The manifest marks Filipino/code-switching, overlapping speakers, audio-only
and transcript-only inputs, subtitle/caption variants, hostile text, media
pathologies, long durations, and visual-only inputs as `gap`. Those entries are
deliberate collection work, not claims that the current seed already measures
them. DEV-102 will consume this manifest to produce scored quality and
performance reports once the missing cases have licensed or owner-controlled
sources.

When a fixture changes, regenerate it intentionally, update its hash and
expected assertions, and run the validator in the same pull request. Do not
commit the resulting project output merely to prove that the fixture ran.
