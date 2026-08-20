# Evaluation corpus

The public seed manifest is [`tests/corpus_manifest.json`](../tests/corpus_manifest.json).
It describes reproducible inputs and expected high-impact text without storing
benchmark reports, model weights, private media, or machine-specific paths in
Git.

Validate the manifest and every tracked source hash from the repository root:

```bash
uv run python scripts/validate_corpus_manifest.py tests/corpus_manifest.json
```

Run the generated seed and emit a local JSON/Markdown evaluation report:

```bash
uv run python scripts/evaluate_corpus.py --manifest tests/corpus_manifest.json --baseline tests/corpus_baseline.json --output-root path/to/local-corpus-output --report-json path/to/local-corpus-output/report.json --report-markdown path/to/local-corpus-output/report.md
```

The checked-in baseline contains quality targets and a 15% performance
regression policy. It deliberately has no elapsed-time numbers: timing is
host-dependent, so a release or performance audit must supply a baseline made
with the same model revision and request `--require-performance-baseline`.
The evaluator refuses a comparison when the scoring version, corpus hash, or
model revision differs.

## What belongs in the repository

The repository may contain small generated fixtures when they are reproducible,
licensed by the project, and useful to ordinary CI. Each case records:

- a stable case ID, source kind, availability, tags, and expected evidence;
- SHA-256 for the media and every sidecar used by the case (text sidecars are
  hashed with CRLF/CR normalized to LF so Windows and Unix checkouts agree);
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

The checked-in generated cases cover clean English speech, a sparse talking
head, a slide lecture, a screen tutorial with visible state change, WebVTT and
ASS caption candidates, and a hostile subtitle-text case containing Markdown,
HTML, a filesystem-looking path, and a shell-like command. The manifest still
marks Filipino/code-switching, overlapping speakers, audio-only and
transcript-only inputs, hostile OCR text, media pathologies, long durations,
and visual-only inputs as `gap`. The caption case proves format parsing and
auditable candidate selection only; it is not an ASR, OCR, multilingual, or
overlap benchmark. Those entries are deliberate collection work, not claims
that the current seed already measures them. The evaluator will produce scored
quality and performance reports for new cases once they have licensed or
owner-controlled sources; the deterministic seed evaluator runs all current
cases in ordinary CI.

When a fixture changes, regenerate it intentionally, update its hash and
expected assertions, and run the validator in the same pull request. Do not
commit the resulting project output merely to prove that the fixture ran.
