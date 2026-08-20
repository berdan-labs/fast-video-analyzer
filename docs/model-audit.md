# Release-candidate model audit

The model audit is an owner-controlled release gate, not a pull-request test.
It exercises the optional local capabilities that ordinary CI must not download
or assume: Qwen3 ASR, Qwen3 forced alignment, MOSS diarization, PP-OCRv5,
Tesseract, faster-whisper large-v3, and the configured local semantic-vision
provider.

The public lane map is [`tests/model_audit_manifest.json`](../tests/model_audit_manifest.json).
It records stable test-module and model identities only. It contains no model
paths, credentials, source media, or benchmark output.

## Run it on the release machine

Use a clean checkout at the release-candidate commit and a fresh directory
outside the repository for every audit. The command never downloads weights;
prepare and verify optional models separately, then run offline:

```bash
uv sync --locked --extra dev
uv run python scripts/run_model_audit.py \
  --work-root ../fast-video-analyzer-audits/v0.2.0-2026-08-20 \
  --model-revision "owner-audit-v0.2.0" \
  --vision-mode local
```

The command runs `tests/model_dependent` once, executes the manifest-driven
corpus evaluator, and writes sanitized `model-audit.json` and
`model-audit.md` files. It also leaves the JUnit XML, generated projects, and
raw corpus report in the same owner-local work directory so a failure can be
investigated without putting private material in Git.

The checked-in seed baseline deliberately has no host timing values. To make
performance part of a release gate, supply an owner-local baseline with the
same corpus hash, scoring version, and model revision, then add
`--require-performance-baseline`. Never fill host timings into the public seed
baseline merely to make this command pass.

For the larger owner-controlled corpus, keep the lane map unchanged and pass
the private files explicitly:

```bash
uv run python scripts/run_model_audit.py \
  --work-root ../fast-video-analyzer-audits/v0.2.0-owner-corpus \
  --corpus-manifest ../fast-video-analyzer-corpus/manifest.json \
  --corpus-baseline ../fast-video-analyzer-corpus/baseline.json \
  --model-revision "owner-audit-v0.2.0" \
  --require-performance-baseline
```

The private baseline must use the exact canonical corpus hash and scoring
version of that manifest. The evaluator will reject a mismatched identity; the
audit runner will not silently substitute the public seed.

## Interpret the result

Each required lane is exactly one of:

- `pass`: every test in that lane ran and passed;
- `fail`: a collected test failed, the test was not collected, or the runner
  could not complete; or
- `unavailable`: the capability was explicitly skipped because its verified
  model, worker, executable, or configured provider was not available.

`unavailable` is useful information but is never a passing result. The overall
gate passes only when every required lane is `pass` and the corpus evaluator's
quality gate passes (plus the performance gate when requested). A scheduled
GitHub-hosted model-dependent run is an availability probe; it is not a
substitute for this controlled-hardware release artifact.

## Release evidence and publication boundary

Before a minor release, attach the exact `model-audit.json` and
`model-audit.md` produced from the tagged commit to the GitHub Release and link
the Markdown artifact from the release notes. Record the repository revision,
model revisions, corpus hash, runtime, and unavailable reasons. Do not attach
model weights, source media, generated projects, JUnit traces, local paths,
environment values, credentials, or raw worker logs.

If a lane is unavailable on the release hardware, either move the audit to a
machine that can run that capability or explicitly narrow the supported claim
and document the decision in the release notes. Do not describe an unavailable
lane as tested, and do not turn a skipped test into a green release gate.
