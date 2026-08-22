# Performance benchmarking

The public workload matrix is [`tests/performance_manifest.json`](../tests/performance_manifest.json).
It freezes comparable journeys and relative regression budgets without
committing host-specific timings, hardware identifiers, model paths, or raw
benchmark output.

Validate the matrix from the repository root:

```bash
uv run python scripts/validate_performance_manifest.py tests/performance_manifest.json
```

## What is frozen

The matrix covers two short generated journeys: subtitle-led caption parsing
and a slide visual change. Each journey is defined for three owner-controlled
profiles:

- `cpu-standard` — portable CPU execution with the checked-in generated media;
- `cuda-midrange` — optional CUDA plus faster-whisper large-v3; and
- `storage-constrained` — an explicitly limited output/cache volume and free
  space reserve.

Each journey has cold, warm/resume, independent-validation, and batch
scenarios. A profile or model that is unavailable is reported as unavailable;
it never becomes a passing benchmark result. The current seed is deliberately
short. Long-duration, real-ASR, OCR, multilingual, overlap, and pathology
workloads remain collection work in the corpus manifest.

## Running owner-local measurements

Use the existing benchmark harness with the exact media and sidecars named by
the corpus case. Keep the output root outside the repository, for example:

```bash
uv run python scripts/benchmark_pipeline.py \
  tests/fixtures/generated/caption-variants.mp4 \
  --workload-id short-subtitle-led \
  --subtitle tests/fixtures/generated/caption-variants.vtt \
  --subtitle tests/fixtures/generated/caption-variants.ass \
  --output ../fast-video-analyzer-owner/performance/dev-301/cpu-short-cold \
  --preset strict --vision-mode none --no-resume --repeat 3
```

`--no-resume` makes the project output cold: each iteration starts without
that project's checkpoints. It does not disable the user-level shared ASR,
visual-frame, or OCR caches, so the result can still be faster than a first run
on a new host. For a cache-cold baseline, disable those accelerators explicitly
for the benchmark process and keep the resulting report separate:

```powershell
$env:VSR_DISABLE_ASR_SHARED_CACHE = "1"
$env:VSR_DISABLE_VISUAL_SHARED_CACHE = "1"
uv run python scripts/benchmark_pipeline.py `
  tests/fixtures/generated/caption-variants.mp4 `
  --subtitle tests/fixtures/generated/caption-variants.vtt `
  --subtitle tests/fixtures/generated/caption-variants.ass `
  --output ../fast-video-analyzer-owner/performance/dev-301/cpu-short-cache-cold `
  --preset strict --vision-mode none --no-resume --repeat 3
```

Record whether shared caches were enabled beside every owner-local report.
Never compare a cache-cold p95 with a shared-cache-warm p95 as if they were the
same workload.

For the warm/resume journey, reuse one output root and omit `--no-resume`:

```bash
uv run python scripts/benchmark_pipeline.py \
  tests/fixtures/generated/caption-variants.mp4 \
  --subtitle tests/fixtures/generated/caption-variants.vtt \
  --subtitle tests/fixtures/generated/caption-variants.ass \
  --output ../fast-video-analyzer-owner/performance/dev-301/cpu-short-warm \
  --preset strict --vision-mode none --repeat 3
```

Add `--independent-validation` when measuring the separate public validation
pass. The batch scenario uses the public `fast-video-analyzer batch` command
against a temporary source directory containing only the selected corpus
inputs; keep that directory and its output owner-local. The matrix marks this
scenario as requiring `faster-whisper-large-v3` because batch sidecar
comparison is an explicit model-dependent path. If that capability is absent,
record `unavailable`/`blocked` with the diagnostic rather than treating the
batch result as a passing subtitle benchmark.

Record the model revision, FFmpeg version, Python/runtime version, operating
system, hardware profile, cold/warm state, corpus hash, run configuration, and
output hash beside the raw report in the owner-controlled performance
directory. The benchmark report exposes wall time, stage timings, whether
visual survey overlapped ASR, peak RSS, output bytes, file count, validation
source, and cache reuse. It also records a sanitized runtime block (no GPU
serials, UUIDs, usernames, or host paths) and stable SHA-256 digests for the
transcript, frame, OCR, script-block, timeline, visual-event, and evidence
metadata lanes. `quality_contract_sha256` combines those lane digests with the
measured media duration and model-summary digest so two named-hardware runs can
be compared without publishing their raw project data. Stage times may
overlap; their sum is not wall time.

These fields are qualification evidence, not an SLO declaration. A hardware
tier may be called qualified only after owner-local, cache-cold, end-to-end
receipts cover the required duration and content classes, finish inside the
stated wall-time budget, pass validation, and match the accepted quality
contract. Keep the media, reports, profiler traces, host inventory, and exact
timings outside Git. Commit only the reusable harness, manifest rules,
generated fixtures, tests, and public methodology.

Pipeline benchmark reports use schema `1.1` and `report_kind` set to
`pipeline-benchmark`; ASR chunk sweeps use the same schema with
`report_kind=asr-chunk-sweep`.

## Owner-local qualification

When a five-hour corpus and a named host are genuinely ready, create the policy
outside Git and evaluate the resulting cold reports:

```bash
uv run python scripts/qualify_benchmark.py \
  --policy ../fast-video-analyzer-owner/qualification/dense-five-hour.json \
  --report ../fast-video-analyzer-owner/qualification/run-001.json
```

The evaluator exits `0` only when the policy schema, workload ID, report schema,
cache state, validation, duration floor, exact quality contract, required lane
digests, elapsed time, and p95 time all pass. It exits non-zero for malformed or
missing evidence; no absent field defaults to success. The committed
`tests/qualification_policy.example.json` is intentionally impossible to
satisfy and is a schema example, not a performance claim. Keep real policies,
reports, media, and host receipts owner-local.

## Budgets and comparison rules

The matrix uses relative budgets because absolute seconds are not portable:

- p95 wall time: no unexplained regression above 10%;
- peak RSS: no unexplained regression above 15%;
- output bytes: no unexplained growth above 10%; and
- quality and validation: must remain green for the same corpus case.

Compare only the same workload, hardware profile, model revision, runtime
family, preset, and corpus hash. A quality improvement may exceed a budget only
when the pull request records the measured trade-off and the owner accepts it.
Do not fill these relative policies with a convenient host timing merely to
make a gate pass. Host baselines belong in the owner-local performance store or
in sanitized release evidence, not in Git.
