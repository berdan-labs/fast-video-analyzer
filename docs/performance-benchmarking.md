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
  --subtitle tests/fixtures/generated/caption-variants.vtt \
  --subtitle tests/fixtures/generated/caption-variants.ass \
  --output ../fast-video-analyzer-owner/performance/dev-301/cpu-short-cold \
  --preset strict --vision-mode none --no-resume --repeat 3
```

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
source, and cache reuse. Stage times may overlap; their sum is not wall time.

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
