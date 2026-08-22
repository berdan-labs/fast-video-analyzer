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
# Run this command once for each arm (A, B, C) and round (1, 2, 3), replacing
# the angle-bracket placeholders with concrete owner-local values. Do not run
# the three arms concurrently: they must contend only with the same host
# baseline, not with one another.
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
  --report ../fast-video-analyzer-owner/qualification/run-001.json \
  --report ../fast-video-analyzer-owner/qualification/run-002.json \
  --report ../fast-video-analyzer-owner/qualification/run-003.json
```

The evaluator exits `0` only when the policy schema, workload ID, report schema,
cache state, validation, duration floor, exact quality contract, required lane
digests, elapsed time, and p95 time all pass. It exits non-zero for malformed or
missing evidence; no absent field defaults to success. The committed
`tests/qualification_policy.example.json` is intentionally impossible to
satisfy and is a schema example, not a performance claim. Keep real policies,
reports, media, and host receipts owner-local.

Two optional policy keys harden multi-report qualification without changing any
existing policy:

- `minimum_report_count` (default `1`) must be a positive integer. Evaluation
  fails closed when fewer report files are supplied than the policy demands.
- `required_runtime_fingerprint_fields` (default `[]`) lists `report.runtime`
  field names that must exist in every supplied report and be identical across
  all of them (for example sanitized host platform or CPU-model fields).
- `max_peak_rss_bytes` and `max_output_bytes` (unset by default) enforce the
  documented resource budgets against each report's
  `performance_summary.peak_rss_bytes` and `performance_summary.output_bytes`.
  When present, missing measurements or over-budget values reject the report;
  they never silently pass.

A real three-round hardware policy — one cold report per interleaved round —
should set `minimum_report_count` to `3`, so a single lucky run can never
qualify a host on its own.

## Five-hour end-to-end qualification experiment

No committed workload covers long-duration media yet: `duration-classes`
remains a corpus gap and both matrix journeys are short generated fixtures.
Before any scheduling default changes, run this paired, cache-cold A/B/C
experiment on one named CUDA host with one owner-controlled recording of at
least five hours. It uses only existing harness flags and environment
policies; media, policies, and reports stay owner-local.

Hypothesis: on a CUDA host with at least twelve logical CPUs, overlapping the
transcript-independent FFmpeg survey with local large-v3 ASR reduces cold
five-hour wall time by at least 10 percent without changing any quality lane
digest, slowing the ASR stage by more than 10 percent, or exceeding the RSS
and output-byte budgets. Exact-frame warmup (`VSR_PARALLEL_VISUAL_WARMUP`)
earns default consideration only if its arm also holds those gates.

Design: ASR-led input (no supplied subtitles or transcript), strict preset,
identical vision configuration in every arm, shared caches disabled, three
rounds of one cold iteration per arm in interleaved A, B, C order so thermal
and system drift spreads evenly across arms:

| Arm | `VSR_PARALLEL_VISUAL_SURVEY` | `VSR_PARALLEL_VISUAL_WARMUP` |
| --- | --- | --- |
| A (control) | `0` | off (default) |
| B (survey overlap) | `1` | off (default) |
| C (warmup overlap) | `1` | `on` |

```powershell
$env:VSR_DISABLE_ASR_SHARED_CACHE = "1"
$env:VSR_DISABLE_VISUAL_SHARED_CACHE = "1"
$env:VSR_DISABLE_SEMANTIC_SHARED_CACHE = "1"
$env:VSR_PARALLEL_VISUAL_SURVEY = "0"   # arm A; "1" for arms B and C
Remove-Item Env:VSR_PARALLEL_VISUAL_WARMUP -ErrorAction SilentlyContinue  # arms A/B
$env:VSR_PARALLEL_VISUAL_WARMUP = "on"  # arm C only
uv run python scripts/benchmark_pipeline.py `
  <owner-five-hour-media.mp4> `
  --workload-id dense-five-hour-survey-ab `
  --output ../fast-video-analyzer-owner/performance/five-hour/<arm>-<round> `
  --preset strict --no-resume --repeat 1
```

Record the runtime block beside every report. When arm C is enabled, record
the effective `VSR_PARALLEL_VISUAL_WARMUP_MAX_FRAMES` cap (default 1024)
because a five-hour survey can exceed it; raise it explicitly (bounded at
4096) rather than letting the cap silently truncate the treatment.

A companion compute-type matrix keeps the same three-arm shape on the same
named host and corpus: arm A remains the CUDA `float16` control, arm B applies
`VSR_FASTER_WHISPER_COMPUTE_TYPE=int8_float16` as the treatment, and arm C runs
plain `int8` as a secondary treatment. Every compute-type arm must pass the
same correctness, ASR-contention, RSS/output-byte, and timing gates above; no
compute-type result is recorded yet, and no precision default changes without
those owner-local receipts.

Correctness gates, applied to every iteration of every arm before any timing
is considered:

- `validation_valid` is true, status is not `blocked`, and
  `quality.available` is true with `blocking_failure_count` of zero;
- `quality_contract_sha256`, `ordered_text_sha256`, and all seven
  `lane_sha256` values are byte-identical across all arms and iterations
  (scheduling changes must be quality-invariant);
- arms B and C show `parallel_visual_survey: true` and a positive
  `survey_parallel_elapsed_seconds`, proving the overlap actually engaged;
- the ASR stage in arms B/C takes no more than 1.10 times arm A's ASR stage
  (GPU inference must not be starved by decoder contention);
- `peak_rss_bytes` no more than 1.15 times arm A and `output_bytes` no more
  than 1.10 times arm A.

Timing thresholds: every iteration's `elapsed_seconds` and the arm's
`timing_summary.p95_seconds` must be at most 900 seconds, evaluated through
an owner-local `qualify_benchmark.py` policy with `min_media_duration_s` of
18000 and the quality contract digest pinned from the accepted arm A run.

Pre-authorized outcomes, and nothing else:

- Arm A alone exceeds 900 seconds: change no defaults. Use
  `stage_elapsed_seconds` from the reports to name the dominant stage and
  design the next experiment there (most likely ASR checkpoint geometry via
  `--asr-chunk-sweep`).
- Arm B passes every gate with at least 10 percent median wall-time
  improvement over A: the authorized public change is documentation and,
  once owner media hashes exist, a `dense-five-hour` workload row bound to an
  external `owner_controlled` corpus case. No pipeline code changes.
- Arm C additionally passes every gate including the ASR-contention gate:
  the single authorized production change is flipping the `auto` branch of
  `_parallel_visual_warmup_enabled` in `pipeline.py` from `False` to `True`
  for automatic adapters on long media, with its unit tests updated. Any
  further default change requires its own measured matrix.

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
