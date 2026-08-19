# Pipeline contract

## Contents

- Stage order
- Source decisions
- Resume and invalidation
- Status handling

## Stage order

Run ingest and identity, lightweight probe, transcript discovery/validation, selective repair or ASR, timeline construction, video survey, candidate packets, deterministic metadata, semantic enrichment, reconstruction, audit, and atomic rendering in that order. Preserve partial state before returning review-required or blocked.

Optional OCR and semantic-vision adapters are resolved only for video inputs;
transcript- and audio-only runs do not import or inspect visual model backends,
and transcript-only cache keys do not inspect unused ASR model manifests.
Configuration instances remain JSON-Schema-validated on every load; the shipped
schema's pinned SHA-256 only skips redundant schema self-validation, and any
schema-resource edit falls back to the full defensive check.

Use deterministic chronology and mappings. Never use a language or vision model as proof of completeness. Start with one provisional block per substantive segment and group only when exact IDs, word order, and residual text remain recoverable. Emit important visual-only events as blocks.

## Source decisions

Prefer user human transcripts, user subtitles, human embedded tracks, official captions, other human captions, auto captions, then local ASR—but validate every candidate. Select or merge reliable intervals; never concatenate competing sources. Preserve raw sidecars under `.state/transcript/original/`.

Repair only suspect intervals with padded audio clips and globally offset timestamps. If no usable candidate exists, run only explicitly installed, hash-verified local workers. Prefer `faster-whisper` `large-v3` for Filipino/mixed-language speech (`--language fil` or `VSR_PREFER_WHISPER=1`); retry CPU/int8 when CUDA/cuBLAS cannot load. Qwen/MOSS/forced-alignment workers are legacy compatibility adapters and are not probed by ordinary runs. Rank usable candidates, disclose ordered-text and high-impact-token disagreement, and route material disagreement to review. Never download during a normal run or select a fixture adapter in production.
An existing complete offline large-v3 snapshot can be selected without a duplicate
3 GB copy by setting `VSR_FASTER_WHISPER_LARGE_V3_PATH`; the resolver validates
the five required files and includes their stat signatures in the automatic
ASR cache identity.
On CPU-only hosts, automatic faster-whisper construction uses a bounded physical-core approximation (up to eight threads) instead of all SMT threads; `VSR_ASR_CPU_THREADS` provides an explicit override (`0` restores backend auto-selection). On CUDA hosts, automatic large-v3 construction uses two CTranslate2 workers for media up to three minutes only when the visible GPU reports at least 8 GiB; smaller/unknown GPUs and long-form media stay at one worker. Override this scheduling choice with `VSR_FASTER_WHISPER_NUM_WORKERS=1..8`. Worker count is included in the checkpoint/cache identity and changes scheduling only; it does not alter decoder settings or transcript merge rules.
The local RTX 3060 benchmark produced identical transcript/timestamp digests with one or two workers: 120 seconds measured 14.082 s versus 12.700 s, while the 600-second direct decode measured 81.932 s versus 80.271 s and the complete pipeline measured 81.559 s versus 90.905 s. The duration guard intentionally keeps long-form unattended runs at one worker because end-to-end critical-path timing, not an isolated decoder sample, is the authority.
Standard faster-whisper decoding remains the accuracy-first default. For a measured throughput experiment, set `VSR_FASTER_WHISPER_BATCHED=1` (or `VSR_FASTER_WHISPER_INFERENCE_MODE=batched`) and optionally `VSR_FASTER_WHISPER_BATCH_SIZE` between 1 and 64. BatchedInferencePipeline shares the loaded weights, but its scheduling can change segment boundaries; the adapter records `inference_mode`, `batch_size`, and an explicit review warning, and checkpoint identity separates batched from standard output. Do not enable it silently in a strict run.
On the local FC101 Filipino benchmark (120 seconds, large-v3 CUDA/float16, batch size 4), standard decoding took 13.107 s and batched decoding 4.844 s (2.706x), while word counts were 248 and 249 and the text boundaries differed. Treat this as a throughput/review trade-off, not an accuracy claim; remeasure on representative media and keep standard mode for unattended strict output.
When no language is supplied, checkpointed Whisper chunks retain independent
per-chunk language detection by default. An explicit `VSR_ASR_LANGUAGE_HINT=1`
opt-in carries the first chunk's language forward only after a high-confidence
(>=0.90) detection; this can reduce detection work for Filipino recordings but
may change long-form decoding, so the strategy is included in ASR metadata and
the checkpoint/run cache keys.
When a faster-whisper checkpoint plan resolves to one chunk covering the complete
media span, the adapter uses an explicit full-media passthrough and decodes the
original container directly instead of creating an intermediate WAV. This is a
scheduling-only optimization: bounded multi-chunk requests and all other adapters
retain the extraction boundary, and the checkpoint records
`full_media_passthrough=true` for auditability.
After candidate selection, canonical `primary_language` is inferred from
dominant segment/word labels only when one language clearly outweighs the rest;
mixed or ambiguous audio remains `und`, so language reporting never becomes a
new unsupported claim.
When `transcript.compare_candidates=false` is explicitly configured, the
language-aware resolver probes only the first usable backend and skips absent
worker/model verification and unused adapter construction. The strict/default
comparison path still verifies and runs the independent candidate ensemble; the
shortcut is a policy-controlled performance optimization, not an accuracy
fallback.

Model verification hashes every recorded file on first use, then reuses a
stat-bound receipt only while the manifest and file size/timestamp/inode
signatures remain unchanged. This removes repeated multi-gigabyte verification
I/O from normal runs without weakening the explicit `models verify --full`
audit path.
Use `models report` for a read-only model-store footprint rollup; it ranks
complete on-disk bytes, separates verified/unverified storage, and emits only
explicit removal commands. `--with-workers` adds isolated-runtime readiness and
failure reasons, totals verified-but-worker-unavailable bytes, and labels legacy
packs while suggesting a current replacement without assuming that deletion is
safe. It never downloads or
removes weights.
Worker readiness is cached briefly against worker Python, manifest, and probe-module
signatures so repeated reports do not repeatedly start heavy isolated interpreters;
`workers verify` bypasses that cache for an explicit fresh check.

The offline doctor also reports the bounded worker/thread settings selected for
the host, including the validator metadata-verification pool. These are
scheduling telemetry only; use the benchmark harness before overriding them.

Independent metadata verification runs in a bounded pool
(`VSR_VALIDATOR_METADATA_WORKERS`, default up to sixteen on hosts with at least
sixteen logical CPUs). It
preserves strict per-image schema, embedded-payload, file-hash, and pixel checks
while retaining deterministic error ordering; set the override only after a
representative validation benchmark.

## Resume and invalidation

Key each stage by source/sidecar/config/code/tool/model/prompt inputs. Include pixels, base revision, observation inputs, prompt hash, and reconciliation rules in metadata-enrichment keys. Invalidate only dependent downstream stages. Before rebuilding a failed visual stage, rotate stale packets and generated images into timestamped visual history so old IDs cannot be revalidated as current evidence. Reconcile interrupted image/ledger transactions to the last internally consistent revision.

Full-ASR work is checkpointed in bounded overlapping windows. Valid chunk files
are reused byte-for-byte on resume; corrupted or mismatched chunks alone are
recomputed. Progress/ETA telemetry is observational and must never make an
otherwise valid evidence checkpoint unusable. The selected model and runtime
path remain explicit in the manifest, including an auditable CPU/int8 fallback
when CUDA/cuBLAS is unavailable.
Automatically resolved production ASR also mirrors each validated chunk into a
local content-addressed cache under the platform application-cache directory.
The cache key binds the immutable media digest, interval/chunk geometry,
language policy, backend/model, device, compute type, and decoding settings;
therefore a fresh output directory can reuse exact transcript bytes without
rerunning Whisper. A shared hit is copied into the project-local checkpoint
before merge and is recorded in `shared_cache_hit_indexes`. Set
`VSR_DISABLE_ASR_SHARED_CACHE=1` to keep transcript state project-local, or set
`VSR_ASR_SHARED_CACHE_DIR` to an explicitly controlled local directory. Cache
read/write failures are non-fatal and fall back to the ordinary project
checkpoint path.
Exact source-keyed visual frame PNGs and completed OCR observations use bounded
application-cache siblings as well. Frame keys bind the source digest, requested
schedule, FFmpeg version, and code version; OCR keys additionally bind the
adapter/engine identity. Shared hits still materialize ordinary evidence files
and pass the same size/hash/pixel and observation contracts.
Shared ASR and visual-frame caches default to 512 MiB each; shared OCR defaults
to 256 MiB. Context-free visual-survey receipts share the bounded visual-cache
budget, so a later project can skip the expensive full-duration detector when
the source, FFmpeg build, thresholds, and survey policy match exactly. Old flat
ASR/OCR receipts and old visual-frame schedules are pruned
oldest-first after a successful write. Override budgets with
`VSR_ASR_SHARED_CACHE_MAX_BYTES`, `VSR_VISUAL_SHARED_CACHE_MAX_BYTES`, and
`VSR_OCR_SHARED_CACHE_MAX_BYTES`; `0` disables writes while leaving project-local
checkpoint behavior intact.
The persisted progress envelope and manifest retain only the newest 32 chunk
timings plus an omitted-count marker; the final ASR metadata and individual
atomic chunk files retain the complete timing history. Chunk-start events update
the small progress envelope, while completed chunks refresh the full manifest,
avoiding duplicate large JSON writes without weakening interruption recovery.
Manifest runtime identity is captured once per process, so repeated progress
serializations do not rerun the host platform probe.
The source digest is computed once per run and reused for the run cache key,
manifest, media identity, and ASR checkpoint key, avoiding repeated full-file
hash I/O on long recordings.
On setup, source hashing overlaps configuration/OCR preparation and the
independent FFmpeg/FFprobe version probes run together; the same completed
cryptographic digest and tool strings remain authoritative cache-key inputs.
After a successful full output proof, the pipeline writes
`.state/validation-receipt.json` as a stat-bound warm-resume receipt. A cache
hit may skip the public metadata walk only when the receipt's run key,
canonical-file signature (with the stored canonical-state digest as the
manifest-only compatibility fallback), complete generated-file inventory
(including size, timestamps, inode, and symlink records), and
`metadata_verified` check all match. The receipt and volatile canonical/run-manifest telemetry are excluded
from the inventory by design; any evidence, Markdown, checkpoint, or canonical
state edit invalidates it and returns to the ordinary full validator. This is
an internal acceleration layer only: `validate <PROJECT_DIR>` always performs
the independent public proof, and a missing, malformed, or stale receipt is a
normal cache miss rather than a failure.
When only ASR chunk/overlap geometry changes, the full run key still changes and
transcript/ASR work remains isolated, but a compatible prior project can reuse
the exact source-pixel visual state. The fast path requires matching block IDs,
timings, segment IDs, and embedded frame links; any topology or visual-setting
change falls back to a complete visual rebuild.
The shared SHA-256 helper uses Python's C-level `hashlib.file_digest` when
available and falls back to the explicit chunked loop on Python 3.10 or when a
custom chunk size is requested; both paths produce the same digest.

Hard-cut and 2 fps adaptive surveys share one labeled FFmpeg decode pass when
both branches are enabled. Detector-only survey branches terminate in sinks
and use a tiny synthetic keepalive output, so a no-cut/no-change branch cannot
make the null muxer fail and force two fallback decodes. The same cold pass emits hard-cut and
periodic/contextual safety frames that are pixel-equivalent to guarded exact
extraction; adaptive samples are measurement-only and are always re-extracted
exactly. Empty-stream/
VFR edge cases fall back to the independently tested two-pass survey. Visual
requests separated by at least the bounded look-ahead window are decoded in one
FFmpeg pass with measured `showinfo` timestamps; dense/VFR request groups fall
back to the exact per-frame seek path. Late request groups use input seeking with absolute `-copyts` PTS;
the output manifest keeps a bounded
visual event history and a resource snapshot (output bytes/files, reclaimable
checkpoint bytes/files, and current/peak resident memory). Use the dry-run-first `retention report`/`retention
prune --keep N [--apply]` commands at the output-root boundary; only recognized
generated project directories may be removed.
Retention discovery also handles named run collections (for example
`<output-root>/public/<project>`): only descendants carrying the canonical
`.state/canonical-project.json` marker are counted or eligible, while models,
source media, and unrelated directories remain untouched.
Each report also separates reclaimable `.state/cache`, raw visual-frame, and
OCR checkpoint bytes/files from canonical evidence usage, making duplicate
stage artifacts visible before any destructive prune. It additionally reports
unmarked generated-footprint bytes/files and an `observed_generated_*` total;
these values include interrupted benchmark/profile trees found by the orphan
scanner, while `unclassified_*` and `observed_root_*` reconcile the entire
requested output root without making source, model, or arbitrary files prune candidates.
The report reuses its canonical-project inventory for orphan discovery and
prunes already-classified trees from the informational root walk, keeping
large retention audits bounded without changing the byte totals or prune scope.
Orphan-only reports and prune plans use the same marker-only path inventory, so
they do not size complete projects that are outside their action scope.
Project sizing itself uses one `os.walk`/`lstat` pass per marked project, preserving
symlink refusal and reclaimable-byte accounting without repeated `Path` metadata calls.
The read-only `retention orphans <OUTPUT_ROOT>` command additionally reports
incomplete trees with known reconstruction footprints but no canonical marker;
these are informational only and never become automatic prune candidates.
After inspection, `retention prune-orphans <OUTPUT_ROOT>` provides a second
dry run; only explicit `--apply` removes recognized orphan directories. It
rechecks containment, symlink status, marker absence, and generated-footprint
markers immediately before each deletion. Marked canonical projects remain
eligible only for the separate `retention prune --keep N --apply` workflow.
For a validated project that is already complete enough to preserve canonical
evidence, `cache compact <PROJECT_DIR>` reports the exact duplicate visual/OCR
checkpoint bytes; add `--apply` to remove only those caches while retaining ASR,
repair, survey, evidence, and metadata state. The local vision adapter retries
one malformed response with a schema-correction prompt before recording an
explicit claim-free fallback.
For a source folder, `fast-video-analyzer batch` is the bounded execution
wrapper. It probes and sorts every supported video, forecasts the next project
from duration-aware historical rates, requires a configurable free-space reserve,
and writes `.challenge-batch.json` after each completed source. It runs one
pipeline at a time so ASR, OCR, semantic VLM, and PNG writers cannot multiply
peak disk/RAM usage. `--dry-run` is non-mutating apart from the small manifest;
`--semantic-max-packets N` time-spreads semantic packets and records deferred
IDs as non-blocking review work. A storage-blocked forecast returns exit 4
without starting that source.
The `run` and `batch` commands may omit `--output`; a single run then writes
`<video stem>.md` beside the source in `<video stem> (Analyzer Outputs)`, while
batch keeps its resumable manifest under `(Analyzer Batch Outputs)` and places
each project beside its source. Set `VSR_OUTPUT_ROOT` for a dedicated SSD/output
volume while retaining the same resumable layout. `run
--semantic-max-packets N` exposes the same bounded Codex handoff budget as batch;
larger reference-only bundles reduce manual resume passes without copying PNGs.
For long videos on a host with an NVIDIA runtime and at least twelve logical
CPUs, the default `auto` policy (`VSR_PARALLEL_VISUAL_SURVEY` unset or set to
`auto`) overlaps the structural survey with automatically resolved local ASR in
one bounded worker. This is scheduling-only: the final visual stage adds
transcript context after ASR completes, adaptive samples remain
measurement-only, and any survey-worker failure falls back to the sequential
path. Set `VSR_PARALLEL_VISUAL_SURVEY=0` to disable it or `=1` to force overlap
after a host-specific benchmark; explicitly injected ASR adapters never opt
into `auto` because their device/resource contract is caller-owned.
Adjacent-frame analysis runs in bounded contiguous chunks with a one-frame
overlap. Each worker reads a frame once, the overlap preserves the exact
before/after comparison at chunk boundaries, and ordered reassembly keeps IDs,
quality scores, difference regions, and dHashes deterministic.
When an OCR adapter is enabled, its checkpointed reads overlap this independent
analysis phase; no-adapter runs avoid the extra orchestration pool.
`cache purge <PROJECT_DIR>` additionally removes the project-local stage cache,
both visual survey receipts, and raw visual-frame checkpoints, while preserving the
canonical project, evidence images, source media, and ASR checkpoints.
It does not delete cross-project ASR, visual-frame, OCR, or visual-survey caches; those are
recoverable acceleration state and can be isolated with the corresponding
`*_SHARED_CACHE_DIR` variables (or disabled with the matching
`VSR_DISABLE_*_SHARED_CACHE=1` setting) when locality is required.

After a valid run, the pipeline compacts those completed visual/OCR caches by
default because canonical evidence is already committed and an unchanged rerun
returns through the run-cache key. Interrupted or blocked runs keep their
checkpoints for resume. Set `VSR_KEEP_COMPLETED_CHECKPOINTS=1` while iterating
on downstream visual settings to retain them; the manifest records
`performance.checkpoint_compaction` and refreshed output byte/file usage.
The compact measured survey receipts remain intentionally: they contain no image
pixels, but allow a downstream visual rebuild to reuse source-keyed detector
timings instead of repeating the hard-cut/2-fps scan.

The measured visual survey is cached in `.state/checkpoints/visual-survey.json`.
The cache key binds the immutable source digest, FFmpeg version, code version,
survey flags/thresholds, interval, and the bounded speech-reference points used
for contextual candidates. A matching cache restores measured candidate PTS and
time-base records; malformed, incomplete, or stale entries are treated as a
cache miss and replaced atomically. This saves the hard-cut/2 fps decode on
repeat or downstream visual rebuilds without weakening cold-run coverage. On a
cold run, shared hard-cut and periodic PNGs are materialized directly into the
raw-frame checkpoint; only adaptive/context requests not covered by that
exact-safe set invoke the guarded extraction path.

The companion `.state/checkpoints/visual-survey-structural.json` receipt omits
transcript reference points and stores only structural candidates. A changed
subtitle/transcript context can therefore reuse that detector result and merge
new contextual candidates; request-specific frames remain exact-seek guarded
when the raw-frame schedule changed.

Raw pre-metadata frames use a separate bounded checkpoint at
`.state/checkpoints/visual-frames/<key>/`. Each PNG is restored only after its
manifest request, measured PTS/time base, byte size, and SHA-256 agree; any
corruption or partial checkpoint triggers normal FFmpeg extraction. If a changed
context creates a new schedule key, the implementation scans at most 64 prior
manifests and copies only non-conflicting frames whose source digest, FFmpeg
version, measured timing, size, and SHA-256 agree; missing timestamps still use
the guarded exact path. The complete visual-frame checkpoint set is capped at
512 MiB by default (`VSR_VISUAL_FRAME_CACHE_MAX_BYTES` overrides it; `0`
disables writes); the oldest prior schedule receipts are pruned after a
successful write, so resumability cannot create unbounded artifact growth.
Copies between the project-local and cross-project raw checkpoint trees use a
same-volume hardlink when the filesystem supports it, then fall back to the
existing atomic byte copy for cross-volume or unsupported paths. Evidence
restores remain independent byte copies; hardlinks are safe between acceleration
trees because raw checkpoint bytes are immutable and metadata enrichment replaces
evidence paths atomically. Set
`VSR_DISABLE_CHECKPOINT_HARDLINKS=1` for a copy-only diagnostic run.
When a changed transcript/context creates a new visual schedule, the bounded
prior-schedule lookup searches both the project-local and shared raw-frame trees;
only source/FFmpeg/timing/hash-agreeing requests are reused, and conflicts are
discarded rather than guessed.
The schedule key excludes worker counts because concurrency changes bounded
scheduling, not measured PTS, validated PNG pixels, or downstream evidence. A host can
tune `VSR_FRAME_EXTRACT_WORKERS` and resume without duplicating an otherwise
identical raw-frame receipt.

Creation pixel normalization and metadata commits reuse one bounded visual
worker pool while remaining ordered phases. This removes repeated thread-pool
startup without combining the read and write phases or weakening atomic failure
handling.
Checkpoint copies and SHA-256 accumulation use the same bounded pool, while the
checkpoint manifest is written in deterministic request order. Valid checkpoint
restores use that bounded pool for independent verified copies, preserving
frame order while avoiding a serial copy/fsync pass. Exact-safe hard-cut and
periodic survey frames use the pool when materialized into evidence; adaptive
requests still follow the guarded measured extraction path.

Completed local OCR observations use a separate bounded checkpoint at
`.state/checkpoints/ocr/<key>.json`. Each entry binds the source digest,
adapter/engine settings, code version, and normalized source-pixel hash. A
retry restores exact observation content (including token boxes and uncertainty)
and remaps only the current frame/observation IDs. Independent OCR subprocesses
use their own bounded pool (`VSR_OCR_WORKERS`, default up to twelve on hosts with at least
16 logical CPUs and a conservative half-CPU pool on smaller hosts; an explicit override can
raise it to sixteen after host benchmarking), separate from FFmpeg decoder workers;
this changes scheduling only, not OCR inputs or uncertainty records.
For per-frame OCR adapters, completed observations are flushed atomically to
the project checkpoint every `VSR_OCR_CHECKPOINT_BATCH` results (default 16),
so an interrupted worker run resumes already completed pixels instead of
discarding the entire stage.
The flush is local acceleration state and never changes canonical IDs, token
boxes, or uncertainty. A retry remaps only the current frame/observation IDs;
changed pixels are rerun and OCR failures are not cached. The default budget is 64 MiB
(`VSR_OCR_CACHE_MAX_BYTES` overrides it; `0` disables writes). `cache purge`
removes OCR checkpoints along with stage-cache, survey, and raw-frame state while
leaving canonical evidence and ASR checkpoints intact.
Tesseract version probes are single-flight and stat-bound across adapters, while
per-frame recognition remains independent so uncertainty and token boxes are preserved. The
run manifest also records the bounded scheduler snapshot and per-backend ASR runtime settings,
so CPU/GPU comparisons remain interpretable without changing cache identity or evidence.

The repository also ships `scripts/benchmark_pipeline.py` for a repeatable
offline timing check. It calls the public pipeline, validates the produced
project, reports the manifest's stage/resource telemetry and post-write disk
parity without fetching weights or contacting a service. `--repeat N` adds conservative min/median/p95
timings; warm repetitions share the project to measure resume, while
`--no-resume` places each cold iteration in a fresh isolated `cold-NNN/`
directory (including a single iteration) so stage checkpoints cannot be reused
accidentally. Every report also includes `performance_summary` with measured
wall time, stage-time sum, peak RSS, output bytes/files, and any parallel-survey
completion time; compare wall time when stages overlap.
Warm-cache hits are labeled `measurement_mode=warm-cache-hit` and
`stage_telemetry_current=false`, because their stage records are inherited from
the previous manifest rather than measured during the tiny resume call.
The harness reuses `run_pipeline`'s final validation proof by default, so a
benchmark does not immediately decode and verify every image a second time.
Use `--independent-validation` when an external public-validator pass is needed;
the report records `validation_source` as `pipeline-final` or
`independent-public` so timing comparisons remain explicit.
Reports keep pipeline critical-path time in `elapsed_seconds` and expose
`validation_elapsed_seconds`/`total_elapsed_seconds` separately, preventing an
optional audit pass from being mistaken for model or decode work.
The harness accepts `--asr-chunk-seconds` and `--asr-overlap-seconds` so local
large-v3 window economics can be measured on the target host without changing
the strict preset by accident. For an isolated cold comparison, use
`--asr-chunk-sweep 150,300,600,900`; each size gets a fresh output tree and the
recommendation is emitted only when validation, substantive-segment coverage,
and word coverage agree. When no transcript/subtitle is supplied, a verified
local faster-whisper adapter is reused across candidates so model initialization
does not dominate the comparison; output/checkpoint trees remain isolated. A
recommendation is host-specific evidence, not an accuracy proof, and a coverage
disagreement intentionally yields no winner.
For media that has no usable transcript, `scripts/profile_visual_stage.py`
isolates the same deterministic visual stage with a neutral scaffold and no
ASR/OCR/semantic-model dependency.
The profiler computes the immutable source digest once and passes it through
survey and raw-frame cache boundaries, keeping long-media timing free of a
redundant second full-file hash.

The default semantic route creates a bounded, file-based Codex/subagent review
bundle only when unobserved packets remain. It references existing evidence
PNGs (never copies pixels), includes script/OCR context and a strict annotation
schema, and keeps each image's metadata transaction and revision history
durable. The host applies responses only after canonical, packet, and frame
hashes are rechecked; missing or uncertain responses remain review-only. The
legacy local semantic command is explicit compatibility behavior and is not
started by an ordinary run.
Semantic event/block/review links are applied to the already-loaded project
before that observation's canonical commit, so a batch performs one canonical
project serialization per accepted observation plus one final audit commit;
there is no redundant second full-project write for the same image.
Each continuation also records bounded elapsed seconds and
observations-per-second telemetry in the provider-usage manifest entry; these
measurements describe the local run and never become an accuracy claim.
Those per-observation commits patch only the known changed canonical roots after
the transaction marker and PNG read-back proof. A process
restart or unexpected state falls back to a complete atomic write, preserving
the one-commit crash boundary without reserializing unchanged transcript/timeline
evidence on the normal path.
Within one sequential batch, validated root and array-item offsets are reused
after each atomic commit; the stat-bound receipt invalidates that optimization
on restart or external edits without changing the safe fallback behavior.
Semantic continuation preflight indexes canonical frame IDs once instead of
scanning the full frame list for every packet. Visual-stage sufficiency links
also use stable frame-to-review indexes, so long-form review bookkeeping stays
linear in retained frames and actual links while preserving source order.
Large new canonical state writes use deterministic compact UTF-8 JSON to reduce
disk amplification; small projects retain readable formatting, and readers plus
the single Markdown artifact retain the same schema.
The built-in local llama.cpp provider opts into a bounded shared semantic cache.
Entries are reusable only when the provider/model revision, prompt template,
complete packet, and every referenced evidence image's decoded RGBA pixel hash
match.  Using the canonical pixel digest (rather than the mutable PNG container
digest) lets a metadata rewrite reuse the same visual inference safely; host-agent,
external, and non-opted-in providers remain project-local. Set
`VSR_DISABLE_SEMANTIC_SHARED_CACHE=1` (or `VSR_DISABLE_VISUAL_SHARED_CACHE=1`)
to disable it, or use `VSR_SEMANTIC_SHARED_CACHE_DIR` and
`VSR_SEMANTIC_SHARED_CACHE_MAX_BYTES` to control the local cache. In the historical,
explicitly enabled slide-lecture local Qwen3-VL run, two observations took 41.239 s cold and
0.616 s from exact semantic cache hits (66.9x); cache telemetry is recorded in
the provider usage manifest and does not alter canonical evidence. If a local
provider returns a structurally invalid annotation, the packet is committed as
an explicit review-only fallback with no image claim. Packet-local context 400s,
truncated JSON, and missing frame citations remain isolated to that packet so
healthy later packets still receive a model attempt; the circuit breaker opens
only after repeated shared provider-health failures (connection/timeouts or
5xx responses). Set `VSR_SEMANTIC_FAILURE_LIMIT` to tune that health bound.
Within one continuation pass, packets with the same frame IDs/paths and the
same evidence questions reuse the first schema-validated visual annotation under
the new candidate ID. Packets with different IDs may also reuse only when their
ordered frame roles, questions, and canonical decoded-pixel (or file) digests
are identical; every reused citation is remapped to the target IDs and passed
through the normal packet validator. Differing pixels or question scopes never
reuse it. Both reuse counts are recorded in provider telemetry and canonical
commits remain one-per-packet. Packet JSON is validated through a bounded,
stat-keyed process-local cache (invalidated on size/mtime/inode changes), which
avoids reparsing the same packet during discovery, selection, and continuation
preflight without persisting untrusted cache state.
The same exact-content key is also persisted as a bounded `content-*` semantic
cache entry when the local provider succeeds. A later process or project can
reuse that annotation only when the model revision, prompt hash, ordered frame
roles, questions, and decoded pixel digests match; the stored citations are
remapped to the target packet and revalidated before commit. Invalid or stale
content entries are ignored as cache misses, and the normal shared-cache size
limit prunes them with the other semantic entries.
Pruning uses a stat-bound byte ledger and scans at a bounded cadence (or
immediately when the cap is crossed), avoiding an O(cache-size) directory walk
for every observation while keeping the configured storage bound authoritative.
For multi-hour media, `VSR_SEMANTIC_MAX_PACKETS` can impose a deterministic
semantic budget: roughly half the expensive slots are evenly spaced temporal
anchors, while the remaining slots prioritize measured scene-change/OCR/
consequential packets with stable ties. Deferred packet
IDs become a non-blocking review item; deterministic frames, OCR, and transcript
evidence are never discarded, and a later run without the budget resumes the
deferred semantic work.
The scheduler updates one unresolved automatic budget-frontier item in place as
that set shrinks, and removes it when the frontier is empty. Older user-decided
review items and all observation/correction history remain untouched, preventing
bounded continuation metadata from growing quadratically.
To continue only that deferred work, create a bounded host bundle:
`fast-video-analyzer review bundle create <PROJECT_DIR> --max-packets N`.
After a Codex/subagent writes matching response JSON files, apply it with
`fast-video-analyzer review bundle apply <PROJECT_DIR> --bundle <BUNDLE>`.
The host reuses persisted transcript, frames, OCR, packets, and metadata, so no
ingest, ASR, scene detection, frame extraction, or OCR reruns occur. A stale,
missing, or uncertain response remains review-required rather than becoming a
claim.
Telemetry is committed by patching only the canonical `manifest` field after
the evidence commit; the large evidence arrays are not re-encoded a second
time, keeping repeated bounded passes I/O-efficient without weakening the
atomic-write or validation guarantees.
If a prior local run recorded an HTTP 400 transport fallback, add
`--retry-fallbacks` to retry only those explicitly identified packets after the
adapter's smaller transport retry. The local adapter first resizes images,
then drops only same-timestamp supplemental crops (and, as a final bounded
retry, keeps one focus/action/result frame); omitted transport IDs are recorded
in uncertainty while canonical evidence remains unchanged. Malformed or
schema-invalid fallbacks remain review-only unless a separate, deliberate
provider change is made. After the model's one explicit citation retry, the
adapter may repair only the narrow missing-focus evidence-list citation; it
preserves every model-authored fact and records the structural repair in
uncertainty, while all claims still pass the normal packet validator.
Batch preflight counts these explicit HTTP-400 retry markers separately from
event-level observation status, so `semantic-batch --retry-fallbacks` does not
silently skip a project whose only remaining work is a persisted fallback.
When the local prompt/schema has deliberately changed, add
`--retry-semantic-pending` to revisit `semantic_pending` observations whose
stored prompt hash differs from the current adapter. This is
opt-in and deterministic: unchanged-prompt pending markers are not repeatedly
recomputed; legacy positive-confidence pending markers are prioritized before
already-conservative zero-confidence markers, and visible claims are still accepted only after the normal packet,
metadata, and validation gates.
For a directory containing multiple canonical projects, create and apply one
bounded review bundle per project. The legacy `semantic-batch` command remains
available only for explicit local-model compatibility runs; the normal host
route never starts a local VLM server, skips projects with no pending packets,
and preserves the free-space reserve.
Requests are serial by default. A two-worker mode exists for explicit
experiments (`VSR_ALLOW_UNSAFE_SEMANTIC_PARALLEL=1` plus `--workers 2`), but it
can exhaust llama.cpp's shared KV cache on dense 4--5-frame packets and turn
otherwise valid work into HTTP-400/500 review fallbacks. Keep one worker for
production quality; the same option is available on the single-project
`semantic` command, and `VSR_SEMANTIC_WORKERS=1..2` is the API-level override.
On the RTX 3060 12-GiB reference host, a shadow run of four real pending
packets measured 22.2 s with one slot versus 17.1 s with two (23% faster and
about 8.7 GiB peak GPU memory). Core event types, confidences, frame citations,
text candidates, and consequential changes matched; one non-core description
wording differed, so two-slot mode remains an explicit throughput benchmark
rather than the accuracy-first default.
Routine batch output is deliberately compact: it reports counts and bounded ID
samples instead of dumping every applied observation or deferred packet. Add
`--full-output` when a machine-readable per-packet audit is specifically needed;
the canonical project and manifest always retain the complete records either way.
Projects created with an older Tesseract TSV parser can be repaired in place with
`fast-video-analyzer evidence ocr refresh <PROJECT_DIR> --workers N`.
This reads existing evidence PNGs, updates only canonical OCR and packet context,
rerenders the single Markdown output, and validates metadata; it does not decode
the source video, rerun ASR, or rewrite image pixels. A changed OCR result is
reported as a recommendation for a later targeted semantic pass.
For explicit legacy local-Qwen compatibility runs, the Qwen3-VL server uses a 32,768-token context by default so dense
4--5-frame packets do not fail before generation; each response remains a
bounded 768-token structured object with optional thinking disabled. A real
Filipino production-packet benchmark produced the same validated result at
768 tokens while reducing warm response latency by more than half.
`VSR_LOCAL_VISION_MAX_TOKENS` (256--4096) and
`VSR_LOCAL_VISION_MAX_IMAGE_EDGE` (768--2560) are explicit transport benchmark
overrides; the default full profile is 1,280 pixels and 768 tokens. Packets
without meaningful OCR or frame-linked text context automatically use a conservative
896-pixel/448-token profile (or 768-pixel/384-token for a single frame) because
they are normally text-free person/scene evidence; empty OCR markers are not
treated as text-bearing, while actual OCR/text-bearing packets retain the full
profile so exact visible text is not lost. Filipino, dense-OCR,
and email-screen packets preserved event type/confidence at 1,280 while cutting
warm transport latency roughly 30--40% versus 1,600.
A compact text-free request that returns confidence-zero `semantic_pending` is
escalated once to the full 1,280-pixel/768-token profile; if that retry fails,
the conservative pending result is retained rather than inventing a claim.
Transcript-linked OCR can contain thousands of repeated TSV geometry rows. The
local transport projects those rows to de-duplicated readable candidates with
an 8,192-character per-block bound and an explicit truncation marker; complete
OCR text, geometry, and uncertainty remain in the canonical packet and the
supplied pixels remain authoritative. This projection is versioned in the
transport/cache profile so old annotations cannot cross the prompt contract.
The prompt and its one deterministic correction retry enumerate the exact
focus/action/result IDs and require at least one in `evidence_frame_ids`, even
for `no_change` or `semantic_pending`; a response that still fails packet
validation remains a claim-free review fallback.
Prepared image data URLs use a 64 MiB stat/inode-bound in-process LRU cache by
default, so repeated frame references avoid PNG decode/resize/base64 work while
the canonical evidence pixels remain untouched. Set
`VSR_LOCAL_VISION_TRANSPORT_CACHE_MAX_BYTES=0` to disable it; the cache is
memory-only and is never another SSD artifact.
Semantic shared-cache keys bind the exact model/prompt, decoded frame hashes,
selected transport profile, and prompt-side OCR/transcript/context projection.
Packet caches are preferred when the complete packet matches; content-remap
caches are used only after context validation and citation remapping, so a
faster cache hit cannot silently cross OCR or neighboring-event evidence.
Images are resized only in the transport request; canonical evidence pixels are
unchanged. During finalization, only validator-identified
generated candidate files without canonical mirrors are pruned, preventing stale
rebuild artifacts from blocking an otherwise valid project.

Frame-quality and neighbor-difference calculations run in a bounded worker pool
once ordered neighbors are known; candidate scoring and canonical commits remain
serial and deterministic.
Candidate payload/revision bookkeeping copies only list containers; unchanged
nested evidence envelopes are reused read-only until a candidate is replaced.
Rejected-frame enrichment reuses the creation-stage validated metadata mirror
and only reopens a PNG when that mirror is absent, preserving the guarded
fallback without repeating normal-path metadata parsing.
The PNG analysis pool is independently bounded up to eight workers by default
(`VSR_FRAME_ANALYSIS_WORKERS` accepts an explicit 1--8 override), while the
FFmpeg extraction/metadata pools retain their conservative four-worker cap.
Each worker decodes a current/previous PNG pair once and reuses those in-memory
pixels for quality, difference, and the current frame's dHash. The deterministic
metadata commit reuses that dHash instead of opening every retained PNG again;
the standalone helpers retain their independent path-based safety behavior and
historical return contracts.
The bounded comparison image is shared by quality and neighbor-difference
calculations within the worker, so the same frame is not resized or color-
converted twice. This is an internal reuse only; public path-based helpers keep
their independent decoding and validation behavior.
Quality, clipping, edge-density, change-ratio, and mean-difference statistics
use Pillow's native 8-bit histograms. Their integer sums and thresholds are
unchanged, avoiding Python-level pixel scans without adding a NumPy dependency
or changing frame scores and evidence decisions.
Selection receives the same precomputed dHash through each candidate, so local
deduplication does not reopen neighboring PNGs; candidates created by external
callers without that field still use the guarded path-based fallback.

Batch frame extraction stops at the last request's 250 ms look-ahead window.
Two-request groups and genuinely sparse three-request targets use bounded exact seeks, while larger survey groups keep the
one-pass guarded selector; both paths preserve measured PTS and lossless PNG
pixels. Sparse groups larger than two requests use the exact-seek route when
their density is at or below the measured 0.04 requests/second threshold.
For a long span (at least ten minutes), the bounded cost model also permits
exact seeks up to 0.10 requests/second; this avoids traversing many minutes of
high-resolution video for a sparse survey even when its density is just above
the short-span crossover. Dense groups retain the one-pass selector. The extraction pool also bounds
FFmpeg's per-process decoder threads from the requested worker count (maximum
four threads per process by default). This prevents concurrent sparse seeks from
oversubscribing the host while retaining the same measured PTS, frame count, and
lossless pixels; callers can override the bound with
`extract_frames(..., ffmpeg_threads=...)` when hardware-specific benchmarking
justifies it.
The separate full-duration hard-cut/adaptive survey also receives a bounded
codec-thread count (up to four by default; VSR_SURVEY_FFMPEG_THREADS is an
explicit 1--8 override). The standalone hard-cut/adaptive fallback uses the
same bound, so a disabled branch does not silently spawn an unbounded decoder.
This changes decoder scheduling only: the measured survey PTS, branch
attribution, and emitted PNG bytes remain authoritative.

Independent output validation uses a separate bounded metadata pool (up to
eight workers by default; `VSR_VALIDATOR_METADATA_WORKERS` accepts an explicit
1--16 override). It parallelizes only independent per-image checks and merges
errors in canonical frame order; the actual pool is additionally capped at the
number of known frames so tiny projects do not pay for idle threads. Schema,
embedded-payload, file-hash, and decoded-pixel invariants remain unchanged.
When validation has a strict project-root snapshot, containment uses lexical
paths plus lstat-based symlink/reparse-point rejection; callers without that
snapshot retain the legacy strict-resolve path.
Repeated component checks share the snapshot within one validation and all
recorded component signatures are rechecked before returning, so a mutation
during validation remains a failure rather than a cache hit.

Metadata-only revisions reuse a previously verified pixel hash and compare the
encoded PNG IDAT stream instead of decoding the same large image repeatedly;
public verification keeps the independent full-decode default. Localized crops
use the already decoded crop buffer for deterministic quality, dHash, and pixel
identity, reuse the validated in-memory parent metadata mirror, and preserve
both creation and deterministic revision records without performing a transient
duplicate PNG metadata write. Crop decode/write/quality preparation runs in a
bounded visual worker pool, while revision numbering, packet mutation, and
canonical commits remain ordered and sequential; the progress stream exposes a
`crop_preparation_completed` boundary for profiling. Crop packet additions are
held in the already-written packet mirrors and flushed once per changed event
after the crop loop, preserving partial full-frame packet durability while
avoiding repeated crop-time packet reads and atomic writes.
Creation envelopes likewise decode/hash each extracted source once, then use the
internal IDAT-preserving metadata path; the writer accumulates the exact
post-write file SHA-256 and IDAT digest while copying bytes, avoiding a second
full read just to populate canonical ``file_hash`` fields or prove the encoded
pixel stream was preserved. Public metadata writes retain their independent
full-decode defaults, and ``embed_metadata_with_file_hash`` exposes the digest
only to deterministic internal callers.
Sufficiency revisions use the fast path only after a current whole-file hash
matches the canonical frame hash; absent or stale hashes deliberately retain
the independent decoded-pixel verification route.
The visual enrichment loop validates the canonical metadata mirror already
returned by the creation/candidate write, avoiding an immediate embedded PNG
reparse; records from legacy callers without that mirror retain the guarded
embedded-read fallback.
Generated-run validation verifies the canonical whole-file SHA-256 before using
the embedded pixel hash for metadata consistency, then checks PNG dimensions and
Description from headers without inflating IDAT. This avoids a redundant PNG
decode while the public `validate` command keeps the independent decoded-pixel
mode. Repeated internal validation passes reuse unchanged verified envelopes
from a bounded cache keyed by the image stat signature, canonical payload
digest, and expected pixel hash; edits to either bytes or canonical state force
a fresh verification. Markdown and forbidden HTML artifacts are gathered in one
shared tree walk, avoiding duplicate project traversal on each validation pass.
Hash-backed internal validation also reuses the parsed canonical JSON object in
a two-project stat-bound cache; public validation reparses canonical state, and
validated evidence-image models are reused for the same unchanged entry after
the full schema model validates once. Public validation reparses and revalidates canonical state; any canonical-file
signature change invalidates both internal entries.
The deterministic audit report is cached under the same stat-bound internal
policy; public validation always recomputes it.
For a normal run, the output-contract audit is written before the final
post-compaction validation, so the returned validation proves the canonical,
audit, Markdown, and compacted checkpoint state together. A blocked rewrite is
validated immediately because it intentionally skips compaction; this keeps the
failure evidence durable without adding a redundant validation pass to healthy
runs.
The initial healthy-run preflight checks structure, links, schema, timeline, and
audit/output contracts with metadata verification disabled; the returned final
validation performs the full independent metadata proof after compaction. If
that final proof fails, the project is marked blocked and the failure is
preserved.
Healthy runs defer the resource-usage snapshot until checkpoint compaction, so
the large canonical project is not serialized once merely to add intermediate
telemetry. The final canonical and run-manifest writes still include the complete
resource and compaction records; blocked runs publish the same telemetry before
their final full validation, and failed validations retain telemetry as well.
When only the root `audit` or `manifest` field changes, the canonical writer
atomically patches that field while preserving unchanged evidence bytes. The
patcher validates the resulting JSON and falls back to a complete redacted write
for malformed, legacy, or structurally unexpected state.

## Status handling

Use `review_required` for usable output with consequential uncertainty, and `blocked` when a required stage cannot safely continue. Never mark a failed/skipped stage complete. Automatic checks never create `fully_verified`.
