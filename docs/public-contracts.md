# Public contracts

Status: current for the `v0.1.x` line. Last reviewed: 2026-08-20.

This document records the surfaces that users and integrations may rely on.
It is intentionally smaller than the implementation and is maintained when a
public command, output, schema, or compatibility promise changes.

## Status vocabulary

- **Stable:** documented, contract-tested, and changed only with release notes
  and compatibility evidence.
- **Provisional:** usable and tested, but allowed to change in a minor release
  while the product contract is still being shaped.
- **Deprecated:** retained for existing users; new integrations should use the
  named replacement. Removal requires a documented migration and release-note
  entry.
- **Internal:** implementation detail with no compatibility promise. Do not
  import it or parse it from an automation workflow.

## Console entrypoints

`fast-video-analyzer` is canonical. These entrypoints invoke the same parser
and behavior:

| Entry point | Status |
| --- | --- |
| `fast-video-analyzer` | Stable canonical command |
| `long-video-analyzer` | Stable compatibility alias |
| `video-script-reconstructor` | Stable compatibility alias |

The aliases remain until a release note names a replacement and gives users a
migration window. Do not add another alias without a concrete compatibility
case and a contract test.

## CLI surface

Stable lifecycle and evidence commands:

| Surface | Contract |
| --- | --- |
| `doctor` | Non-mutating capability and prerequisite report; `--offline` is the default safety posture. |
| `diagnostic-bundle` | Writes a sanitized support ZIP; `diagnostics` and `support-bundle` are compatibility aliases. |
| `plan` | No-download, no-full-processing plan for one input. |
| `run` | Reconstructs one input into one project and emits machine-readable result JSON. |
| `batch` | Sequential, resumable processing of a source folder with storage guards. |
| `validate` | Validates one project’s links, hashes, metadata, chronology, and state. |
| `evidence metadata show/verify` | Reads or verifies image evidence metadata. `show --json` is the machine-readable form. |
| `evidence packet show` | Reads one stored visual packet. |
| `evidence observation ingest` | Ingests a validated external observation into a project. |
| `evidence ocr refresh` | Reuses existing evidence to refresh OCR packets and validation. |
| `review list/show/apply` | Lists, inspects, and applies attributable human review decisions. |
| `review bundle create/create-all/apply` | Creates or applies hash-bound offline review bundles. |
| `finalize` | Applies an attributable human final sign-off; it never happens automatically. |

Owner-facing or provisional capability and storage commands:

| Surface | Reason for provisional status |
| --- | --- |
| `models report/list/fetch/verify/remove` | Optional model-store lifecycle and worker readiness are still evolving. Downloads and removal remain explicit. |
| `workers list/install/verify` | Isolated heavyweight runtime management is host-dependent. |
| `retention report/orphans/prune/prune-orphans` | Cleanup semantics are safety-sensitive and require owner review; `--apply` is destructive. |
| `cache purge/compact` | Cache layout is an optimization detail and may change without project-schema migration. |

Legacy paths:

| Surface | Replacement/status |
| --- | --- |
| `semantic`, `semantic-batch` | Deprecated compatibility paths; use `review bundle` for host-agent/subagent review. |
| `--vision-mode local` | Deprecated explicit legacy local-Qwen path; use the default host-agent bundle or a documented future provider. |
| `review bundle batch-create`, `review bundle create-batch` | Deprecated aliases for `review bundle create-all`. |

The parser-drift test enumerates every current top-level command and the
compatibility aliases. A new command must update this document, its help or
contract test, and the appropriate user documentation in the same change.

## Exit codes and result status

| Code | Meaning | Typical response |
| ---: | --- | --- |
| `0` | Requested operation completed successfully. | Consume the JSON/result or continue. |
| `1` | Unexpected internal failure. | Preserve sanitized diagnostics and report a defect. |
| `2` | Usage, input, configuration, missing-file, or argument error. | Correct the command or input. |
| `3` | Evidence exists but review or deferred work remains. | Inspect the project/review queue; do not treat it as a clean final answer. |
| `4` | Blocked, invalid, unsafe, or failed validation/contract. | Stop, preserve state, and follow the relevant runbook. |

`review bundle apply --accept-partial` may return `0` after a valid partial
commit with no missing or invalid responses; the project can still remain
`review_required`. A zero exit code does not mean `fully_verified`.

For `run`, stdout JSON contains `project_dir`, `markdown`, `status`,
`exit_code`, `validation`, and `validation_errors`. Progress, when enabled,
goes to stderr so stdout remains machine-readable.

The project status values are:

- `automatically_checked`: deterministic validation passed; human review may
  still be appropriate.
- `review_required`: unresolved evidence, uncertainty, or deferred work exists.
- `human_reviewed`: an attributable review decision was applied.
- `fully_verified`: an attributable final sign-off was applied; automation can
  never create this status.
- `blocked`: the project cannot safely claim completion.

## Python imports

Stable Python surface today:

- `video_script_reconstructor.__version__`
- `long_video_analyzer.__version__`

There is no stable high-level Python execution facade yet. The following paths
are **provisional** because they are documented for local use but are module
internals rather than a versioned API:

- `video_script_reconstructor.pipeline.run_pipeline`
- `video_script_reconstructor.validate_output.validate_project`
- `video_script_reconstructor.cli.main`

All other package modules, classes, helper functions, cache files, and provider
implementations are internal. The CLI is the compatibility surface until a
small typed API is deliberately designed and tested. Documentation must label
provisional imports instead of presenting them as stable library APIs.

## Schemas and generated output

- `configs/schema.json` is the canonical JSON Schema for persisted project
  records; the current persisted schema version is `1.0`.
- `configs/strict.yaml` and `configs/balanced.yaml` are supported preset
  configurations. Unknown configuration fields are rejected.
- `tests/acceptance_manifest.json` is the CI acceptance-suite contract, not a
  user project format.
- `tests/corpus_manifest.json` is the public evaluation-corpus and provenance
  contract; it contains source hashes and coverage gaps, not run output.
- `.state/canonical-project.json` is the authoritative project record.
  Markdown is rendered from it and must not be treated as the canonical store.
- `.state/run-manifest.json`, `.state/audit.json`, review queues, checkpoints,
  caches, and packet files are generated state. Read them through documented
  commands; do not hand-edit them or build integrations around incidental
  filenames.

One run produces one Markdown report plus linked evidence and state. With no
`--output`, a video uses `<video stem> (Analyzer Outputs)` beside the source.
With `--output <root>`, the project is `<root>/<video stem>`. The supported
project layout is:

```text
<project>/
├── <video stem>.md
├── evidence/
│   ├── full/
│   └── crops/
└── .state/
```

Generated project folders, source media, model weights, and raw benchmark
results do not belong in the repository.

## Environment variables

Environment variables are secondary configuration, not a replacement for an
explicit CLI option. Empty or invalid values are ignored or rejected according
to the owning subsystem; callers should capture the effective environment when
it matters for reproducibility because not every tuning value is persisted.

Supported operational variables:

| Area | Names |
| --- | --- |
| Output and stores | `VSR_OUTPUT_ROOT`, `VSR_MODEL_ROOT`, `VSR_WORKER_ROOT`, `VSR_TESSERACT_PATH`, `VSR_FASTER_WHISPER_LARGE_V3_PATH` |
| Source selection | `VSR_PREFER_WHISPER`, `VSR_ASR_LANGUAGE_HINT` |
| Review/limits | `VSR_SEMANTIC_MAX_PACKETS`, `VSR_HOST_REVIEW_MAX_PACKETS`, `VSR_SEMANTIC_FAILURE_LIMIT`, `VSR_KEEP_COMPLETED_CHECKPOINTS` |
| Worker counts | `VSR_ASR_CPU_THREADS`, `VSR_FASTER_WHISPER_NUM_WORKERS`, `VSR_OCR_WORKERS`, `VSR_OCR_REFRESH_WORKERS`, `VSR_FRAME_EXTRACT_WORKERS`, `VSR_FRAME_ANALYSIS_WORKERS`, `VSR_CROP_PREP_WORKERS`, `VSR_SURVEY_FFMPEG_THREADS`, `VSR_VALIDATOR_METADATA_WORKERS`, `VSR_SEMANTIC_WORKERS` |
| Cache budgets/locations | `VSR_DISABLE_ASR_SHARED_CACHE`, `VSR_ASR_SHARED_CACHE_DIR`, `VSR_ASR_SHARED_CACHE_MAX_BYTES`, `VSR_DISABLE_VISUAL_SHARED_CACHE`, `VSR_VISUAL_SHARED_CACHE_DIR`, `VSR_VISUAL_SHARED_CACHE_MAX_BYTES`, `VSR_DISABLE_SEMANTIC_SHARED_CACHE`, `VSR_SEMANTIC_SHARED_CACHE_DIR`, `VSR_SEMANTIC_SHARED_CACHE_MAX_BYTES`, `VSR_OCR_CACHE_MAX_BYTES`, `VSR_OCR_SHARED_CACHE_MAX_BYTES`, `VSR_VISUAL_FRAME_CACHE_MAX_BYTES` |

Advanced or compatibility-only variables are recognized but provisional:

```text
VSR_ALLOW_LEGACY_LOCAL_MODELS
VSR_ALLOW_UNSAFE_SEMANTIC_PARALLEL
VSR_DISABLE_CHECKPOINT_HARDLINKS
VSR_FASTER_WHISPER_BATCH_SIZE
VSR_FASTER_WHISPER_BATCHED
VSR_FASTER_WHISPER_COMPUTE_TYPE
VSR_FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT
VSR_FASTER_WHISPER_DEVICE
VSR_FASTER_WHISPER_INFERENCE_MODE
VSR_FASTER_WHISPER_VAD_FILTER
VSR_LOCAL_VISION_COMMAND
VSR_LOCAL_VISION_ENDPOINT
VSR_LOCAL_VISION_MAX_IMAGE_EDGE
VSR_LOCAL_VISION_MAX_TOKENS
VSR_LOCAL_VISION_MODEL
VSR_LOCAL_VISION_TRANSPORT_CACHE_MAX_BYTES
VSR_MOSS_SPEECH_PYTHON
VSR_OCR_BATCH_SIZE
VSR_OCR_CHECKPOINT_BATCH
VSR_PADDLE_OCR_PERSISTENT_WORKER
VSR_PADDLE_OCR_PYTHON
VSR_PARALLEL_VISUAL_SURVEY
VSR_QWEN_SPEECH_PYTHON
```

Variables used only by tests or subprocess fixtures are not a user contract:

```text
VSR_FASTER_WHISPER_INTERVAL_START_MS
VSR_FASTER_WHISPER_INTERVAL_END_MS
VSR_FASTER_WHISPER_SMOKE_AUDIO
VSR_GUARDED_MOTION_DEDUP
VSR_RESULT
```

Other libraries’ environment variables, arbitrary `VSR_*` names, and secrets
are not supported configuration surfaces.

## Change and deprecation policy

For a stable surface, a change requires a focused issue or PR, a regression or
contract test, updated documentation, and a changelog/migration note when
serialized output or behavior changes. A schema change must include a reader
for supported earlier versions or an explicit migration boundary.

Provisional surfaces may change in a minor release, but the replacement and
reason must be recorded. Deprecated aliases remain available for the stated
migration window; removal is never silent. Internal names may change without
notice and must not be used by new integrations.

Contract evidence is maintained by the installed-wheel packaging suite, the
CLI compatibility tests, the acceptance manifest, the repository verifier, and
the platform smoke workflows. A green test proves only the surface it covers;
it does not promote a provisional API to stable status.
