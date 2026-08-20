# Development plan

Status: active

Planning baseline: 2026-08-20, `v0.1.0`

Execution model: one human owner, one primary development agent, protected
pull-request workflow, zero required approvals, and no second administrator.

This is the primary implementation plan for developing Fast Video Analyzer.
`ROADMAP.md` summarizes direction for users; `docs/maintenance-backlog.md`
tracks repository operations. This document controls product-development
sequence, evidence, and release gates.

## Mission

Make Fast Video Analyzer the most dependable local-first way to turn long-form
video, audio, subtitles, or transcripts into one chronological, evidence-linked
Markdown project that another human or AI agent can audit without replaying the
entire source.

The product wins through fidelity, traceability, resumability, and useful local
operation—not by producing the shortest summary or the most fluent unsupported
description.

## Current baseline

The following facts were observed on 2026-08-20 and are the starting point for
this plan:

| Area | Current evidence | Development implication |
| --- | --- | --- |
| Distribution | `v0.1.0` is available through GitHub Releases and PyPI Trusted Publishing | Improve the PyPI-first onboarding path before adding broad features |
| Test contract | 553 tests are collected; unit, integration, e2e, mutation, and packaging suites are mandatory | Preserve the contract and add measurement where real-model behavior is not covered |
| Platforms | Python 3.10–3.12 CI and scheduled Windows, macOS, and Linux CLI smoke tests exist | Expand smoke coverage only for proven user journeys |
| Model coverage | Seven real-model tests exist but are intentionally outside ordinary pull-request CI | Establish a repeatable release-candidate model audit on controlled hardware |
| Fixtures | Three generated video families cover talking head, slide lecture, and screen tutorial behavior | Add licensed/reference corpora for accuracy, multilingual speech, OCR, and long-duration behavior |
| Performance | An offline benchmark harness and local GPU measurements exist | Freeze representative workloads and enforce regression budgets |
| Architecture | `pipeline.py` is about 8,500 lines and `semantic_pipeline.py` about 3,400 lines | Extract bounded stage interfaces after behavior is measured; do not perform a broad rewrite |
| Public API | `run_pipeline` is documented, but package exports and compatibility policy are minimal | Define and test a stable API before integrations depend on internals |
| User adoption | The repository has no external issue backlog yet | Dogfood real workflows and collect structured evidence before optimizing for hypothetical demand |
| Operations | Protected `main`, CI, CodeQL, release provenance, PyPI OIDC, and an encrypted restore-tested mirror are present | Keep governance stable and spend new effort on product quality |

Known planning inconsistencies to correct first:

- `README.md` still leads with installation from GitHub even though a verified
  PyPI release exists.
- Version `0.1.0` is repeated in package and CLI code instead of coming from one
  authoritative source.
- The old roadmap still lists completed work and a second maintainer, which
  conflicts with the deliberate single-administrator policy.
- Model-independent CI is strong, but release-quality claims about ASR, OCR,
  diarization, and semantic vision need repeatable real-model evidence.

## Product principles

Every development decision must preserve these rules:

1. Evidence outranks fluency. A useful `review_required` result is better than
   a polished unsupported claim.
2. Automatic checks never produce `fully_verified`; attributable human review
   remains the final authority.
3. Local and offline operation is the default. Downloads, remote media, and
   external AI are explicit opt-ins.
4. One canonical project, one Markdown report, deterministic IDs, measured
   timestamps, immutable source identity, and linked original-resolution
   evidence remain the output contract.
5. Resume, cache, and performance work may change scheduling but not evidence
   semantics.
6. Public schema or CLI changes require migration and compatibility evidence.
7. Heavy models stay optional and isolated; ordinary CI must not download them.
8. The single-owner model remains intentional. Do not add a second
   administrator or an impossible review quorum.

## Product scorecard

Before optimizing a stage, measure the applicable values below on a frozen
corpus. Each benchmark result must record source hashes, configuration, model
and revision, hardware, runtime versions, cold/warm state, and output hashes.

### Fidelity and correctness

- transcript word error rate and high-impact-token error rate;
- timestamp median and p95 absolute error;
- subtitle interval coverage, ordering, drift, and source-selection decisions;
- OCR character/word error rate, exact high-impact text accuracy, and box
  intersection quality;
- visual event recall, selected-frame usefulness, small-change retention, and
  duplicate-frame rate;
- unsupported-claim count, disputed-claim count, and citation validity;
- substantive timeline coverage and ordered-meaning coverage;
- percentage of projects ending as automatically checked, review required,
  blocked, or failed.

Release invariants are stricter than aggregate quality scores:

- zero unsupported claims in the release corpus;
- zero lost or altered high-impact tokens in reference-grounded cases;
- 100% valid evidence links, pixel hashes, metadata, chronology, and output
  contracts;
- no material WER/CER/event-recall regression without an explicit documented
  quality trade-off and owner decision.

### Performance and operability

- cold and warm wall time by stage;
- real-time factor for ASR and complete-pipeline processing;
- p50/p95 time, peak RSS/VRAM, output bytes, cache bytes, and temporary bytes;
- resume time and exact stage/checkpoint reuse after interruption;
- setup success and time-to-first-valid-report on Windows, macOS, and Linux;
- diagnostic quality: failures name the unavailable capability, safe fallback,
  and exact next action.

Once the first frozen baseline exists, an unexplained regression greater than
10% in p50 or p95 runtime, peak memory, or output footprint blocks merging.
Quality improvements may exceed that budget only when the PR records the
measured trade-off.

## Execution phases

Phases are ordered by dependency, not calendar date. A later phase may be
prototyped early, but it does not become the active priority until the earlier
exit gate is met.

## Phase 0 — Make project truth current

Objective: remove contradictions and establish one authoritative development
baseline.

### DEV-001: Distribution and documentation truth

- Change the primary install command to `pip install fast-video-analyzer`.
- Keep source-install and optional-capability instructions separate.
- Add one end-to-end first-run example for subtitle-led use and one for local
  ASR, each with expected files and status interpretation.
- Verify every documented command against an installed wheel outside the
  checkout.

Acceptance:

- a new user can install `v0.1.x`, run `doctor --offline`, analyze a generated
  fixture, validate it, and find the Markdown output using only `README.md`;
- documentation tests or a scripted smoke journey fail when a command drifts.

### DEV-002: One version authority

- Read installed version metadata from `importlib.metadata` or another single
  packaging authority.
- Remove independent hard-coded version strings from CLI and package code.
- Add source-tree and installed-wheel tests for `--version` and
  `video_script_reconstructor.__version__`.

Acceptance:

- changing the project version in the release source changes every public
  version surface;
- release tag, wheel metadata, CLI, and import version agree.

### DEV-003: Public contract inventory

- List supported CLI commands, exit codes, Python imports, schemas, environment
  variables, output paths, and compatibility aliases.
- Classify each as stable, provisional, internal, or deprecated.
- Define deprecation notice and removal policy before adding new aliases.

Acceptance:

- all stable surfaces have contract tests;
- no documentation recommends importing an internal module accidentally.

Exit gate: onboarding truth, version truth, and public-contract inventory are
merged and green on all supported platforms.

## Phase 1 — Establish a real quality baseline

Objective: measure the product on representative evidence before selecting
accuracy work.

### DEV-101: Corpus specification and provenance

Create a manifest-driven evaluation corpus with licensed, generated, or
owner-controlled sources. Media may remain outside Git when licensing or size
requires it; manifests must retain hashes and provenance.

Minimum coverage:

- clean English speech;
- Filipino and English/Filipino code-switching;
- overlapping speakers and speaker changes;
- slide lectures with small and large text;
- software tutorials with menus, code, cursors, and small UI state changes;
- talking-head video with sparse visual information;
- audio-only, transcript-only, valid/invalid subtitles, and embedded captions;
- variable-frame-rate, rotated, damaged, silent, and visual-only media;
- 30-second, 3-minute, 30-minute, and multi-hour duration classes;
- hostile subtitle/OCR text that resembles Markdown, HTML, paths, or commands.

Each reference case must declare what can be scored automatically and what
needs a bounded human review rubric.

### DEV-102: Corpus evaluator

Extend the benchmark tooling with a manifest-driven evaluator that emits both
machine-readable JSON and a compact Markdown report.

It must calculate scorecard metrics, compare against a checked-in baseline,
separate quality from performance, and refuse comparisons when model revision,
corpus hash, or scoring version is incompatible.

Acceptance:

- deterministic fixture evaluations run in ordinary CI;
- real-model evaluations can run locally without editing test code;
- a deliberately dropped segment, altered number, missed UI change, invalid
  citation, or 15% performance slowdown is visible in the report and fails the
  appropriate gate.

### DEV-103: Release-candidate model audit

Run the existing model-dependent suite plus corpus evaluation on controlled
hardware before minor releases. Keep it separate from untrusted pull-request
execution and record only non-secret environment/model evidence.

Required lanes:

- faster-whisper large-v3 on CPU and available CUDA;
- Tesseract and PP-OCRv5;
- configured diarization and forced-alignment compatibility paths;
- the default host-agent semantic bundle path;
- any local semantic provider still claimed as supported.

Acceptance:

- each lane is pass, fail, or explicitly unavailable with a reason;
- release notes link to the exact benchmark/model audit artifact;
- unavailable hardware is never represented as a passing test.

Exit gate: a frozen corpus, baseline results, and reproducible model-audit
procedure exist. All later quality priorities are selected from measured
failures or user reports.

## Phase 2 — Improve evidence quality

Objective: reduce the largest measured fidelity and review-workload failures.

### DEV-201: Transcript authority

- Improve language-aware source selection and interval-level comparison using
  benchmark evidence.
- Target Filipino/code-switched speech, names, numbers, flags, code, and noisy
  audio first.
- Measure chunk-boundary duplication/loss, timestamp drift, selective repair,
  and CPU fallback behavior.
- Preserve every raw candidate and explain the chosen intervals.

Do not replace the accuracy-first standard Whisper path with faster batched
decoding unless corpus evidence proves an acceptable trade-off.

### DEV-202: Speaker and alignment quality

- Separate anonymous speaker-turn quality from identity; never infer identity
  from voice or appearance.
- Measure diarization error, boundary error, overlap handling, and forced
  alignment accuracy.
- Make unavailable diarization an explicit capability result rather than a
  silent loss of information.

### DEV-203: OCR authority

- Evaluate PP-OCRv5 and Tesseract on slides, code, UI, rotated text, small text,
  punctuation, and multilingual content.
- Prefer exact high-impact text and defensible boxes over normalized prose.
- Improve crop/context selection only when the corpus shows it raises recall
  without breaking pixel and provenance invariants.

### DEV-204: Visual event recall

- Measure hard cuts, gradual transitions, small UI changes, before/action/after
  sequences, repeated slides, and sparse long-form scenes.
- Tune or extend selection from missed-event evidence, not isolated aesthetic
  preference.
- Keep measured PTS and original-resolution evidence authoritative.

### DEV-205: Semantic review efficiency

- Minimize unresolved packets while preserving claim conservatism.
- Improve temporal coverage, high-change prioritization, OCR context, bundle
  resumability, and reviewer ergonomics.
- Track reviewed packets per hour, duplicate review rate, citation failures,
  and post-review unsupported claims.

Exit gate: the target corpus shows a material quality or review-effort
improvement, all release invariants pass, and regressions are documented.

## Phase 3 — Performance and architectural control

Objective: reduce processing cost and internal change risk without changing
evidence semantics.

### DEV-301: Freeze performance workloads

- Select CPU-only, mid-range CUDA, and storage-constrained profiles.
- Track cold, warm, resume, batch, and validation paths.
- Record FFmpeg/model/runtime versions and distinguish overlapping stage time
  from wall time.

### DEV-302: Optimize from the critical path

Prioritize only measured bottlenecks. Candidate areas include source hashing,
ASR scheduling, survey/extraction strategy, image decode reuse, OCR batching,
semantic transport preparation, validation, and canonical JSON writes.

Every optimization PR must prove:

- identical or intentionally migrated canonical semantics;
- the same or better scorecard result;
- measured speed/memory/disk improvement on the target workload;
- correct cold, warm, interrupted, and invalidated behavior.

### DEV-303: Extract stage boundaries

Reduce concentration in `pipeline.py` and `semantic_pipeline.py` incrementally.
Start by extracting one cohesive stage with an explicit typed input/output
contract, characterization tests, and unchanged public behavior.

Preferred boundaries:

- run identity and stage-key construction;
- transcript discovery/selection/repair;
- visual survey and candidate preparation;
- canonical project transaction/finalization;
- semantic packet selection/execution/commit;
- final validation and status publication.

Do not set a line-count target as the objective. The objective is lower change
coupling, clearer invariants, and easier independent testing.

### DEV-304: Failure and interruption testing

- Add deterministic fault injection at atomic writes, worker boundaries,
  FFmpeg subprocesses, cache materialization, and semantic commits.
- Verify resume from every durable boundary.
- Verify disk-full, permission, malformed-media, missing-model, and process
  interruption behavior without false success or evidence loss.

Exit gate: performance budgets are enforced, the most volatile stages have
explicit boundaries, and interruption recovery is proven rather than inferred.

## Phase 4 — User workflow and stable integrations

Objective: make the proven pipeline easier to operate and integrate without
weakening its contract.

### DEV-401: Guided capability setup

- Make `doctor` and `plan` produce concise, copyable next actions.
- Add a non-mutating capability profile for common modes: subtitle-led,
  Whisper, multilingual OCR, and full local review.
- Keep all downloads and worker installation explicit.

### DEV-402: Review workflow ergonomics

- Improve filtering, ordering, bounded context, and batch progress for review
  queues and bundles.
- Add machine-readable summaries suitable for local tooling.
- Consider a local review UI only after CLI/bundle metrics prove where a UI
  materially reduces review time. A UI must not become another canonical
  evidence store.

### DEV-403: Stable Python API

- Export a small typed facade for plan, run, validate, review, and result
  models.
- Document sync behavior, exceptions, status semantics, and compatibility.
- Keep pipeline internals private and add installed-wheel API tests.

### DEV-404: Agent and automation integrations

- Keep `SKILL.md` aligned with the stable CLI/API.
- Add an MCP or other integration only when it can be a thin adapter over the
  stable contract and retains local/offline and evidence-safety boundaries.
- Never let an integration reinterpret visible media instructions as commands.

Exit gate: first-time setup and the complete analyze/validate/review journey
are tested end to end, and integrations depend only on stable surfaces.

## Phase 5 — 1.0 readiness

Objective: make compatibility and recovery promises sustainable.

- Freeze versioned canonical schemas and publish migration fixtures.
- Define supported input/container/subtitle matrices and tested limits.
- Complete backward-reading tests for all released project formats.
- Document deprecation, support, security, and data-retention policies in one
  user-facing location.
- Run the full corpus, model audit, platform smoke, security audit, packaging,
  clean install, backup, and restore gates.
- Require at least one real owner dogfood project to be reviewed to completion
  from an installed release candidate.

Exit gate: `1.0.0` can read supported earlier projects, produces a stable
versioned output contract, and has reproducible quality/performance evidence.

## Planned release sequence

Release numbers express contract scope, not dates:

| Release | Theme | Minimum exit evidence |
| --- | --- | --- |
| `0.1.1` | Distribution truth | PyPI-first onboarding, one version authority, installed-wheel user journey |
| `0.2.0` | Measured quality | Frozen corpus/evaluator, release-candidate model audit, first measured fidelity improvements |
| `0.3.0` | Review throughput | Lower review workload with unchanged claim and citation safety |
| `0.4.0` | Performance and architecture | Enforced performance budgets, proven interruption recovery, bounded stage extraction |
| `1.0.0` | Stable contract | Schema/API compatibility, migrations, full corpus and recovery evidence |

Patch releases may ship bounded correctness, documentation, packaging, or
security fixes without waiting for the next phase.

## Immediate PR queue

I will execute this queue in order unless a security report, regression, or
real user failure justifies reprioritization:

1. **DEV-001/002:** PyPI-first README, first-run smoke journey, and single
   version authority.
2. **DEV-003:** stable/provisional/internal public-contract inventory.
3. **DEV-101:** corpus manifest, provenance rules, reference format, and corpus
   licensing boundaries.
4. **DEV-102:** deterministic corpus evaluator and baseline comparison.
5. **DEV-103:** first complete release-candidate model audit report.
6. **DEV-201–205:** highest-severity measured quality failure.
7. **DEV-301:** frozen performance matrix and budgets.
8. **DEV-303/304:** first characterized stage extraction plus interruption
   fault injection.
9. **DEV-401/402:** measured setup or review-workflow bottleneck.
10. Prepare the next release only after its exit evidence is complete.

## Issue and pull-request protocol

Every engineering issue must contain:

- observed evidence and reproduction;
- user impact and affected contract;
- hypothesis, not an assumed cause;
- smallest viable change;
- acceptance tests and benchmark cases;
- security/privacy/compatibility implications;
- rollback or migration plan;
- explicit exclusions.

Every pull request must:

- address one bounded issue;
- update contracts/docs when behavior changes;
- add a failing-before/passing-after test for defects;
- run the repository verifier, format, lint, typing, affected tests, and any
  applicable corpus/performance gate;
- preserve protected-main and squash-only history;
- record measured results rather than adjectives such as “faster” or “better.”

## Triage and prioritization

Use this order:

1. security or privacy vulnerability;
2. evidence corruption, unsupported claim, false success, or data loss;
3. release/install breakage or inability to resume;
4. measured accuracy or high-impact-token failure;
5. severe performance/storage regression;
6. review/setup usability supported by observed friction;
7. new capability with a named user journey and acceptance corpus;
8. internal cleanup that reduces measured change risk.

Within one level, prefer the smallest change that removes the most verified
risk. Do not prioritize by novelty or code volume.

## Definition of done

A phase item is done only when:

- its acceptance evidence exists and is independently reproducible;
- mandatory and applicable model/corpus/performance suites pass;
- failure and rollback behavior is documented;
- user-facing contracts and migration notes are current;
- no open high-severity regression is deferred silently;
- `main` is green after merge;
- the change is present in the next release notes.

“Implemented,” “tested with mocks,” and “works on my machine” are progress
states, not completion evidence.

## Explicitly not now

- a broad pipeline rewrite;
- a hosted cloud service or telemetry-by-default path;
- a second administrator or mandatory external reviewer;
- automatic `fully_verified` status;
- mandatory model downloads in ordinary CI;
- a GUI before review-workflow evidence justifies it;
- new compatibility aliases without demonstrated use;
- automatic publishing or destructive retention without the existing explicit
  owner boundary;
- optimizing benchmark numbers by weakening fidelity, provenance, or review
  requirements.

## Operating cadence

- Weekly: triage issues and failed workflows, run the next bounded PR, review
  dependency noise, and refresh the immediate queue.
- Monthly: run the deterministic corpus/performance baseline, inspect model and
  cache storage, review public-contract drift, and check release readiness.
- Before a minor release: run the controlled real-model audit and clean-install
  user journey.
- Quarterly: perform the encrypted restore drill, threat-model review, schema
  compatibility audit, and roadmap reprioritization.

The human owner retains the actions that require credentials, recovery-key
custody, external account consent, irreversible publication, or destructive
data changes. All other work proceeds through evidence-backed protected pull
requests.
