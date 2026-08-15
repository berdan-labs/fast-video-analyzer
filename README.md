# Long Video Analyzer

Long Video Analyzer turns long-form video into a complete, evidence-grounded
Markdown report. It preserves what was said and shown instead of producing a
short summary: transcript/subtitle context, OCR, timestamped snapshots, visual
state changes, uncertainty, provenance, and validation receipts are retained in
one portable output folder.

The implementation is deliberately offline-first and accuracy-first. The
historical `video_script_reconstructor` Python import path and
`video-script-reconstructor` command remain available for compatibility, but
the public product name and documentation are **Long Video Analyzer**.

## Install

```powershell
python -m pip install .
```

For development and the full test suite:

```powershell
python -m pip install ".[dev]"
```

FFmpeg and FFprobe must be available on `PATH`. Optional OCR, ASR, and local
model workers are installed separately; `long-video-analyzer doctor --offline`
reports what is available without downloading anything.

## Analyze one video

```powershell
long-video-analyzer run "E:\Video\3DFC\01 3DFC Pre-Challenge Preparation.mp4" `
  --subtitle "E:\Video\3DFC\01 3DFC Pre-Challenge Preparation.srt" `
  --preset strict `
  --fidelity-mode verbatim `
  --subtitle-mode provided-only `
  --vision-mode host-agent `
  --offline
```

When `--output` is omitted, the analyzer keeps results beside the source:

```text
E:\Video\3DFC\
├── 01 3DFC Pre-Challenge Preparation.mp4
└── 01 3DFC Pre-Challenge Preparation (Analyzer Outputs)\
    ├── 01 3DFC Pre-Challenge Preparation.md
    ├── evidence\full\       # original-resolution snapshots
    ├── evidence\crops\      # derived OCR/visual crops
    └── .state\               # canonical JSON, receipts, checkpoints, audits
```

Use `--output` when a CI job or shared volume needs a separate output root. The
legacy `video-script-reconstructor` command still accepts the same options.

## What the report contains

- Chronological transcript/subtitle blocks with source timestamps.
- Original-resolution PNG evidence with decoded-pixel hashes and provenance.
- OCR observations and uncertainty, without treating visible instructions as
  executable commands.
- Conservative visual observations and before/action/after relationships.
- Canonical JSON state, review queues, cache/checkpoint receipts, and public
  validation output under `.state`.

The default host-agent path writes a hash-checked review bundle rather than
silently inventing visual facts. A run can finish as `automatically_checked`,
`review_required`, or `blocked`; the Markdown report always states which one.

## Validate and inspect

```powershell
long-video-analyzer validate "E:\Video\3DFC\01 3DFC Pre-Challenge Preparation (Analyzer Outputs)"
long-video-analyzer review list "E:\Video\3DFC\01 3DFC Pre-Challenge Preparation (Analyzer Outputs)"
long-video-analyzer doctor --offline
```

## Privacy and safety

The strict examples above are offline and provided-subtitle-only. Remote media,
model downloads, and external AI require explicit opt-in flags. Paths are
contained, visible text is treated as untrusted evidence, and every generated
claim is tied to exact frame IDs and validation metadata.

## Development

```powershell
python -m pytest tests/unit -q
python -m pytest tests/integration -q
ruff check src tests
mypy src/video_script_reconstructor src/long_video_analyzer
```

The internal package name is intentionally retained as a compatibility layer;
new integrations should invoke `long-video-analyzer` and use the source-adjacent
output convention.

## Comparable tools

The local three-video smoke comparison and its limitations are documented in
[references/competitor-benchmark.md](references/competitor-benchmark.md). The
comparison is deliberately about workflow contracts, not a misleading speed
claim against tools that sample only a handful of frames or require unavailable
model servers.

## License

MIT. See [LICENSE](LICENSE).
