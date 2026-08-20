# Fast Video Analyzer

Fast Video Analyzer turns a video into a chronological record of its spoken
content, visible text, and representative frames. It uses supplied subtitles or
Whisper ASR for speech, OCR for text in frames, and scene detection to choose
where to capture screenshots.

Each run writes one Markdown report and a folder of linked screenshots, crops,
and supporting data. Read it to review a recording without repeatedly scrubbing
through the video, or use it as source material for an LLM or AI agent.

## Output

- A time-ordered report with timestamps, transcript blocks, visible text, and
  selected frames.
- Supplied subtitles or a locally generated Whisper transcript.
- Full-size screenshots and OCR crops linked from the report.
- A project folder containing the report, images, and validation data for later
  review.

---

## Installation

### Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **FFmpeg & FFprobe**: Must be available on your system `PATH`.

### Install

```bash
python -m pip install --upgrade fast-video-analyzer
```

This installs the published package from PyPI. The base package supports the
subtitle-led workflow below and does not download model weights. Install an
optional capability only when you need it:

```bash
python -m pip install "fast-video-analyzer[asr]"  # local Whisper ASR
python -m pip install "fast-video-analyzer[ocr]"  # Python OCR wrapper
```

The `asr` extra still requires a locally available, verified model before an
offline ASR run. The `ocr` extra still requires a supported OCR executable.
Use `fast-video-analyzer models list` and `fast-video-analyzer doctor --offline`
to inspect capability readiness; optional model downloads are always explicit.

### Install from source

```bash
git clone https://github.com/berdan-labs/fast-video-analyzer.git
cd fast-video-analyzer
python -m pip install -e ".[asr,ocr]"
```

Verify your local environment:

```bash
fast-video-analyzer --version
fast-video-analyzer doctor --offline
```

For a concise operator view without machine-specific paths, add `--summary`.
The default command remains the full diagnostic JSON for support and
troubleshooting:

```bash
fast-video-analyzer doctor --offline --summary
```

Create a support bundle when asking for help. It contains sanitized capability
metadata only; it does not copy source media, transcripts, screenshots,
generated projects, credentials, or filesystem paths:

```bash
fast-video-analyzer diagnostic-bundle --output fast-video-analyzer-diagnostic.zip
```

---

## First successful run

The subtitle-led path needs no model download. Run the commands from a clean
working directory and replace the example paths with your own files:

```bash
fast-video-analyzer doctor --offline
fast-video-analyzer run "path/to/video.mp4" --subtitle "path/to/video.srt" --output "path/to/analyzer-output" --preset strict --offline
fast-video-analyzer validate "path/to/analyzer-output/video"
```

The `run` command writes JSON to standard output. Use its `project_dir` and
`markdown` fields to find the result. A `review_required` status and exit
code `3` mean the evidence was produced but still needs human review; they are
not the same as a failed or invalid project. The final `validate` command
should exit `0` and report `"valid": true`.

The output option is a root: the project directory is created below it using
the source video stem. In the example above, it is
`path/to/analyzer-output/video`.

The output directory contains one Markdown report plus its evidence and state:

```text
path/to/analyzer-output/
└── video/
    ├── video.md
    ├── evidence/
    └── .state/
```

### Local ASR workflow

After installing the `asr` and `models` extras, prepare a verified local model
while network access is explicitly allowed:

```bash
python -m pip install "fast-video-analyzer[asr,models]"
fast-video-analyzer models fetch faster-whisper-large-v3
fast-video-analyzer models verify faster-whisper-large-v3
fast-video-analyzer run "path/to/video.mp4" --subtitle-mode force-asr --output "path/to/analyzer-output" --preset strict --offline
```

If an ASR run is interrupted, rerun the same command with the same output
root. The resumable transcript checkpoints are retained, and the run manifest
records the interrupted transcript stage instead of presenting a false success.

`models fetch` is the explicit network-enabled preparation step; do not run it
when working in a network-denied environment. Once the model is verified,
`--offline` prevents the analysis run from downloading anything.

The installed wheel also keeps the historical entrypoints working:

```bash
long-video-analyzer doctor --offline
video-script-reconstructor doctor --offline
```

All three entrypoints invoke the same parser and implementation. Nested
compatibility aliases such as `review bundle batch-create` and
`review bundle create-batch` are covered by the CLI compatibility tests.

## Python API (stable)

The synchronous facade below is the supported library seam for one-input
tooling. It plans, runs, validates, and inspects review items without exposing
pipeline stages, provider adapters, or persisted JSON dictionaries. Results
are immutable typed snapshots; `review_required` and `blocked` are returned as
statuses rather than being mistaken for successful completion. See the
[Python API reference](docs/python-api.md) and the
[public contract inventory](docs/public-contracts.md) for compatibility and
exception rules.

```python
from pathlib import Path
from video_script_reconstructor.api import list_review_items, run, validate

result = run(
    Path("recording.mp4"),
    output_root=Path("outputs"),
    subtitles=[Path("recording.srt")],
    preset="strict",
    offline=True,
)

print(f"Report: {result.markdown_path}")
print(f"Output directory: {result.project_dir}")
print(f"Status: {result.status}")
report = validate(result.project_dir)
if result.status == "review_required":
    for item in list_review_items(result.project_dir):
        print(item.review_id, item.required_action)
```

---

## Output structure

Outputs are written alongside the source video by default:

```text
<video_stem> (Analyzer Outputs)/
├── <video_stem>.md       # Chronological Markdown notes with linked evidence
├── evidence/
│   ├── full/            # Full-resolution scene keyframes
│   └── crops/           # OCR bounding crops (code, slides, text)
└── .state/              # JSON state manifests, checksums, and audit receipts
```

---

## Validation and review

Verify output integrity against timeline rules and image pixel hashes:

```bash
fast-video-analyzer validate "path/to/video (Analyzer Outputs)"
fast-video-analyzer review list "path/to/video (Analyzer Outputs)"
```

Before a first run, `plan --summary` prints the selected workflow, estimated
evidence/storage, prerequisites, and copyable run/validate commands without
processing the media:

```bash
fast-video-analyzer plan "path/to/video.mp4" \
  --subtitle "path/to/video.srt" --offline --summary
```

If the run returns `review_required` (exit code `3`), continue with the
copyable no-copy bundle handoff in
[docs/review-workflow.md](docs/review-workflow.md). It explains how to inspect
review IDs, create bounded host-agent requests, apply attributable responses,
and perform the final human sign-off without copying source media into the
handoff.

---

## Privacy and security

Media processing, frame extraction, and local model inference run without
telemetry or cloud calls. Subtitles and OCR text are treated as untrusted input
and escaped in Markdown deliverables.

---

## Development

```bash
uv sync --locked --extra dev
uv run python scripts/verify_repo.py
uv run ruff format --check scripts/verify_repo.py
uv run ruff check src tests scripts
uv run mypy src/video_script_reconstructor
uv run pytest tests/unit tests/integration -q
```

The full mandatory acceptance gate also includes the end-to-end, mutation, and
packaging suites:

```bash
uv run pytest tests/e2e tests/mutation tests/packaging -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [OPERATIONS.md](OPERATIONS.md),
[docs/releasing.md](docs/releasing.md), [docs/runbooks.md](docs/runbooks.md),
[docs/corpus-evaluation.md](docs/corpus-evaluation.md), and
[SUPPORT.md](SUPPORT.md) for maintainer and contributor workflows.

---

## License

[MIT License](LICENSE)
