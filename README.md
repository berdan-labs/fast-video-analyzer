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
pip install "git+https://github.com/berdan-labs/fast-video-analyzer.git"
```

### Install from source

```bash
git clone https://github.com/berdan-labs/fast-video-analyzer.git
cd fast-video-analyzer
pip install -e ".[asr,ocr]"
```

Verify your local environment:

```bash
fast-video-analyzer doctor --offline
```

---

## Quickstart

Analyze a video with an existing subtitle file:

```bash
fast-video-analyzer run "path/to/video.mp4" \
  --subtitle "path/to/video.srt" \
  --preset strict \
  --offline
```

If no subtitles are provided, run with offline Whisper ASR:

```bash
fast-video-analyzer run "path/to/video.mp4" \
  --subtitle-mode force-asr \
  --preset strict \
  --offline
```

## Python API

```python
from pathlib import Path
from video_script_reconstructor.pipeline import run_pipeline

result = run_pipeline(
    input_value=Path("recording.mp4"),
    output_root=Path("outputs"),
    subtitles=[Path("recording.srt")],
    preset="strict",
)

print(f"Report: {result.markdown_path}")
print(f"Output directory: {result.project_dir}")
print(f"Status: {result.status}")
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

---

## Privacy and security

Media processing, frame extraction, and local model inference run without
telemetry or cloud calls. Subtitles and OCR text are treated as untrusted input
and escaped in Markdown deliverables.

---

## Development

```bash
pytest tests/unit -q
pytest tests/integration -q

ruff check src tests
mypy src/video_script_reconstructor
```

---

## License

[MIT License](LICENSE)
