# Fast Video Analyzer

Fast Video Analyzer analyzes videos into structured Markdown notes for LLMs and
AI agents. It uses Whisper ASR to transcribe speech, OCR to read text on screen,
and scene detection to select the moments that matter.

The result is a time-ordered note with the transcript, scene screenshots, and
on-screen text in one place. Supporting files stay beside the note, so a person
or downstream tool can trace a finding back to the source video.

## What you get

- A Markdown note organised by time.
- Supplied subtitles or a locally generated Whisper transcript.
- Screenshots of selected scenes and crops of on-screen text.
- A local output folder containing the images and data needed to review the
  result.

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
