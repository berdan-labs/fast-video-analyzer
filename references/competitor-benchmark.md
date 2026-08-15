# Comparable long-video tools

This is a local smoke comparison, not a claim of equivalent accuracy. Each
tool was attempted on three locally available Freight101 videos on 2026-08-16.
The samples, subtitle sidecars, and generated outputs are not part of this
repository.

| Project | License | Three-video result | Practical distinction |
|---|---|---|---|
| [`guimatheus92/mcp-video-analyzer`](https://github.com/guimatheus92/mcp-video-analyzer) | MIT | Passed: 52.0s, 79.0s, and 76.4s; 11–12 key frames per video, OCR/timeline JSON emitted | Fast MCP/JSON inspection, but the documented frame cap and token-limited completion are not a script-grade chronological evidence artifact. |
| [`byjlw/video-analyzer`](https://github.com/byjlw/video-analyzer) | Apache-2.0 | Three 30-second cutdowns completed Whisper but all vision calls returned Ollama connection errors; each still wrote `analysis.json` | Simple transcript + vision prototype; requires a running Ollama/API backend and does not make failed visual analysis a blocking result. |
| [`wenhaochai/MovieChat`](https://github.com/wenhaochai/MovieChat) | BSD-3-Clause code; model/dependency terms apply | Three attempts stopped before video inference because `omegaconf`, Conda environment, and checkpoints were unavailable | Research long-video QA system with substantial GPU/model setup rather than a portable evidence report. |
| [`PKU-YuanGroup/Video-LLaVA`](https://github.com/PKU-YuanGroup/Video-LLaVA) | Apache-2.0 code with upstream/model restrictions | Three attempts stopped before inference because `transformers`, the 7B checkpoint, and the CUDA environment were unavailable | General video-language QA; not a transcript/OCR/provenance pipeline. |

The first row was the only directly successful end-to-end comparison. The
other three were still tested on three videos each; their blockers are recorded
instead of treating an unavailable model server or checkpoint as a successful
analysis. Long Video Analyzer is intentionally positioned around a different
contract: timestamp-aligned transcript context, original-resolution evidence,
OCR, deterministic metadata and pixel checks, hash-gated semantic review, and
one portable Markdown project.
