from __future__ import annotations

from pathlib import Path

from video_script_reconstructor.subtitle_parse import ParsedTranscriptSegment
from video_script_reconstructor.transcript_repair import (
    ExtractionResult,
    extract_interval_audio,
    repair_suspect_intervals,
)
from video_script_reconstructor.whisper_adapter import ModelIndependentASRAdapter


def seg(identifier: str, start: int, end: int, text: str) -> ParsedTranscriptSegment:
    return ParsedTranscriptSegment(identifier, start, end, "fixture", text, text)


def test_interval_extraction_honors_bounds_and_padding(tmp_path: Path) -> None:
    media = tmp_path / "input.mp4"
    media.write_bytes(b"media")
    output = tmp_path / "clip.wav"
    commands: list[tuple[str, ...]] = []

    def runner(command):
        commands.append(tuple(command))
        output.write_bytes(b"audio")

    result = extract_interval_audio(
        media, output, 1000, 2000, context_padding_ms=250, media_duration_ms=2100, runner=runner
    )
    assert (result.actual_start_ms, result.actual_end_ms) == (750, 2100)
    assert commands[0][commands[0].index("-ss") + 1] == "0.750"
    assert commands[0][commands[0].index("-t") + 1] == "1.350"


def test_selective_repair_preserves_reliable_segments_and_offsets_asr(tmp_path: Path) -> None:
    media = tmp_path / "input.mp4"
    media.write_bytes(b"media")
    original = [seg("good", 0, 900, "keep exactly"), seg("bad", 1000, 2000, "wrng token")]
    extracted_requests: list[tuple[int, int, int]] = []

    def extractor(media_path, output_path, start, end, *, context_padding_ms, media_duration_ms):
        path = Path(output_path)
        path.write_bytes(b"audio")
        extracted_requests.append((start, end, context_padding_ms))
        return ExtractionResult(path, start, end, 800, 2200, context_padding_ms, ("fixture",))

    adapter = ModelIndependentASRAdapter(
        lambda path, **kwargs: [seg("new", 200, 1200, "right token")]
    )
    outcome = repair_suspect_intervals(
        media,
        original,
        [(1000, 2000)],
        adapter,
        context_padding_ms=200,
        extractor=extractor,
        work_dir=tmp_path / "work",
    )
    assert extracted_requests == [(1000, 2000, 200)]
    assert outcome.segments[0] is original[0]
    assert outcome.segments[1].raw_text == "wrng token"
    assert outcome.segments[1].repaired_text == "right token"
    assert outcome.records[0].action == "replace"
    assert outcome.records[0].alignment_evidence["opcodes"]
