from __future__ import annotations

from pathlib import Path

from video_script_reconstructor.media_probe import MediaProbeResult, MediaStream
from video_script_reconstructor.subtitle_parse import ParsedTranscriptSegment
from video_script_reconstructor.subtitle_sources import (
    build_embedded_subtitle_extract_command,
    discover_embedded_subtitle_tracks,
    extract_embedded_subtitle_track,
    rank_candidates,
    select_candidate_intervals,
)


def seg(identifier: str, start: int, end: int, text: str) -> ParsedTranscriptSegment:
    return ParsedTranscriptSegment(identifier, start, end, "fixture", text, text)


def test_ranking_uses_quality_not_blind_provenance() -> None:
    human = {
        "candidate_id": "human",
        "source_type": "user_human_transcript",
        "origin": "human.srt",
        "segments": [
            seg("h1", 0, 100, "..."),
            seg("h2", 100, 200, "..."),
            seg("h3", 200, 300, "..."),
        ],
    }
    asr = {
        "candidate_id": "asr",
        "source_type": "local_asr",
        "origin": "local",
        "segments": [seg("a1", 0, 3000, "A complete reliable sentence.")],
    }
    ranked = rank_candidates([human, asr], media_duration_ms=3000)
    assert ranked[0].candidate_id == "asr"
    assert all(candidate.quality_metrics for candidate in ranked)


def test_user_subtitle_outranks_equally_reliable_embedded_track() -> None:
    user = {
        "candidate_id": "user",
        "source_type": "user_subtitle",
        "origin": "provided.srt",
        "segments": [seg("u1", 0, 3000, "Exact reliable sentence.")],
    }
    embedded = {
        "candidate_id": "embedded",
        "source_type": "embedded_human_subtitle",
        "origin": "media.mkv#stream=2",
        "segments": [seg("e1", 0, 3000, "Exact reliable sentence.")],
    }
    ranked = rank_candidates([embedded, user], media_duration_ms=3000)
    assert ranked[0].candidate_id == "user"


def test_interval_selection_chooses_one_candidate_per_interval() -> None:
    first = {
        "candidate_id": "first",
        "source_type": "user_subtitle",
        "origin": "one.srt",
        "segments": [seg("a", 0, 1000, "First interval."), seg("b", 4000, 5000, "Last interval.")],
    }
    second = {
        "candidate_id": "second",
        "source_type": "automatic_caption",
        "origin": "two.vtt",
        "segments": [seg("c", 0, 5000, "Continuous alternate caption track.")],
    }
    ranked = rank_candidates([first, second], media_duration_ms=5000)
    selections = select_candidate_intervals(ranked, media_duration_ms=5000)
    assert selections
    assert all(selection.end_ms > selection.start_ms for selection in selections)
    assert len({(selection.start_ms, selection.end_ms) for selection in selections}) == len(
        selections
    )


def test_embedded_track_discovery_preserves_language_disposition_and_support() -> None:
    probe = MediaProbeResult(
        source_path=Path("fixture.mkv"),
        duration_ms=5000,
        size_bytes=1,
        container="matroska",
        bit_rate=None,
        streams=(
            MediaStream(
                2,
                "subtitle",
                "subrip",
                5000,
                0,
                "1/1000",
                language="eng",
                tags={"title": "Human authored"},
                disposition={"default": 1},
            ),
            MediaStream(
                3,
                "subtitle",
                "hdmv_pgs_subtitle",
                5000,
                0,
                "1/1000",
                language="jpn",
                tags={},
                disposition={"forced": 1},
            ),
        ),
        chapters=(),
        source_metadata={},
    )
    tracks = discover_embedded_subtitle_tracks(probe.source_path, probe=probe)
    assert tracks[0].supported
    assert tracks[0].source_type == "embedded_human_subtitle"
    assert tracks[0].language == "eng"
    assert tracks[0].disposition["default"] == 1
    assert not tracks[1].supported
    assert "not a supported text" in str(tracks[1].unsupported_reason)


def test_embedded_extract_uses_safe_argv_and_atomic_raw_output(tmp_path: Path) -> None:
    source = tmp_path / "hostile; name [x].mkv"
    source.write_bytes(b"media")
    probe = MediaProbeResult(
        source_path=source,
        duration_ms=1000,
        size_bytes=5,
        container="matroska",
        bit_rate=None,
        streams=(
            MediaStream(
                4,
                "subtitle",
                "ass",
                1000,
                0,
                "1/1000",
                language="en",
                tags={"title": "Official"},
                disposition={"default": 1},
            ),
        ),
        chapters=(),
        source_metadata={},
    )
    track = discover_embedded_subtitle_tracks(source, probe=probe)[0]
    command = build_embedded_subtitle_extract_command(source, track, tmp_path / "out.ass")
    assert command[command.index("-i") + 1] == str(source)
    assert command[command.index("-map") + 1] == "0:4"
    captured: list[list[str]] = []

    def runner(argv):
        captured.append(list(argv))
        Path(argv[-1]).write_text(
            "[Events]\nFormat: Start, End, Text\nDialogue: 0:00:00.00,0:00:01.00,Exact\n",
            encoding="utf-8",
        )

    extracted = extract_embedded_subtitle_track(source, track, tmp_path / "raw", runner=runner)
    assert extracted.path.is_file()
    assert extracted.path.read_text(encoding="utf-8").endswith("Exact\n")
    assert captured[0][captured[0].index("-i") + 1] == str(source)
