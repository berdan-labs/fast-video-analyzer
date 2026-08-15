from __future__ import annotations

from video_script_reconstructor.pipeline import (
    _can_use_visual_only_fallback,
    _infer_primary_language,
    _select_asr_candidate_set,
)


def _candidate(identifier: str, origin: str) -> dict[str, object]:
    return {
        "candidate_id": identifier,
        "source_type": "local_asr",
        "origin": origin,
        "language": "fil",
        "authorship": "auto_generated",
        "segments": [
            {
                "segment_id": f"{identifier}-T1",
                "start_ms": 0,
                "end_ms": 1_000,
                "raw_text": "kumusta",
                "normalized_text": "kumusta",
                "substantive": True,
            }
        ],
    }


def test_explicit_whisper_preference_wins_when_candidate_is_usable() -> None:
    candidates = [_candidate("TC000001", "moss-transcribe-diarize"), _candidate("TC000002", "faster-whisper")]

    segments, decision, disagreements = _select_asr_candidate_set(
        candidates,
        media_duration_ms=1_000,
        expected_language="fil",
        preferred_backend="faster-whisper",
    )

    assert segments[0]["raw_text"] == "kumusta"
    assert "TC000002" in decision
    assert "Explicit preference" in decision
    assert disagreements == []


def test_primary_language_inference_normalizes_filipino_and_rejects_mixed_audio() -> None:
    segments = [
        {"language": "tl", "normalized_text": "Magandang araw sa inyong lahat."},
        {"language": "fil-PH", "normalized_text": "Ito ang ating aralin."},
        {"language": "en", "normalized_text": "short"},
    ]
    assert _infer_primary_language(segments) == "tl"
    assert _infer_primary_language(
        [
            {"language": "en", "normalized_text": "English content."},
            {"language": "tl", "normalized_text": "Nilalamang Tagalog."},
        ]
    ) is None


def test_primary_language_inference_can_use_word_labels_when_segment_label_missing() -> None:
    segments = [
        {
            "normalized_text": "Mabuhay tayo",
            "words": [
                {"language": "tl", "text": "Mabuhay"},
                {"language": "tl", "text": "tayo"},
            ],
        }
    ]
    assert _infer_primary_language(segments) == "tl"


def test_visual_only_fallback_requires_all_asr_candidates_to_be_empty() -> None:
    common = {
        "kind": "video",
        "segments": [],
        "asr_completed_without_segments": True,
        "asr_had_failure": False,
    }
    assert _can_use_visual_only_fallback(
        **common,
        asr_produced_segments=False,
    )
    assert not _can_use_visual_only_fallback(
        **common,
        asr_produced_segments=True,
    )
