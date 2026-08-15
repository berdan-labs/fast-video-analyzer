from __future__ import annotations

from pathlib import Path

import pytest

from video_script_reconstructor.render_markdown import render_markdown, safe_fenced_block
from video_script_reconstructor.security import (
    ContainmentSnapshot,
    SecurityError,
    safe_relative_path,
    safe_slug,
    sha256_file,
    validate_remote_url,
)


def _render_project() -> dict:
    return {
        "source_title": "Hostile # title",
        "project_status": "automatically_checked",
        "generated_at_utc": "2026-01-01T00:00:00Z",
        "fidelity_mode": "verbatim",
        "primary_language": "en",
        "visual_source_available": False,
        "input_reference": "source.txt",
        "media": {"media_id": "M1", "duration_ms": None},
        "chapters": [
            {
                "chapter_id": "C001",
                "title": "Navigational",
                "start_ms": None,
                "end_ms": None,
                "block_ids": ["B000001"],
                "source_authored": False,
            }
        ],
        "script_blocks": [
            {
                "block_id": "B000001",
                "chapter_id": "C001",
                "start_ms": None,
                "end_ms": None,
                "speaker": "Speaker 1",
                "spoken_text": "# injected heading\n<script>alert(1)</script> [link](outside)",
                "visual_description": "[no visual source available]",
                "on_screen_text": [],
                "relevant_non_speech_audio": [],
                "frame_ids": [],
                "transcript_segment_ids": ["T000001"],
                "visual_event_ids": [],
                "image_claim_ids": [],
                "metadata_revision_ids": [],
                "metadata_sufficiency_decision_ids": [],
                "transformation_ids": [],
                "confidence": 1.0,
                "verification_status": "automatically_checked",
                "uncertainty": [],
            }
        ],
        "frames": [],
        "review_items": [],
        "audit": {
            "source_segment_coverage": {
                "covered": 1,
                "total": 1,
                "missing_ids": [],
                "partial_ids": [],
                "duplicate_ids": [],
            },
            "unsupported_spoken_statements": [],
            "unsupported_visual_statements": [],
            "high_impact_token_discrepancies": [],
            "blocking_failures": [],
            "final_project_status": "automatically_checked",
        },
        "transcript_source_decision": "User text",
        "corrections": [],
        "manifest": {},
    }


def test_renderer_escapes_untrusted_headings_html_and_links() -> None:
    text = render_markdown(_render_project())
    assert "\\# injected heading" in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "\\[link\\](outside)" in text
    assert text.count("## Document map") == 1
    assert "## Complete chronological analysis" in text


def test_dynamic_fence_exceeds_evidence_backticks() -> None:
    rendered = safe_fenced_block("alpha\n````\nomega", "text")
    assert rendered.startswith("`````text\n")
    assert rendered.endswith("\n`````")


def test_paths_urls_and_windows_names_are_guarded(tmp_path: Path) -> None:
    assert safe_slug("CON") == "_CON"
    root = tmp_path.resolve()
    with pytest.raises(SecurityError):
        safe_relative_path(root, "../escape.png")
    with pytest.raises(SecurityError):
        validate_remote_url("http://127.0.0.1/media.mp4")
    assert validate_remote_url("https://example.com/media.mp4").startswith("https://")


def test_containment_snapshot_detects_changes_after_cached_component_checks(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    nested = root / "nested"
    nested.mkdir()
    target = nested / "artifact.txt"
    target.write_text("before", encoding="utf-8")
    snapshot = ContainmentSnapshot()

    assert safe_relative_path(
        root,
        "nested/artifact.txt",
        root_resolved=root,
        containment_snapshot=snapshot,
    ) == target
    target.write_text("after", encoding="utf-8")
    with pytest.raises(SecurityError, match="changed during validation"):
        snapshot.verify_unchanged()


def test_sha256_file_fast_path_matches_custom_chunk_fallback(tmp_path: Path) -> None:
    payload = (b"deterministic hash payload\x00" * 200_000) + b"tail"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_file(path, chunk_size=64 * 1024)
