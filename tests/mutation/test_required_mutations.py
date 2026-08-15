from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, PngImagePlugin

from video_script_reconstructor.audit import audit_project
from video_script_reconstructor.cache import cache_key
from video_script_reconstructor.frame_selection import deduplication_decision
from video_script_reconstructor.image_claims import claim_has_independent_support
from video_script_reconstructor.image_metadata import (
    MetadataSecurityError,
    embed_metadata,
    read_embedded_metadata,
    validate_metadata_security,
)
from video_script_reconstructor.metadata_reconcile import (
    StaleBaseRevisionError,
    reconcile_observation,
)
from video_script_reconstructor.pipeline import run_pipeline
from video_script_reconstructor.render_markdown import render_markdown
from video_script_reconstructor.schemas import (
    EvidenceRegion,
    ImageClaim,
    VisualAnalysisObservation,
)
from video_script_reconstructor.validate_output import validate_project

ROOT = Path(__file__).resolve().parents[2]


def _audit_base() -> dict:
    return {
        "media": {"duration_ms": 5_000},
        "project_status": "automatically_checked",
        "transcript_segments": [
            {
                "segment_id": "T000001",
                "raw_text": "Dog bites man with tool --strict and value 42.",
                "normalized_text": "Dog bites man with tool --strict and value 42.",
                "substantive": True,
            }
        ],
        "script_blocks": [
            {
                "block_id": "B000001",
                "start_ms": 0,
                "end_ms": 1_000,
                "spoken_text": "Dog bites man with tool --strict and value 42.",
                "visual_description": "",
                "transcript_segment_ids": ["T000001"],
                "frame_ids": [],
                "visual_event_ids": [],
                "image_claim_ids": [],
                "verification_status": "automatically_checked",
            }
        ],
        "frames": [],
        "visual_events": [],
        "review_items": [],
        "sufficiency_decisions": [],
        "final_signoffs": [],
    }


@pytest.fixture(scope="module")
def base_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("mutation-base")
    fixtures = root / "fixtures"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_fixtures.py"), str(fixtures)],
        check=True,
    )
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=root / "output",
        subtitles=[fixtures / "slide-lecture.srt"],
    )
    assert result.validation is not None and result.validation.valid
    return result.project_dir


def _clone(base_project: Path, tmp_path: Path) -> Path:
    target = tmp_path / "project"
    shutil.copytree(base_project, target)
    return target


def _canonical(project: Path) -> tuple[Path, dict]:
    path = project / ".state" / "canonical-project.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_mutation_01_dropped_transcript_segment_is_identified() -> None:
    project = _audit_base()
    project["transcript_segments"] = []
    report = audit_project(project)
    assert "timeline_errors" in report["blocking_failures"]
    assert any(
        "B000001: cites missing transcript segment T000001" == error
        for error in report["timeline_errors"]
    )


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [
        ("Dog bites man with tool --strict.", "partial_or_reordered_segments"),
        (
            "Dog bites man with tool --strict and value 42. Unsupported sentence.",
            "partial_or_reordered_segments",
        ),
        ("Man bites dog with tool --strict and value 42.", "partial_or_reordered_segments"),
        ("Dog bites man with tool --strict and value 43.", "high_impact_token_discrepancy"),
        ("Dog bites man with tool --fast and value 42.", "high_impact_token_discrepancy"),
    ],
)
def test_mutations_02_to_06_fidelity_changes_fail(replacement: str, expected: str) -> None:
    project = _audit_base()
    project["script_blocks"][0]["spoken_text"] = replacement
    report = audit_project(project)
    assert expected in report["blocking_failures"]
    assert "T000001" in (
        report["source_segment_coverage"]["partial_ids"]
        or [item["segment_id"] for item in report["high_impact_token_discrepancies"]]
    )


def test_mutation_07_corrupt_timestamp_identifies_block() -> None:
    project = _audit_base()
    project["script_blocks"][0]["end_ms"] = 9_000
    report = audit_project(project)
    assert "timeline_errors" in report["blocking_failures"]
    assert any("B000001" in item for item in report["timeline_errors"])


def test_mutation_08_deleted_image_fails_link_check(base_project: Path, tmp_path: Path) -> None:
    project = _clone(base_project, tmp_path)
    image = next((project / "evidence" / "full").glob("*.png"))
    image.unlink()
    result = validate_project(project)
    assert any("missing evidence/" in error or "cannot decode" in error for error in result.errors)


def test_mutation_09_broken_image_link_fails(base_project: Path, tmp_path: Path) -> None:
    project = _clone(base_project, tmp_path)
    markdown = next(project.glob("*.md"))
    text = markdown.read_text(encoding="utf-8").replace(
        "evidence/full/", "evidence/full/missing-", 1
    )
    markdown.write_text(text, encoding="utf-8")
    result = validate_project(project)
    assert any("missing evidence/full/missing-" in error for error in result.errors)


def test_mutation_10_broken_anchor_fails(base_project: Path, tmp_path: Path) -> None:
    project = _clone(base_project, tmp_path)
    markdown = next(project.glob("*.md"))
    text = markdown.read_text(encoding="utf-8").replace(
        '<a id="B000001"></a>', '<a id="BROKEN"></a>'
    )
    markdown.write_text(text, encoding="utf-8")
    result = validate_project(project)
    assert any("missing anchor target #B000001" in error for error in result.errors)


def test_mutation_11_orphan_final_image_fails(base_project: Path, tmp_path: Path) -> None:
    project = _clone(base_project, tmp_path)
    source = next((project / "evidence" / "full").glob("*.png"))
    shutil.copy2(source, source.with_name("F999999__00h00m00s000__full.png"))
    result = validate_project(project)
    assert any("orphan final images" in error and "F999999" in error for error in result.errors)


def test_mutation_12_crop_without_full_parent_fails(base_project: Path, tmp_path: Path) -> None:
    project = _clone(base_project, tmp_path)
    path, canonical = _canonical(project)
    canonical["frames"][0]["parent_full_frame_id"] = "F999999"
    _write(path, canonical)
    result = validate_project(project)
    assert any(
        "lacks parent F999999" in error or "canonical_schema" in error for error in result.errors
    )


def test_mutation_13_one_character_change_survives_dedup(tmp_path: Path) -> None:
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    for path, value in ((before, "42"), (after, "43")):
        image = Image.new("RGB", (240, 100), "white")
        ImageDraw.Draw(image).text((80, 35), value, fill="black")
        image.save(path)
    decision = deduplication_decision(
        before, after, left_ocr="42", right_ocr="43", protect_small_changes=True
    )
    assert decision.is_duplicate is False
    assert "ocr_change" in decision.protected_reasons


def test_mutation_14_requested_time_cannot_replace_measured_actual(
    base_project: Path, tmp_path: Path
) -> None:
    project = _clone(base_project, tmp_path)
    path, canonical = _canonical(project)
    frame = canonical["frames"][0]
    frame["requested_ms"] = frame["actual_ms"] + 123
    frame["actual_ms"] = frame["requested_ms"]
    frame["offset_ms"] = 0
    _write(path, canonical)
    result = validate_project(project)
    assert any(
        "actual time disagrees with raw PTS" in error or "embedded payload disagrees" in error
        for error in result.errors
    )


def test_mutation_15_unsupported_visual_description_fails() -> None:
    project = _audit_base()
    block = project["script_blocks"][0]
    block["frame_ids"] = ["F000001"]
    block["visual_description"] = "A button was clicked and succeeded."
    report = audit_project(project)
    assert "unsupported_statements" in report["blocking_failures"]
    assert "B000001" in report["unsupported_visual_statements"]


def test_mutation_16_semantic_pending_cannot_be_auto_verified() -> None:
    project = _audit_base()
    block = project["script_blocks"][0]
    block["frame_ids"] = ["F000001"]
    block["visual_description"] = "[visual evidence retained; semantic description pending review]"
    block["verification_status"] = "automatically_checked"
    report = audit_project(project)
    assert "unsupported_statements" in report["blocking_failures"]
    assert "B000001" in report["unsupported_visual_statements"]


def test_mutation_17_fully_verified_requires_human_signoff() -> None:
    project = _audit_base()
    project["project_status"] = "fully_verified"
    report = audit_project(project)
    assert "fully_verified_without_human_signoff" in report["blocking_failures"]


def test_mutation_18_sidecar_hash_invalidates_cache() -> None:
    first = cache_key("media", "sidecar-a", "strict")
    second = cache_key("media", "sidecar-b", "strict")
    assert first != second


def test_mutation_19_hostile_subtitle_is_escaped() -> None:
    project = _audit_base()
    project.update(
        {
            "source_title": "Hostile",
            "generated_at_utc": "2026-01-01T00:00:00Z",
            "fidelity_mode": "verbatim",
            "primary_language": "en",
            "visual_source_available": False,
            "input_reference": "hostile.srt",
            "chapters": [
                {
                    "chapter_id": "C001",
                    "title": "Navigation",
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "block_ids": ["B000001"],
                }
            ],
            "audit": audit_project(project),
            "transcript_source_decision": "test",
            "corrections": [],
            "manifest": {},
        }
    )
    project["script_blocks"][0].update(
        {
            "chapter_id": "C001",
            "spoken_text": "# injected\n<script>x</script>",
            "visual_description": "[no visual source available]",
            "on_screen_text": [],
            "relevant_non_speech_audio": [],
            "metadata_revision_ids": [],
            "metadata_sufficiency_decision_ids": [],
            "transformation_ids": [],
            "uncertainty": [],
            "confidence": 1.0,
        }
    )
    text = render_markdown(project)
    assert "\\# injected" in text
    assert "&lt;script&gt;x&lt;/script&gt;" in text


def test_mutation_20_evidence_link_cannot_escape_project(
    base_project: Path, tmp_path: Path
) -> None:
    project = _clone(base_project, tmp_path)
    markdown = next(project.glob("*.md"))
    text = markdown.read_text(encoding="utf-8").replace("evidence/full/", "../", 1)
    markdown.write_text(text, encoding="utf-8")
    result = validate_project(project)
    assert any(
        "Path escapes project root" in error or "Unsafe relative path" in error
        for error in result.errors
    )


def test_mutation_21_stripped_metadata_fails(base_project: Path, tmp_path: Path) -> None:
    project = _clone(base_project, tmp_path)
    path = next((project / "evidence" / "full").glob("*.png"))
    with Image.open(path) as image:
        pixels = image.copy()
    pixels.save(path, "PNG")
    result = validate_project(project)
    assert any("has no 'video-script-reconstructor'" in error for error in result.errors)


def test_mutation_22_embedded_id_change_fails_canonical_match(
    base_project: Path, tmp_path: Path
) -> None:
    project = _clone(base_project, tmp_path)
    image = next((project / "evidence" / "full").glob("*.png"))
    payload = read_embedded_metadata(image).model_dump(mode="json")
    payload["image"]["image_id"] = "F999999"
    embed_metadata(image, payload)
    result = validate_project(project)
    assert any("disagrees with the canonical-state mirror" in error for error in result.errors)


def test_mutation_23_pixel_change_without_hash_update_fails(
    base_project: Path, tmp_path: Path
) -> None:
    project = _clone(base_project, tmp_path)
    image_path = next((project / "evidence" / "full").glob("*.png"))
    with Image.open(image_path) as image:
        image.load()
        payload_text = image.text["video-script-reconstructor"]
        description = image.text["Description"]
        changed = image.convert("RGBA")
    changed.putpixel((0, 0), (255, 0, 255, 255))
    info = PngImagePlugin.PngInfo()
    info.add_itxt("video-script-reconstructor", payload_text, zip=False)
    info.add_text("Description", description, zip=False)
    changed.save(image_path, "PNG", pnginfo=info)
    result = validate_project(project)
    assert any("pixel hash mismatch" in error for error in result.errors)


def test_mutation_24_canonical_payload_change_without_image_fails(
    base_project: Path, tmp_path: Path
) -> None:
    project = _clone(base_project, tmp_path)
    path, canonical = _canonical(project)
    canonical["evidence_image_metadata"][0]["knowledge"]["selection_reason"] = (
        "Mutated ledger only."
    )
    _write(path, canonical)
    result = validate_project(project)
    assert any(
        "canonical-state mirror" in error or "canonical_schema" in error for error in result.errors
    )


def test_mutation_25_stale_base_cannot_overwrite_newer_evidence() -> None:
    observation = VisualAnalysisObservation(
        observation_id="VA000001",
        image_ids=["F000001"],
        base_revision_id="MR000001",
        actor_kind="host_agent",
        actor_label="host",
        observed_at_utc="2026-01-01T00:00:00Z",
        purpose="test stale base",
        prior_metadata_visible=True,
        rationale="bounded test",
        validation_result="accepted",
    )
    with pytest.raises(StaleBaseRevisionError):
        reconcile_observation(
            [],
            observation,
            revision_id="MR000003",
            current_revision_id="MR000002",
            allow_stale_reconcile=False,
        )


def test_mutation_26_deleted_earlier_observation_breaks_revision(
    base_project: Path, tmp_path: Path
) -> None:
    project = _clone(base_project, tmp_path)
    path, canonical = _canonical(project)
    canonical["metadata_revisions"][0]["observation_ids"] = ["VA000001"]
    canonical["visual_observations"] = []
    _write(path, canonical)
    result = validate_project(project)
    assert any("cites missing observation VA000001" in error for error in result.errors)


def test_mutation_27_repeated_same_model_is_not_independent() -> None:
    region = EvidenceRegion(image_id="F000001", whole_frame_basis=True)
    claim = ImageClaim(
        claim_id="IC000001",
        claim_class="exact_text",
        statement="Value 42 is visible.",
        status="supported",
        importance="high_impact",
        high_impact_token=True,
        supporting_image_ids=["F000001"],
        evidence_regions=[region],
        supporting_observation_ids=["VA000001", "VA000002"],
    )
    common = {
        "image_ids": ["F000001"],
        "actor_kind": "multimodal_model",
        "actor_label": "same",
        "provider": "p",
        "model": "m",
        "model_version": "1",
        "prompt_template_hash": "same",
        "observed_at_utc": "2026-01-01T00:00:00Z",
        "purpose": "repeat",
        "analysis_depth": "cumulative",
        "prior_metadata_visible": True,
        "rationale": "same evidence",
        "validation_result": "accepted",
    }
    observations = [
        VisualAnalysisObservation(observation_id="VA000001", **common),
        VisualAnalysisObservation(observation_id="VA000002", **common),
    ]
    assert claim_has_independent_support(claim, observations) is False


def test_mutation_28_metadata_fact_without_claim_citation_fails(
    base_project: Path, tmp_path: Path
) -> None:
    project = _clone(base_project, tmp_path)
    path, canonical = _canonical(project)
    block = canonical["script_blocks"][0]
    block["visual_description"] = "A consequential metadata-only fact."
    block["image_claim_ids"] = []
    _write(path, canonical)
    result = validate_project(project)
    assert any("lacks an image-claim citation" in error for error in result.errors)


@pytest.mark.parametrize(
    "malicious",
    [
        {"value": "api_key=sk-live_abcdefghijklmnopqrstuvwxyz"},
        {"value": "C:\\Users\\private\\secret.txt"},
        {"value": "x" * 2_000_000},
    ],
)
def test_mutation_29_malicious_portable_metadata_is_rejected(malicious: dict) -> None:
    with pytest.raises(MetadataSecurityError):
        validate_metadata_security(malicious, max_payload_bytes=1_000_000)


def test_mutation_30_false_sufficiency_with_uninspected_action_fails() -> None:
    project = _audit_base()
    project["sufficiency_decisions"] = [
        {
            "decision_id": "MS000001",
            "status": "sufficient",
            "unattempted_evidence_actions": ["Inspect adjacent frame F000002"],
        }
    ]
    report = audit_project(project)
    assert "false_metadata_sufficiency" in report["blocking_failures"]
