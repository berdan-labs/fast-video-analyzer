from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

import video_script_reconstructor.evidence as evidence_module
import video_script_reconstructor.semantic_pipeline as semantic_module
import video_script_reconstructor.subagent_review as subagent_module
from video_script_reconstructor.errors import ValidationFailure
from video_script_reconstructor.evidence import ingest_project_observation, verify_image_metadata
from video_script_reconstructor.frame_quality import perceptual_dhash
from video_script_reconstructor.image_metadata import read_embedded_metadata
from video_script_reconstructor.ocr import OCRAdapter, OCRObservation, OCRToken
from video_script_reconstructor.pipeline import run_pipeline
from video_script_reconstructor.providers.base import ProviderDescriptor, VisionProvider
from video_script_reconstructor.resource_usage import resource_snapshot
from video_script_reconstructor.review import apply_review, finalize_project
from video_script_reconstructor.security import sha256_file
from video_script_reconstructor.semantic_pipeline import (
    apply_vision_provider,
    pending_packet_count,
    run_semantic_batch,
    run_semantic_pass,
)
from video_script_reconstructor.subagent_review import apply_review_bundle
from video_script_reconstructor.validate_output import (
    read_trusted_validation_receipt,
    validate_project,
)
from video_script_reconstructor.vision_packets import VisionAnnotation, VisionPacket

REPOSITORY = Path(__file__).resolve().parents[2]


class PixelStateOCRAdapter(OCRAdapter):
    """Deterministic CI adapter that reads the decoded fixture's real state color."""

    def available(self) -> bool:
        return True

    def recognize(
        self,
        image_path: str | Path,
        *,
        frame_id: str,
        observation_id: str,
        crop_id: str | None = None,
        language: str | None = None,
    ) -> OCRObservation:
        with Image.open(image_path) as image:
            red, green, _blue = image.convert("RGB").getpixel((70, 270))
        text = "ENABLED" if green - red > 40 else "DISABLED"
        return OCRObservation(
            observation_id=observation_id,
            frame_id=frame_id,
            crop_id=crop_id,
            bounding_region=(60, 230, 200, 55),
            raw_engine_text=f" {text}\n",
            normalized_interpretation=text,
            confidence=0.99,
            alternatives=(),
            language=language or "eng",
            uncertain_characters=(),
            engine="fixture-pixel-state-ocr",
            engine_version="1",
            human_decision=None,
            tokens=(OCRToken(text, 99.0, (60, 230, 200, 55), 1, 1, 1, 1, 1),),
        )


class FixtureVisionProvider(VisionProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._descriptor = ProviderDescriptor(
            provider_id="fixture-local-vision",
            route="local",
            model="fixture-vlm",
            model_version="revision-1",
            adapter_version="1.0",
            network_required=False,
        )

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor

    def annotate(self, packet: VisionPacket, *, project_root: Path) -> VisionAnnotation:
        self.calls.append(packet.candidate_id)
        focus = next(
            (frame for frame in packet.frames if frame.role == "focus"),
            packet.frames[0],
        )
        assert (project_root / focus.path).is_file()
        return VisionAnnotation(
            candidate_id=packet.candidate_id,
            factual_visible_description=f"A colored slide is directly visible in {focus.frame_id}.",
            event_type="slide_state",
            evidence_frame_ids=[focus.frame_id],
            before_action_after_roles={focus.frame_id: "context"},
            exact_visible_text_candidates=[],
            consequential_changes=[],
            confidence=0.9,
            uncertainty=[],
            statements_not_inferred=["No spoken wording or person identity was inferred."],
        )


class InvalidResponseVisionProvider(FixtureVisionProvider):
    """Provider double that exercises the conservative semantic fallback path."""

    def annotate(self, packet: VisionPacket, *, project_root: Path) -> VisionAnnotation:
        self.calls.append(packet.candidate_id)
        raise ValidationFailure("fixture response omitted a valid focus-frame citation")


class ProviderHealthFailureVisionProvider(FixtureVisionProvider):
    """Provider double for shared server-health circuit behavior."""

    def annotate(self, packet: VisionPacket, *, project_root: Path) -> VisionAnnotation:
        self.calls.append(packet.candidate_id)
        raise ValidationFailure(
            "Local llama.cpp vision request failed: HTTP Error 503: Service Unavailable"
        )


class PacketLocalFailureSequenceVisionProvider(FixtureVisionProvider):
    """Two packet-local failures followed by a healthy observation."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def annotate(self, packet: VisionPacket, *, project_root: Path) -> VisionAnnotation:
        self.attempts += 1
        if self.attempts == 1:
            self.calls.append(packet.candidate_id)
            raise ValidationFailure(
                "Local llama.cpp vision request failed: HTTP Error 400: Bad Request"
            )
        if self.attempts == 2:
            self.calls.append(packet.candidate_id)
            raise ValidationFailure(
                "Annotation must cite at least one focus/action/result frame from its packet"
            )
        return super().annotate(packet, project_root=project_root)


class Transient400VisionProvider(FixtureVisionProvider):
    """Provider double that exposes the explicit HTTP-400 retry continuation."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_once = True

    def annotate(self, packet: VisionPacket, *, project_root: Path) -> VisionAnnotation:
        if self.fail_once:
            self.fail_once = False
            self.calls.append(packet.candidate_id)
            raise ValidationFailure(
                "Local llama.cpp vision request failed: HTTP Error 400: Bad Request"
            )
        return super().annotate(packet, project_root=project_root)


def _generate(root: Path) -> Path:
    subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts" / "generate_fixtures.py"), str(root)],
        check=True,
    )
    return root


def test_transcript_only_preserves_untimed_text(tmp_path: Path) -> None:
    transcript = tmp_path / "evidence.txt"
    transcript.write_text("First exact line.\nSecond exact line with 42.\n", encoding="utf-8")
    result = run_pipeline(transcript, output_root=tmp_path / "out")
    assert result.status == "automatically_checked"
    assert result.exit_code == 0
    text = result.markdown_path.read_text(encoding="utf-8")
    assert text.index("First exact line.") < text.index("Second exact line with 42.")
    assert text.count(".md") == 0
    assert len(list(result.project_dir.rglob("*.md"))) == 1
    assert not list(result.project_dir.rglob("*.html"))


def test_pipeline_applies_explicit_spoken_name_only_to_source_segment(tmp_path: Path) -> None:
    transcript = tmp_path / "named-speaker.srt"
    transcript.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n"
        "Hi, I'm Coach Princess, isang coach ng Freight Course 101.\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\n"
        "I am a coach and I am from Manila.\n",
        encoding="utf-8",
    )

    result = run_pipeline(transcript, output_root=tmp_path / "out", vision_mode="none")
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    segments = canonical["transcript_segments"]
    assert segments[0]["speaker_label"] == "Coach Princess"
    assert segments[1].get("speaker_label") is None
    assert "Explicit spoken self-identification evidence" in canonical[
        "transcript_source_decision"
    ]


def test_non_video_run_does_not_resolve_optional_visual_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import video_script_reconstructor.model_store as model_store_module
    import video_script_reconstructor.pipeline as pipeline_module

    def unexpected_ocr_resolution() -> object:
        pytest.fail("transcript-only runs must not resolve the optional OCR adapter")

    monkeypatch.setattr(pipeline_module, "_auto_ocr_adapter", unexpected_ocr_resolution)
    monkeypatch.setattr(
        model_store_module,
        "model_directory",
        lambda *_args, **_kwargs: pytest.fail(
            "transcript-only runs must not inspect optional model manifests"
        ),
    )
    transcript = tmp_path / "no-visual-capabilities.txt"
    transcript.write_text("Exact transcript-only wording.\n", encoding="utf-8")

    result = run_pipeline(transcript, output_root=tmp_path / "out", vision_mode="auto")

    assert result.status == "automatically_checked"
    assert result.exit_code == 0


def test_auto_video_vision_mode_stays_on_host_agent_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compatibility spelling must never construct the legacy local VLM."""

    fixtures = _generate(tmp_path / "fixtures")
    import video_script_reconstructor.providers.llama_cpp as llama_module

    def unexpected_local_provider(*_args: object, **_kwargs: object) -> object:
        pytest.fail("vision_mode=auto must not construct the local Qwen provider")

    monkeypatch.setattr(llama_module, "LlamaCppVisionProvider", unexpected_local_provider)
    result = run_pipeline(
        fixtures / "screen-tutorial.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "screen-tutorial.srt"],
        vision_mode="auto",
    )

    assert result.validation is not None and result.validation.valid
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    usage = next(
        item
        for item in reversed(canonical["manifest"]["provider_usage"])
        if item.get("purpose") == "visual"
    )
    assert usage["route"] == "host_agent"
    assert usage["provider"] == "codex-subagent"
    assert canonical["manifest"]["model_versions"]["vision_mode"] == "host-agent"
    assert all(
        item.get("provider") != "llama.cpp-local"
        for item in canonical["manifest"]["provider_usage"]
    )


def test_host_agent_review_budget_is_explicitly_configurable(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "screen-tutorial.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "screen-tutorial.srt"],
        vision_mode="host-agent",
        semantic_max_packets=240,
    )
    assert result.validation is not None and result.validation.valid
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    usage = next(
        item
        for item in reversed(canonical["manifest"]["provider_usage"])
        if item.get("purpose") == "visual"
    )
    assert usage["review_max_packets"] == 240


def test_host_agent_review_budget_defaults_to_long_form_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normal one-prompt run must not silently stop at the old probe cap."""

    monkeypatch.delenv("VSR_HOST_REVIEW_MAX_PACKETS", raising=False)
    monkeypatch.delenv("VSR_SEMANTIC_MAX_PACKETS", raising=False)
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "screen-tutorial.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "screen-tutorial.srt"],
        vision_mode="host-agent",
    )
    assert result.validation is not None and result.validation.valid
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    usage = next(
        item
        for item in reversed(canonical["manifest"]["provider_usage"])
        if item.get("purpose") == "visual"
    )
    assert usage["review_max_packets"] == 4096


def test_text_review_updates_human_verified_state_and_rerenders_atomically(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "wording.txt"
    transcript.write_text("Original uncertain wording.\n", encoding="utf-8")
    result = run_pipeline(transcript, output_root=tmp_path / "out")
    canonical_path = result.project_dir / ".state" / "canonical-project.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["review_items"].append(
        {
            "review_id": "R000001",
            "severity": "high",
            "category": "wording",
            "start_ms": None,
            "end_ms": None,
            "block_ids": ["B000001"],
            "segment_ids": ["T000001"],
            "event_ids": [],
            "frame_ids": [],
            "ocr_observation_ids": [],
            "image_claim_ids": [],
            "metadata_revision_ids": [],
            "sufficiency_decision_ids": [],
            "problem": "The wording requires attributable human correction.",
            "alternatives": ["Corrected exact wording."],
            "required_action": "Compare the preserved evidence and correct the segment.",
            "blocking": True,
            "decision": None,
            "replacement": None,
            "reviewer": None,
            "decision_timestamp_utc": None,
            "rationale": None,
        }
    )
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    correction = apply_review(
        result.project_dir,
        "R000001",
        reviewer="Ada Reviewer",
        decision="correct",
        replacement="Corrected exact wording.",
        rationale="Compared the preserved source and recorded the exact human decision.",
    )
    updated = json.loads(canonical_path.read_text(encoding="utf-8"))
    segment = updated["transcript_segments"][0]
    assert segment["raw_text"] == "Original uncertain wording."
    assert segment["human_verified_text"] == "Corrected exact wording."
    assert correction["new_value"]["replacement"] == "Corrected exact wording."
    assert updated["project_status"] == "human_reviewed"
    assert updated["audit"]["final_project_status"] == "human_reviewed"
    assert result.markdown_path.read_text(encoding="utf-8").count("Corrected exact wording.") == 1
    assert validate_project(result.project_dir).valid


def test_generated_video_has_measured_frames_and_embedded_metadata(tmp_path: Path) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
    )
    assert result.status == "review_required"
    assert result.exit_code == 3
    assert result.validation is not None and result.validation.valid
    cached_validation = validate_project(result.project_dir, use_cached_file_hash=True)
    assert cached_validation.valid
    assert cached_validation.checks["metadata_integrity_mode"] == "canonical-file-hash"
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    all_frames = canonical["frames"]
    assert 1 <= cached_validation.checks["metadata_workers"] <= len(all_frames)
    frames = [frame for frame in all_frames if not frame.get("parent_full_frame_id")]
    assert len(frames) >= 2
    assert canonical["sufficiency_decisions"]
    timeline_path = result.project_dir / ".state" / "timeline" / "timeline.json"
    image_observations_path = result.project_dir / ".state" / "vision" / "image-observations.json"
    assert timeline_path.is_file()
    assert image_observations_path.is_file()
    # Large machine-state ledgers are compact UTF-8 JSON; Markdown remains the
    # human-facing readable artifact. This avoids repeated whitespace writes on
    # long-form projects without changing parsed state.
    assert timeline_path.read_text(encoding="utf-8").count("\n") == 1
    assert image_observations_path.read_text(encoding="utf-8").count("\n") == 1
    assert all(frame["timestamp_source"] == "ffmpeg-showinfo" for frame in frames)
    assert all(frame["offset_ms"] == frame["actual_ms"] - frame["requested_ms"] for frame in frames)
    metadata_results = verify_image_metadata(result.project_dir)
    assert len(metadata_results) == len(all_frames)
    assert all(item["verified"] for item in metadata_results)
    pixel_values = []
    for frame in frames:
        path = result.project_dir / Path(frame["full_frame_path"])
        with Image.open(path) as image:
            image.load()
            pixel_values.append(image.convert("RGB").getpixel((100, 250)))
        metadata = read_embedded_metadata(path)
        assert metadata.image.image_id == frame["frame_id"]
        assert metadata.image.actual_ms == frame["actual_ms"]
        assert metadata.links.chapter_ids
        assert metadata.links.block_ids
        assert metadata.links.segment_ids
        assert metadata.links.visual_event_ids
        assert metadata.analysis.sufficiency.evaluated_question_ids
    assert len(set(pixel_values)) >= 2
    text = result.markdown_path.read_text(encoding="utf-8")
    spoken_sections = [
        section.split("**Visual**", 1)[0] for section in text.split("**Spoken**")[1:]
    ]
    assert sum(section.count("The exact value is 42.") for section in spoken_sections) == 1
    assert text.index("The exact value is 42.") < text.index("Now the slide changes.")
    assert "[visual evidence retained; semantic description pending review]" in text
    assert len(list(result.project_dir.rglob("*.md"))) == 1
    assert not list(result.project_dir.rglob("*.html"))


def test_normal_run_uses_final_validation_as_the_post_render_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    import video_script_reconstructor.pipeline as pipeline_module

    original = pipeline_module.validate_project
    calls: list[tuple[bool, bool]] = []

    def counted_validate(project_dir: Path, **kwargs: object):
        calls.append(
            (
                bool(kwargs.get("use_cached_file_hash")),
                bool(kwargs.get("verify_metadata", True)),
            )
        )
        return original(project_dir, **kwargs)

    monkeypatch.setattr(pipeline_module, "validate_project", counted_validate)
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
    )

    assert result.validation is not None and result.validation.valid
    assert result.validation.checks["metadata_verified"] is True
    assert calls == [(True, False), (True, True)]


def test_healthy_run_defers_resource_snapshot_to_final_manifest_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    import video_script_reconstructor.pipeline as pipeline_module

    original = pipeline_module.atomic_write_json
    original_update = pipeline_module.atomic_update_json_fields
    canonical_writes: list[Path] = []

    def counted_write(path: Path, payload: object, **kwargs: object) -> None:
        if path.name == "canonical-project.json":
            canonical_writes.append(path)
        original(path, payload, **kwargs)

    def counted_update(path: Path, updates: object, **kwargs: object) -> None:
        if path.name == "canonical-project.json":
            canonical_writes.append(path)
        original_update(path, updates, **kwargs)

    monkeypatch.setattr(pipeline_module, "atomic_write_json", counted_write)
    monkeypatch.setattr(pipeline_module, "atomic_update_json_fields", counted_update)
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )

    assert result.validation is not None and result.validation.valid
    assert len(canonical_writes) >= 3
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    usage = canonical["manifest"]["performance"]["resource_usage"]
    assert usage
    assert isinstance(usage["output"]["reclaimable_bytes"], int)
    assert isinstance(usage["output"]["reclaimable_file_count"], int)
    # The byte/file telemetry is a post-write fixed point, not a stale
    # pre-manifest snapshot whose own JSON serialization was omitted.
    assert usage["output"] == resource_snapshot(result.project_dir)["output"]


def test_local_vision_mode_ingests_schema_valid_semantic_observations(tmp_path: Path) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    provider = FixtureVisionProvider()
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="local",
        vision_provider=provider,
    )
    assert result.validation is not None and result.validation.valid
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    assert provider.calls
    assert len(canonical["visual_observations"]) == len(provider.calls)
    assert all(
        item["actor_kind"] == "multimodal_model" for item in canonical["visual_observations"]
    )
    assert all(item["status"] == "supported" for item in canonical["image_claims"])
    claim_ids = [item["claim_id"] for item in canonical["image_claims"]]
    # Distinct packet scopes may cite the same observed focus frame; semantic
    # observations remain per-event while identical supported claims are
    # canonicalized once.
    assert len(claim_ids) == len(set(claim_ids)) <= len(provider.calls)
    assert any(
        item["analysis"]["semantic_status"] == "observed"
        for item in canonical["evidence_image_metadata"]
    )
    assert "A colored slide is directly visible" in result.markdown_path.read_text(encoding="utf-8")


def test_invalid_semantic_responses_become_review_only_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    monkeypatch.setenv("VSR_SEMANTIC_FAILURE_LIMIT", "2")
    provider = InvalidResponseVisionProvider()
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="local",
        vision_provider=provider,
    )
    assert result.validation is not None and result.validation.valid
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    usage = [
        item
        for item in canonical["manifest"]["provider_usage"]
        if item.get("provider") == provider.descriptor.provider_id
    ][-1]
    # Packet-local schema failures remain conservative fallbacks, but they do
    # not open a global circuit: a later packet still deserves its own model
    # attempt because the loopback provider may be healthy for that scope.
    assert usage["semantic_provider_attempt_failure_count"] == len(provider.calls)
    assert usage["semantic_fallback_annotation_count"] == len(
        canonical["visual_observations"]
    )
    assert usage["semantic_circuit_breaker_triggered"] is False
    assert canonical["project_status"] == "review_required"
    assert "blocked_prerequisite" not in canonical["audit"]["blocking_failures"]
    assert all(
        event["review_status"] == "review_required"
        for event in canonical["visual_events"]
    )
    assert all(
        not item["supported_claim_ids"]
        for item in canonical["frames"]
        if item["metadata"]["analysis"]["semantic_status"] == "observed"
    )
    assert all(
        item["event_type"] == "semantic_pending"
        for item in canonical["visual_events"]
    )


def test_provider_health_failures_open_circuit_after_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    monkeypatch.setenv("VSR_SEMANTIC_FAILURE_LIMIT", "2")
    provider = ProviderHealthFailureVisionProvider()
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="local",
        vision_provider=provider,
    )
    assert result.validation is not None and result.validation.valid
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    usage = [
        item
        for item in canonical["manifest"]["provider_usage"]
        if item.get("provider") == provider.descriptor.provider_id
    ][-1]
    assert usage["semantic_provider_attempt_failure_count"] == 2
    assert usage["semantic_circuit_breaker_triggered"] is True
    assert usage["semantic_fallback_annotation_count"] == len(
        canonical["visual_observations"]
    )


def test_packet_local_failures_do_not_suppress_later_provider_attempts(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    provider = PacketLocalFailureSequenceVisionProvider()
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="local",
        vision_provider=provider,
    )
    assert result.validation is not None and result.validation.valid
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    usage = [
        item
        for item in canonical["manifest"]["provider_usage"]
        if item.get("provider") == provider.descriptor.provider_id
    ][-1]
    assert len(provider.calls) == 3
    assert usage["semantic_provider_attempt_failure_count"] == 2
    assert usage["semantic_circuit_breaker_triggered"] is False
    assert usage["semantic_fallback_annotation_count"] == 2
    assert any(event["annotation_provider"] == provider.descriptor.provider_id for event in canonical["visual_events"])


def test_parallel_packet_local_failures_do_not_report_provider_circuit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    monkeypatch.setenv("VSR_SEMANTIC_FAILURE_LIMIT", "2")
    provider = PacketLocalFailureSequenceVisionProvider()
    summary = apply_vision_provider(
        result.project_dir,
        provider,
        semantic_max_packets=2,
        semantic_workers=2,
    )
    assert len(provider.calls) == 2
    assert summary["semantic_provider_attempt_failure_count"] == 2
    assert summary["semantic_circuit_breaker_triggered"] is False


def test_semantic_packet_budget_preserves_temporal_coverage_and_defers_expensive_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    monkeypatch.setenv("VSR_SEMANTIC_MAX_PACKETS", "2")
    summary = apply_vision_provider(result.project_dir, FixtureVisionProvider())
    assert len(summary["applied"]) == 2
    assert summary["semantic_deferred_event_ids"]
    assert summary["semantic_deferred_event_ids"] == sorted(
        summary["semantic_deferred_event_ids"],
        key=lambda value: int(str(value).removeprefix("V")),
    )
    assert validate_project(result.project_dir).valid
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    assert canonical["project_status"] == "review_required"
    assert "blocked_prerequisite" not in canonical["audit"]["blocking_failures"]


def test_semantic_budget_does_not_reapply_observed_focus_packets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    monkeypatch.setenv("VSR_SEMANTIC_MAX_PACKETS", "1")
    provider = FixtureVisionProvider()
    first = apply_vision_provider(result.project_dir, provider)
    pending_after_first = pending_packet_count(result.project_dir)
    second = apply_vision_provider(result.project_dir, provider)
    assert len(first["applied"]) == 1
    assert len(second["applied"]) == 1
    assert len(provider.calls) == 2
    assert pending_after_first >= 0


def test_semantic_retry_fallbacks_selects_only_transient_http_400_packets(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    provider = Transient400VisionProvider()
    first = apply_vision_provider(result.project_dir, provider, semantic_max_packets=1)
    assert len(first["applied"]) == 1
    assert "HTTP Error 400" in first["semantic_provider_failures"][0]["error"]

    second = apply_vision_provider(
        result.project_dir,
        provider,
        semantic_max_packets=1,
        retry_fallbacks=True,
    )
    assert len(second["applied"]) == 1
    assert second["semantic_retry_fallbacks"] is True
    assert len(provider.calls) == 2


def test_semantic_batch_retry_fallbacks_does_not_skip_observed_fallback_event(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    output_root = tmp_path / "out"
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=output_root,
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    provider = Transient400VisionProvider()
    first = apply_vision_provider(result.project_dir, provider, semantic_max_packets=1)
    assert len(first["applied"]) == 1
    assert pending_packet_count(result.project_dir) == 2

    # Consume the remaining ordinary pending packets so only the observed
    # HTTP-400 fallback remains. Batch preflight must still process it when
    # retry_fallbacks=True instead of skipping on pending_packet_count==0.
    second = apply_vision_provider(result.project_dir, provider, semantic_max_packets=10)
    assert len(second["applied"]) == 2
    assert pending_packet_count(result.project_dir) == 0

    batch = run_semantic_batch(
        output_root,
        provider,
        semantic_max_packets=1,
        retry_fallbacks=True,
        min_free_bytes=0,
    )
    assert batch["processed_count"] == 1
    assert batch["skipped_count"] == 0
    record = batch["projects"][0]
    assert record["retry_fallback_before"] == 1
    assert len(record["summary"]["applied"]) == 1
    assert pending_packet_count(result.project_dir) == 0


def test_semantic_only_pass_updates_manifest_without_rerunning_media_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    provider = FixtureVisionProvider()
    original_validate = semantic_module.validate_project
    validation_calls: list[dict[str, object]] = []

    def counted_validate(project_dir: Path, **kwargs: object):
        validation_calls.append(dict(kwargs))
        return original_validate(project_dir, **kwargs)

    monkeypatch.setattr(semantic_module, "validate_project", counted_validate)
    semantic = run_semantic_pass(result.project_dir, provider, semantic_max_packets=2)
    assert semantic["status"] == "review_required"
    assert len(semantic["summary"]["applied"]) == 2
    assert semantic["validation_errors"] == []
    # Finalization proves the post-render project before run-semantic-pass adds
    # manifest-only telemetry.  The proof is reused for the receipt instead of
    # traversing and validating the whole evidence tree a second time.
    assert validation_calls == [{"use_cached_file_hash": True}]
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    assert any(
        item.get("provider") == provider.descriptor.provider_id
        and item.get("semantic_max_packets") == 2
        and float(item.get("semantic_elapsed_seconds", -1)) >= 0
        and float(item.get("semantic_observations_per_second", -1)) >= 0
        for item in canonical["manifest"]["provider_usage"]
    )
    assert (result.project_dir / ".state" / "validation-receipt.json").is_file()


def test_parallel_semantic_workers_commit_deterministically(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    provider = FixtureVisionProvider()
    summary = apply_vision_provider(
        result.project_dir,
        provider,
        semantic_max_packets=2,
        semantic_workers=2,
    )
    assert summary["semantic_worker_count"] == 2
    assert len(summary["applied"]) == 2
    assert summary["semantic_provider_attempt_failure_count"] == 0
    assert validate_project(result.project_dir).valid
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    observations = canonical["visual_observations"]
    committed = [item for item in observations if item["provider"] == provider.descriptor.provider_id]
    assert [item["observation_id"] for item in committed] == ["VA000001", "VA000002"]


def test_semantic_batch_reuses_one_provider_across_canonical_projects(tmp_path: Path) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    output_root = tmp_path / "collection"
    first = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=output_root / "first",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    second = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=output_root / "second",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    provider = FixtureVisionProvider()

    batch = run_semantic_batch(
        output_root,
        provider,
        semantic_max_packets=1,
        min_free_bytes=0,
    )

    assert batch["status"] == "review_required"
    assert batch["processed_count"] == 2
    assert batch["skipped_count"] == 0
    assert len(batch["projects"]) == 2
    assert all(item["pending_after"] < item["pending_before"] for item in batch["projects"])
    assert len(provider.calls) == 2
    assert first.project_dir != second.project_dir


def test_deterministic_local_semantic_cache_reuses_annotations_across_projects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    monkeypatch.setenv("VSR_SEMANTIC_SHARED_CACHE_DIR", str(tmp_path / "semantic-cache"))

    first_provider = FixtureVisionProvider()
    first_provider.semantic_cacheable = True
    first = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "first-out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="local",
        vision_provider=first_provider,
    )
    assert first.validation is not None and first.validation.valid
    assert first_provider.calls

    second_provider = FixtureVisionProvider()
    second_provider.semantic_cacheable = True
    second = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "second-out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="local",
        vision_provider=second_provider,
    )
    assert second.validation is not None and second.validation.valid
    assert second_provider.calls == []
    canonical = json.loads(
        (second.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    usage = [
        item
        for item in canonical["manifest"]["provider_usage"]
        if item.get("provider") == "fixture-local-vision"
    ][-1]
    assert usage["semantic_cache_enabled"] is True
    assert usage["semantic_cache_hit_count"] == len(first_provider.calls)
    assert usage["semantic_cache_miss_count"] == 0
    # The exact packet cache is preferred when candidate/frame IDs and the
    # full packet match across generated projects; content-remap is the
    # fallback for repeated pixels with different packet identities.
    assert (
        usage["semantic_cache_hit_count"] + usage["semantic_content_cache_hit_count"]
        >= len(first_provider.calls)
    )


def test_semantic_batch_materializes_one_canonical_write_per_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staged semantic links materialize the large canonical root once."""

    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    provider = FixtureVisionProvider()
    canonical_writes = 0
    project_loads = 0
    evidence_write = evidence_module.atomic_write_json
    evidence_update = evidence_module.atomic_update_json_fields
    semantic_write = semantic_module.atomic_write_json
    semantic_update = semantic_module.atomic_update_json_fields
    original_load_project = semantic_module._load_project

    def count_evidence_write(path: Path, payload: object, **kwargs: object) -> None:
        nonlocal canonical_writes
        if path.name == "canonical-project.json":
            canonical_writes += 1
        evidence_write(path, payload, **kwargs)

    def count_semantic_write(path: Path, payload: object, **kwargs: object) -> None:
        nonlocal canonical_writes
        if path.name == "canonical-project.json":
            canonical_writes += 1
        semantic_write(path, payload, **kwargs)

    def count_evidence_update(path: Path, updates: object, **kwargs: object) -> None:
        nonlocal canonical_writes
        if path.name == "canonical-project.json":
            canonical_writes += 1
        evidence_update(path, updates, **kwargs)

    def count_semantic_update(path: Path, updates: object, **kwargs: object) -> None:
        nonlocal canonical_writes
        if path.name == "canonical-project.json":
            canonical_writes += 1
        semantic_update(path, updates, **kwargs)

    def count_project_load(path: Path) -> dict[str, object]:
        nonlocal project_loads
        project_loads += 1
        return original_load_project(path)

    monkeypatch.setattr(evidence_module, "atomic_write_json", count_evidence_write)
    monkeypatch.setattr(evidence_module, "atomic_update_json_fields", count_evidence_update)
    monkeypatch.setattr(semantic_module, "atomic_write_json", count_semantic_write)
    monkeypatch.setattr(semantic_module, "atomic_update_json_fields", count_semantic_update)
    monkeypatch.setattr(semantic_module, "_load_project", count_project_load)
    summary = apply_vision_provider(result.project_dir, provider)

    assert summary["applied"]
    # Per-observation deltas are journaled; the large canonical root is
    # materialized once after the batch and then validated/rendered.
    assert canonical_writes == 1
    assert project_loads == 1
    assert pending_packet_count(result.project_dir) == 0
    assert validate_project(result.project_dir).valid


def test_semantic_batch_journal_recovers_committed_observations_after_failure(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        vision_mode="none",
    )
    failing_provider = FixtureVisionProvider()
    original_annotate = failing_provider.annotate

    def fail_on_second(packet: VisionPacket, *, project_root: Path) -> VisionAnnotation:
        if len(failing_provider.calls) >= 1:
            raise RuntimeError("simulated semantic worker interruption")
        return original_annotate(packet, project_root=project_root)

    failing_provider.annotate = fail_on_second  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated semantic worker interruption"):
        apply_vision_provider(result.project_dir, failing_provider)

    interrupted = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    observed_after_interrupt = [
        item
        for item in interrupted["evidence_image_metadata"]
        if item["analysis"]["semantic_status"] == "observed"
    ]
    # The first observation is durable in the journal, but is intentionally not
    # visible in canonical JSON until restart recovery folds the staged delta.
    assert observed_after_interrupt == []
    assert (
        result.project_dir / ".state" / "checkpoints" / "semantic-batch-journal.jsonl"
    ).is_file()

    resumed = apply_vision_provider(result.project_dir, FixtureVisionProvider())
    # Recovery folds the staged observation before packet selection. Depending
    # on the scheduler's event ordering, the remaining distinct scopes may be
    # submitted together, but the resumed pass must make forward progress.
    assert resumed["applied"]
    assert validate_project(result.project_dir).valid
    final = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    assert all(
        item["analysis"]["semantic_status"] == "observed"
        for item in final["evidence_image_metadata"]
        if item["image"]["image_id"] in {entry["image_id"] for entry in resumed["applied"]}
    )


def test_pipeline_does_not_restart_semantic_provider_when_packets_are_observed(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    first_provider = FixtureVisionProvider()
    first = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        language=None,
        vision_mode="local",
        vision_provider=first_provider,
    )
    assert first.validation is not None and first.validation.valid
    assert first_provider.calls

    # The language hint changes the run key, but the source/config/vision
    # contract still permits visual-state reuse. All semantic packets are
    # already observed, so a retry must not call the provider again.
    retry_provider = FixtureVisionProvider()
    retried = run_pipeline(
        fixtures / "slide-lecture.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "slide-lecture.srt"],
        language="en",
        vision_mode="local",
        vision_provider=retry_provider,
    )
    assert retried.validation is not None and retried.validation.valid
    assert retry_provider.calls == []


def test_public_video_pipeline_preserves_raw_and_interpreted_ocr_changes(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "screen-tutorial.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "screen-tutorial.srt"],
        ocr_adapter=PixelStateOCRAdapter(),
    )
    assert result.validation is not None and result.validation.valid
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    observations = canonical["ocr_observations"]
    states = {item["normalized_interpretation"] for item in observations}
    assert {"DISABLED", "ENABLED"}.issubset(states)
    assert all(
        item["raw_engine_text"].strip() == item["normalized_interpretation"]
        for item in observations
    )
    full_frames = [frame for frame in canonical["frames"] if not frame.get("parent_full_frame_id")]
    crop_frames = [frame for frame in canonical["frames"] if frame.get("parent_full_frame_id")]
    assert len(full_frames) >= 2
    assert crop_frames
    assert all(frame["quality_scores"] for frame in full_frames)
    assert all(frame["perceptual_hashes"] for frame in full_frames)
    assert all(
        frame["perceptual_hashes"]["dhash-8"]
        == perceptual_dhash(result.project_dir / frame["full_frame_path"])
        for frame in full_frames
    )
    assert all(
        frame["perceptual_hashes"]["dhash-8-algorithm"] == "dhash-8-v1"
        and frame["perceptual_hashes"]["dhash-8-verified"] == "true"
        for frame in full_frames
    )
    assert all(frame["ocr_observation_ids"] for frame in full_frames)
    for frame in full_frames:
        embedded = read_embedded_metadata(result.project_dir / frame["full_frame_path"])
        assert embedded.analysis.enrichment_level == "deterministic"
        assert embedded.analysis.ocr_observation_ids == frame["ocr_observation_ids"]
        assert embedded.links.neighbor_image_ids == embedded.analysis.neighbor_image_ids
    blocks = {block["block_id"]: block for block in canonical["script_blocks"]}
    for crop in crop_frames:
        assert re.fullmatch(
            r"evidence/crops/F\d{6}-C\d{2}__\d{2}h\d{2}m\d{2}s\d{3}__detail\.png",
            crop["path"],
        )
        embedded = read_embedded_metadata(result.project_dir / crop["path"])
        transformation_ids = embedded.image.derivation.transformation_ids
        assert len(transformation_ids) == 1
        assert re.fullmatch(r"X\d{6}", transformation_ids[0])
        assert all(
            transformation_ids[0] in blocks[block_id]["transformation_ids"]
            for block_id in crop["linked_block_ids"]
        )
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "DISABLED" in markdown
    assert "ENABLED" in markdown


def test_resume_reuses_identical_artifacts_and_sidecar_change_invalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    source = fixtures / "talking-head.mp4"
    sidecar = fixtures / "talking-head.srt"
    first = run_pipeline(source, output_root=tmp_path / "out", subtitles=[sidecar])
    first_manifest = json.loads(
        (first.project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
    )
    image_hashes = {
        path.name: path.read_bytes()
        for path in (first.project_dir / "evidence" / "full").glob("*.png")
    }
    import video_script_reconstructor.pipeline as pipeline_module

    def unexpected_visual_rebuild(*_args: object, **_kwargs: object) -> object:
        pytest.fail("ASR chunk geometry changes must reuse compatible source-pixel visual state")

    monkeypatch.setattr(pipeline_module, "_extract_visual_evidence", unexpected_visual_rebuild)
    geometry_changed = run_pipeline(
        source,
        output_root=tmp_path / "out",
        subtitles=[sidecar],
        asr_chunk_seconds=150,
        asr_overlap_seconds=15,
    )
    geometry_manifest = json.loads(
        (geometry_changed.project_dir / ".state" / "run-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert geometry_manifest["run_cache_key"] != first_manifest["run_cache_key"]
    assert (
        geometry_manifest["stages"]["visual_evidence"]["detail"]
        == "Reused source-pixel visual state; transcript-only invalidation."
    )
    assert image_hashes == {
        path.name: path.read_bytes()
        for path in (geometry_changed.project_dir / "evidence" / "full").glob("*.png")
    }
    resumed = run_pipeline(source, output_root=tmp_path / "out", subtitles=[sidecar], resume=True)
    second_manifest = json.loads(
        (resumed.project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert second_manifest["run_cache_key"] == first_manifest["run_cache_key"]
    assert image_hashes == {
        path.name: path.read_bytes()
        for path in (resumed.project_dir / "evidence" / "full").glob("*.png")
    }
    sidecar.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nChanged evidence.\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nNothing is summarized.\n",
        encoding="utf-8",
    )
    changed_plan_key = None
    try:
        changed = run_pipeline(source, output_root=tmp_path / "out", subtitles=[sidecar])
        changed_plan_key = json.loads(
            (changed.project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
        )["run_cache_key"]
        assert changed_plan_key != first_manifest["run_cache_key"]
        assert image_hashes == {
            path.name: path.read_bytes()
            for path in (changed.project_dir / "evidence" / "full").glob("*.png")
        }
        changed_manifest = json.loads(
            (changed.project_dir / ".state" / "run-manifest.json").read_text(encoding="utf-8")
        )
        assert (
            changed_manifest["stages"]["visual_evidence"]["detail"]
            == "Reused source-pixel visual state; transcript-only invalidation."
        )
    finally:
        assert changed_plan_key is not None
    assert changed_plan_key != first_manifest["run_cache_key"]


def test_cache_hit_uses_validation_receipt_and_invalidates_after_artifact_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    source = fixtures / "talking-head.mp4"
    sidecar = fixtures / "talking-head.srt"
    first = run_pipeline(source, output_root=tmp_path / "out", subtitles=[sidecar])
    receipt = first.project_dir / ".state" / "validation-receipt.json"
    assert receipt.is_file()
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["project_status"] == first.status
    assert receipt_payload["cache_contract_complete"] is True
    assert len(receipt_payload["canonical_file_signature"]) == 4

    import video_script_reconstructor.pipeline as pipeline_module

    original_validate = pipeline_module.validate_project
    calls: list[Path] = []

    def counted_validate(project_dir: Path, **kwargs: object):
        calls.append(project_dir)
        return original_validate(project_dir, **kwargs)

    monkeypatch.setattr(pipeline_module, "validate_project", counted_validate)
    resumed = run_pipeline(source, output_root=tmp_path / "out", subtitles=[sidecar])
    assert resumed.validation is not None and resumed.validation.valid
    assert resumed.validation.errors == (first.validation.errors if first.validation else [])
    assert resumed.validation.warnings == (first.validation.warnings if first.validation else [])
    assert calls == []

    markdown = resumed.markdown_path
    markdown.write_text(markdown.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rerun = run_pipeline(source, output_root=tmp_path / "out", subtitles=[sidecar])
    assert rerun.validation is not None and rerun.validation.valid
    assert calls


def test_host_agent_bundle_hash_matches_final_canonical_project(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "screen-tutorial.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "screen-tutorial.srt"],
        vision_mode="host-agent",
    )
    assert result.validation is not None and result.validation.valid
    canonical_path = result.project_dir / ".state" / "canonical-project.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    usage = canonical["manifest"]["provider_usage"][-1]
    bundle_dir = Path(usage["review_bundle_dir"])
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["canonical_project_sha256"] == sha256_file(canonical_path)
    assert len(bundle["requests"]) == usage["review_request_count"]
    assert bundle_dir.is_dir()


def test_host_agent_no_work_resume_does_not_publish_empty_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed host route must not accumulate an empty review bundle."""

    fixtures = _generate(tmp_path / "fixtures")
    monkeypatch.setattr(semantic_module, "pending_packet_count", lambda _project: 0)

    def fail_empty_bundle(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("no-work host resume must not create a bundle")

    monkeypatch.setattr(subagent_module, "create_review_bundle", fail_empty_bundle)
    result = run_pipeline(
        fixtures / "screen-tutorial.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "screen-tutorial.srt"],
        vision_mode="host-agent",
    )
    assert result.validation is not None and result.validation.valid
    canonical = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(
            encoding="utf-8"
        )
    )
    usage = canonical["manifest"]["provider_usage"][-1]
    assert usage["review_bundle_dir"] is None
    assert usage["review_request_count"] == 0


def test_host_agent_bundle_apply_rebinds_validation_receipt(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "screen-tutorial.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "screen-tutorial.srt"],
        vision_mode="host-agent",
    )
    canonical_path = result.project_dir / ".state" / "canonical-project.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    usage = next(
        item
        for item in reversed(canonical["manifest"]["provider_usage"])
        if item.get("review_bundle_dir")
    )
    bundle_dir = Path(usage["review_bundle_dir"])
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    for entry in bundle["requests"]:
        packet = VisionPacket.model_validate(
            json.loads((result.project_dir / entry["packet_path"]).read_text(encoding="utf-8"))
        )
        focus = next(frame for frame in packet.frames if frame.role == "focus")
        annotation = VisionAnnotation(
            candidate_id=packet.candidate_id,
            factual_visible_description="No defensible visible fact was established.",
            event_type="semantic_pending",
            evidence_frame_ids=[focus.frame_id],
            before_action_after_roles={focus.frame_id: "context"},
            exact_visible_text_candidates=[],
            consequential_changes=[],
            confidence=0.0,
            uncertainty=["The available stills do not support a defensible semantic claim."],
            statements_not_inferred=["No identity, speech, motion, intent, or hidden state is inferred."],
        )
        (bundle_dir / entry["response_path"]).write_text(
            annotation.model_dump_json(), encoding="utf-8"
        )
    applied = apply_review_bundle(result.project_dir, bundle_dir)
    assert applied["validation_errors"] == []
    refreshed = json.loads(canonical_path.read_text(encoding="utf-8"))
    assert read_trusted_validation_receipt(
        result.project_dir,
        None,
        run_cache_key=refreshed["manifest"]["run_cache_key"],
    ) is not None


def test_append_only_enrichment_updates_markdown_and_preserves_stale_disagreement(
    tmp_path: Path,
) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "screen-tutorial.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "screen-tutorial.srt"],
    )
    canonical_path = result.project_dir / ".state" / "canonical-project.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    frame = canonical["frames"][0]
    image_id = frame["frame_id"]
    creation_revision = frame["latest_revision_id"]
    image_path = result.project_dir / Path(frame["full_frame_path"])
    before_pixels = read_embedded_metadata(image_path).image.pixel_hash.value

    common = {
        "image_ids": [image_id],
        "context_image_ids": [],
        "actor_kind": "host_agent",
        "actor_label": "integration-reviewer",
        "observed_at_utc": "2026-08-09T00:00:00Z",
        "purpose": "Resolve the meaningful visible state.",
        "targeted_question_ids": [],
        "analysis_depth": "cumulative",
        "prior_metadata_visible": True,
        "rationale": "Inspected the original-resolution full frame.",
        "validation_result": "accepted",
    }
    first_observation = {
        **common,
        "observation_id": "VA000001",
        "base_revision_id": creation_revision,
        "proposed_claims": [
            {
                "claim_class": "direct_visible",
                "statement": "A rectangular interface control is visible.",
                "importance": "supporting",
                "confidence": 0.95,
                "evidence_regions": [
                    {
                        "image_id": image_id,
                        "region_xywh_normalized": [0.09, 0.60, 0.33, 0.22],
                        "whole_frame_basis": False,
                    }
                ],
                "relationship": "new",
            }
        ],
        "new_supported_information": ["The control is directly visible."],
    }
    first_path = tmp_path / "observation-1.json"
    first_path.write_text(json.dumps(first_observation), encoding="utf-8")
    first = ingest_project_observation(
        result.project_dir, first_path, base_revision=creation_revision
    )
    assert first["pixel_invariance_verified"] is True
    first_revision = first["new_revision_id"]
    assert "A rectangular interface control is visible." in result.markdown_path.read_text(
        encoding="utf-8"
    )

    second_observation = {
        **common,
        "observation_id": "VA000002",
        "base_revision_id": first_revision,
        "purpose": "Read the visible state label precisely.",
        "proposed_claims": [
            {
                "claim_class": "exact_text",
                "statement": "The control is labeled DISABLED.",
                "importance": "consequential",
                "confidence": 0.98,
                "evidence_regions": [
                    {
                        "image_id": image_id,
                        "region_xywh_normalized": [0.09, 0.60, 0.33, 0.22],
                        "whole_frame_basis": False,
                    }
                ],
                "relationship": "new",
            }
        ],
        "new_supported_information": ["The DISABLED label is readable."],
    }
    second_path = tmp_path / "observation-2.json"
    second_path.write_text(json.dumps(second_observation), encoding="utf-8")
    second = ingest_project_observation(
        result.project_dir, second_path, base_revision=first_revision
    )
    assert second["new_revision_id"] != first_revision
    markdown_after_second = result.markdown_path.read_text(encoding="utf-8")
    assert "The control is labeled DISABLED." in markdown_after_second

    current = json.loads(canonical_path.read_text(encoding="utf-8"))
    first_claim = current["image_claims"][0]["claim_id"]
    third_observation = {
        **common,
        "observation_id": "VA000003",
        "base_revision_id": creation_revision,
        "purpose": "Challenge the earlier control interpretation from a stale base.",
        "prior_metadata_visible": False,
        "analysis_depth": "blind",
        "proposed_claims": [],
        "contradicted_claim_ids": [first_claim],
        "remaining_unknowns": ["The control boundary is ambiguous."],
    }
    third_path = tmp_path / "observation-3.json"
    third_path.write_text(json.dumps(third_observation), encoding="utf-8")
    third = ingest_project_observation(
        result.project_dir, third_path, base_revision=creation_revision
    )
    assert third["stale_base_reconciled"] is True
    assert first_claim in third["disputed_claim_ids"]
    final = json.loads(canonical_path.read_text(encoding="utf-8"))
    assert [item["observation_id"] for item in final["visual_observations"]] == [
        "VA000001",
        "VA000002",
        "VA000003",
    ]
    assert read_embedded_metadata(image_path).image.pixel_hash.value == before_pixels
    final_text = result.markdown_path.read_text(encoding="utf-8")
    assert "A rectangular interface control is visible." not in final_text
    assert "The control is labeled DISABLED." in final_text


def test_review_corrections_are_attributable_and_finalization_is_gated(tmp_path: Path) -> None:
    fixtures = _generate(tmp_path / "fixtures")
    result = run_pipeline(
        fixtures / "talking-head.mp4",
        output_root=tmp_path / "out",
        subtitles=[fixtures / "talking-head.srt"],
    )
    with pytest.raises(ValidationFailure):
        finalize_project(
            result.project_dir,
            reviewer="Ada Reviewer",
            rationale="Attempted before reviewing preserved visual uncertainty.",
        )
    project = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    review_ids = [item["review_id"] for item in project["review_items"]]
    assert review_ids
    for review_id in review_ids:
        correction = apply_review(
            result.project_dir,
            review_id,
            reviewer="Ada Reviewer",
            decision="accept_uncertainty",
            replacement=None,
            rationale="Inspected the cited full frame and accepted the explicit semantic limitation.",
        )
        assert correction["old_value"]["decision"] is None
        assert correction["new_value"]["reviewer"] == "Ada Reviewer"
    signoff = finalize_project(
        result.project_dir,
        reviewer="Ada Reviewer",
        rationale="All mandatory review items and evidence limitations were explicitly checked.",
    )
    assert signoff["reviewer"] == "Ada Reviewer"
    final = json.loads(
        (result.project_dir / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    assert final["project_status"] == "fully_verified"
    assert final["audit"]["final_project_status"] == "fully_verified"
    assert validate_project(result.project_dir).valid
