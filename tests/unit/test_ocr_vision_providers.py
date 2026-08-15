from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from video_script_reconstructor.errors import ReviewRequired, SecurityError, ValidationFailure
from video_script_reconstructor.ocr import (
    TesseractOCRAdapter,
    normalize_ocr_text,
    parse_tesseract_tsv,
)
from video_script_reconstructor.providers.base import ExternalProcessingPermission
from video_script_reconstructor.providers.external import ExternalVisionProvider
from video_script_reconstructor.providers.host_agent import HostAgentVisionProvider
from video_script_reconstructor.providers.local import LocalCommandVisionProvider
from video_script_reconstructor.vision_packets import (
    VisionAnnotation,
    create_vision_packet,
    validate_annotation_for_packet,
)
from video_script_reconstructor.visual_events import (
    annotation_to_visual_event,
    pending_visual_event,
)

TSV = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\t"
    "width\theight\tconf\ttext\n"
    "5\t1\t1\t1\t1\t1\t10\t20\t40\t10\t96.0\tValue:\n"
    "5\t1\t1\t1\t1\t2\t55\t20\t15\t10\t41.0\t4?\n"
)

QUOTED_TSV = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\t"
    "width\theight\tconf\ttext\n"
    "5\t1\t1\t1\t1\t1\t10\t20\t40\t10\t96.0\t\"Quoted\n"
    "5\t1\t1\t1\t1\t2\t55\t20\t15\t10\t41.0\tName\n"
)


def _packet(project: Path):
    evidence = project / "evidence"
    evidence.mkdir()
    Image.new("RGB", (20, 20), "white").save(evidence / "a.png")
    Image.new("RGB", (20, 20), "black").save(evidence / "b.png")
    return create_vision_packet(
        candidate_id="VC000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/a.png",
                "role": "before",
                "requested_ms": 0,
                "actual_ms": 0,
                "raw_pts": 0,
                "time_base": "1/1000",
            },
            {
                "frame_id": "F000002",
                "path": "evidence/b.png",
                "role": "after",
                "requested_ms": 900,
                "actual_ms": 1000,
                "raw_pts": 1000,
                "time_base": "1/1000",
            },
        ],
        raw_ocr=[
            {
                "observation_id": "O000001",
                "frame_id": "F000001",
                "raw_engine_text": "42",
                "normalized_interpretation": "42",
                "confidence": 0.9,
            }
        ],
        questions=["What visible state changed?"],
    )


def _annotation() -> dict[str, object]:
    return {
        "candidate_id": "VC000001",
        "factual_visible_description": "The result panel changes from white to black.",
        "event_type": "visible_state_change",
        "evidence_frame_ids": ["F000001", "F000002"],
        "before_action_after_roles": {"F000001": "before", "F000002": "after"},
        "exact_visible_text_candidates": [],
        "consequential_changes": [
            {
                "statement": "The panel becomes black.",
                "before_frame_id": "F000001",
                "after_frame_ids": ["F000002"],
                "confidence": 0.95,
            }
        ],
        "confidence": 0.95,
        "uncertainty": [],
        "statements_not_inferred": ["No cause or user identity is inferred."],
    }


def test_ocr_keeps_raw_normalized_and_uncertainty_distinct() -> None:
    observation = parse_tesseract_tsv(
        TSV,
        observation_id="O000001",
        frame_id="F000001",
        crop_id=None,
        language="eng",
        engine_version="tesseract 5",
    )
    assert observation.raw_engine_text == "Value: 4?"
    assert observation.normalized_interpretation == "Value: 4?"
    assert observation.uncertain_characters[0]["text"] == "4?"
    assert normalize_ocr_text(" A  B\r\n C ") == "A B\nC"


def test_tesseract_literal_quote_does_not_swallow_following_tsv_rows() -> None:
    observation = parse_tesseract_tsv(
        QUOTED_TSV,
        observation_id="O000002",
        frame_id="F000001",
        crop_id=None,
        language="eng",
        engine_version="tesseract 5",
    )

    assert len(observation.tokens) == 2
    assert observation.raw_engine_text == '"Quoted Name'
    assert observation.tokens[1].text == "Name"


def test_tesseract_discovers_explicit_environment_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"fixture executable")
    monkeypatch.setenv("VSR_TESSERACT_PATH", str(executable))
    assert TesseractOCRAdapter().executable == str(executable)


def test_tesseract_version_probe_is_single_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import video_script_reconstructor.ocr as ocr_module

    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"fixture executable")
    adapter = TesseractOCRAdapter(executable=str(executable))
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="tesseract 5.0 fixture\n", stderr=""
        )

    monkeypatch.setattr(ocr_module.subprocess, "run", fake_run)
    with ThreadPoolExecutor(max_workers=8) as pool:
        versions = list(pool.map(lambda _index: adapter.version(), range(16)))
    assert TesseractOCRAdapter(executable=str(executable)).version() == "tesseract 5.0 fixture"

    assert versions == ["tesseract 5.0 fixture"] * 16
    assert calls == [[str(executable), "--version"]]


def test_annotation_schema_rejects_ungrounded_frame(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    annotation = _annotation()
    annotation["evidence_frame_ids"] = ["F999999"]
    with pytest.raises(ValidationFailure, match="outside its packet"):
        validate_annotation_for_packet(annotation, packet)


def test_annotation_schema_requires_zero_confidence_for_semantic_pending(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    annotation = _annotation()
    annotation.update(
        {
            "event_type": "semantic_pending",
            "confidence": 0.4,
            "factual_visible_description": "No defensible visible fact is established.",
        }
    )
    with pytest.raises(ValidationFailure, match="confidence 0"):
        validate_annotation_for_packet(annotation, packet)


def test_host_agent_handoff_is_pending_until_valid_annotation_exists(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    provider = HostAgentVisionProvider()
    with pytest.raises(ReviewRequired, match="pending host-agent inspection"):
        provider.annotate(packet, project_root=tmp_path)
    request_path = tmp_path / ".state" / "vision" / "VC000001.host-agent-request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["packet"]["candidate_id"] == "VC000001"
    response_path = tmp_path / ".state" / "vision" / "VC000001.annotation.json"
    response_path.write_text(json.dumps(_annotation()), encoding="utf-8")
    accepted = provider.annotate(packet, project_root=tmp_path)
    assert accepted.factual_visible_description.endswith("black.")


def test_host_agent_response_is_rejected_when_source_pixels_change(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    provider = HostAgentVisionProvider()
    with pytest.raises(ReviewRequired):
        provider.annotate(packet, project_root=tmp_path)
    request_path = tmp_path / ".state" / "vision" / "VC000001.host-agent-request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["packet_sha256"]
    assert request["frame_files"][0]["sha256"]
    response_path = tmp_path / ".state" / "vision" / "VC000001.annotation.json"
    response_path.write_text(json.dumps(_annotation()), encoding="utf-8")
    # Keep the packet identity unchanged while replacing one referenced image;
    # a same-ID legacy response must not be treated as current evidence.
    Image.new("RGB", (20, 20), "green").save(tmp_path / "evidence" / "a.png")
    with pytest.raises(ReviewRequired, match="stale"):
        provider.annotate(packet, project_root=tmp_path)
    refreshed = json.loads(request_path.read_text(encoding="utf-8"))
    assert refreshed["frame_files"][0]["sha256"] != request["frame_files"][0]["sha256"]


def test_local_provider_invokes_real_process_and_validates_output(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    script = "import json,sys; json.load(sys.stdin); print(json.dumps(" + repr(_annotation()) + "))"
    provider = LocalCommandVisionProvider(
        [sys.executable, "-c", script], provider_id="local-test-adapter", model="deterministic-test"
    )
    annotation = provider.annotate(packet, project_root=tmp_path)
    assert annotation.candidate_id == packet.candidate_id


def test_external_provider_requires_explicit_network_and_upload_permission() -> None:
    denied = ExternalProcessingPermission(
        allow_network=True, allow_external_service=True, allow_media_upload=False, offline=False
    )
    with pytest.raises(SecurityError, match="explicit permission"):
        ExternalVisionProvider(
            endpoint="https://example.com/vision",
            provider_id="remote",
            model="vision",
            permission=denied,
        )


def test_visual_events_are_grounded_or_explicitly_pending(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    accepted = VisionAnnotation.model_validate(_annotation())
    built = annotation_to_visual_event(accepted, packet, event_index=1, annotation_provider="local")
    pending = pending_visual_event(packet, event_index=2)
    assert built.event.evidence_frame_ids == ["F000001", "F000002"]
    assert built.event.importance == "consequential"
    assert pending.review_required is True
    assert "semantic description pending review" in pending.event.factual_grounded_description
