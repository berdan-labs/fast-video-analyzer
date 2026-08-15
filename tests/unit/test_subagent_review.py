from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import video_script_reconstructor.semantic_pipeline as semantic_module
import video_script_reconstructor.subagent_review as subagent_review_module
from video_script_reconstructor.errors import ValidationFailure
from video_script_reconstructor.providers import CodexSubagentVisionProvider
from video_script_reconstructor.providers.base import ProviderDescriptor, VisionProvider
from video_script_reconstructor.security import atomic_write_json, sha256_file
from video_script_reconstructor.subagent_review import (
    _archive_legacy_sources,
    _request_payload,
    _script_context,
    _select_legacy_review_packets,
    _verify_bundle_inputs,
    apply_review_bundle,
    create_review_bundle,
)
from video_script_reconstructor.vision_packets import (
    VisionAnnotation,
    VisionPacket,
    create_vision_packet,
)


def _bundle_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    project_dir = tmp_path / "project"
    packet_dir = project_dir / ".state" / "vision" / "packets"
    evidence_dir = project_dir / "evidence"
    packet_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    image_path = evidence_dir / "frame.png"
    Image.new("RGB", (16, 16), "blue").save(image_path)
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What is visible?"],
    )
    packet_path = packet_dir / "V000001.json"
    atomic_write_json(packet_path, packet.model_dump(mode="json"))
    canonical_path = project_dir / ".state" / "canonical-project.json"
    atomic_write_json(
        canonical_path,
        {
            "manifest": {"run_id": "RUN000001"},
            "visual_events": [{"event_id": "V000001"}],
        },
    )
    project = json.loads(canonical_path.read_text(encoding="utf-8"))
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "requests").mkdir(parents=True)
    (bundle_dir / "responses").mkdir()
    response_rel = "responses/V000001.annotation.json"
    request = _request_payload(
        project_dir,
        project,
        packet,
        packet_path=packet_path,
        response_path=response_rel,
    )
    request_rel = "requests/V000001.json"
    atomic_write_json(bundle_dir / request_rel, request)
    manifest_entry = {
        "candidate_id": packet.candidate_id,
        "request_path": request_rel,
        "response_path": response_rel,
        "packet_path": packet_path.relative_to(project_dir).as_posix(),
        "packet_sha256": request["packet_sha256"],
        "frame_files": request["frame_files"],
    }
    manifest = {
        "schema_name": "video-script-reconstructor.subagent-review-bundle",
        "schema_version": "1.0",
        "project_dir": str(project_dir),
        "canonical_project_sha256": sha256_file(canonical_path),
        "requests": [manifest_entry],
    }
    atomic_write_json(bundle_dir / "bundle.json", manifest)
    return project_dir, bundle_dir, manifest


def test_review_request_requires_bounded_visible_fact_before_pending(tmp_path: Path) -> None:
    project_dir, bundle_dir, _manifest = _bundle_fixture(tmp_path)
    del project_dir
    request = json.loads((bundle_dir / "requests" / "V000001.json").read_text(encoding="utf-8"))
    rules = request["review_rules"]
    assert any("Do not choose semantic_pending merely" in rule for rule in rules)
    assert any("after inspecting every referenced PNG" in rule for rule in rules)


def test_bundle_verification_requires_request_file_and_payload_match(tmp_path: Path) -> None:
    project_dir, bundle_dir, manifest = _bundle_fixture(tmp_path)

    _verify_bundle_inputs(project_dir, bundle_dir, manifest)
    request_path = bundle_dir / "requests" / "V000001.json"
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    payload["candidate_id"] = "V999999"
    atomic_write_json(request_path, payload)
    with pytest.raises(ValidationFailure, match="disagrees with bundle manifest"):
        _verify_bundle_inputs(project_dir, bundle_dir, manifest)

    request_path.unlink()
    with pytest.raises(ValidationFailure, match="request file is missing"):
        _verify_bundle_inputs(project_dir, bundle_dir, manifest)


def test_bundle_verification_hashes_reused_frame_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir, bundle_dir, manifest = _bundle_fixture(tmp_path)
    canonical_path = project_dir / ".state" / "canonical-project.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["visual_events"].append({"event_id": "V000002"})
    atomic_write_json(canonical_path, canonical)
    packet_dir = project_dir / ".state" / "vision" / "packets"
    packet_one_path = packet_dir / "V000001.json"
    packet_one = VisionPacket.model_validate(
        json.loads(packet_one_path.read_text(encoding="utf-8"))
    )
    packet_two = packet_one.model_copy(update={"candidate_id": "V000002"})
    packet_two_path = packet_dir / "V000002.json"
    atomic_write_json(packet_two_path, packet_two.model_dump(mode="json"))
    request_two_rel = "requests/V000002.json"
    request_two = _request_payload(
        project_dir,
        canonical,
        packet_two,
        packet_path=packet_two_path,
        response_path="responses/V000002.annotation.json",
    )
    atomic_write_json(bundle_dir / request_two_rel, request_two)
    manifest["canonical_project_sha256"] = sha256_file(canonical_path)
    manifest["requests"].append(
        {
            "candidate_id": "V000002",
            "request_path": request_two_rel,
            "response_path": request_two["response_path"],
            "packet_path": request_two["packet_path"],
            "packet_sha256": request_two["packet_sha256"],
            "frame_files": request_two["frame_files"],
        }
    )
    atomic_write_json(bundle_dir / "bundle.json", manifest)
    real_sha256_file = subagent_review_module.sha256_file
    shared_frame = (project_dir / "evidence" / "frame.png").resolve()
    frame_hash_calls = 0

    def counting_sha256_file(path: Path) -> str:
        nonlocal frame_hash_calls
        if path.resolve() == shared_frame:
            frame_hash_calls += 1
        return real_sha256_file(path)

    monkeypatch.setattr(subagent_review_module, "sha256_file", counting_sha256_file)
    _verify_bundle_inputs(project_dir, bundle_dir, manifest)
    assert frame_hash_calls == 1


def test_request_payload_reuses_frame_hash_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir, _bundle_dir, _manifest = _bundle_fixture(tmp_path)
    packet_path = project_dir / ".state" / "vision" / "packets" / "V000001.json"
    packet = VisionPacket.model_validate(json.loads(packet_path.read_text(encoding="utf-8")))
    second_packet = packet.model_copy(update={"candidate_id": "V000002"})
    shared_frame = (project_dir / "evidence" / "frame.png").resolve()
    real_sha256_file = subagent_review_module.sha256_file
    frame_hash_calls = 0

    def counting_sha256_file(path: Path) -> str:
        nonlocal frame_hash_calls
        if path.resolve() == shared_frame:
            frame_hash_calls += 1
        return real_sha256_file(path)

    monkeypatch.setattr(subagent_review_module, "sha256_file", counting_sha256_file)
    frame_hash_cache: dict[Path, str] = {}
    _request_payload(
        project_dir,
        {"visual_events": []},
        packet,
        packet_path=packet_path,
        response_path="responses/V000001.annotation.json",
        frame_hash_cache=frame_hash_cache,
    )
    _request_payload(
        project_dir,
        {"visual_events": []},
        second_packet,
        packet_path=packet_path,
        response_path="responses/V000002.annotation.json",
        frame_hash_cache=frame_hash_cache,
    )
    assert frame_hash_calls == 1


def test_legacy_provider_bundle_is_explicit_and_preserves_source_hashes(
    tmp_path: Path,
) -> None:
    project_dir, _unused_bundle, _unused_manifest = _bundle_fixture(tmp_path)
    canonical_path = project_dir / ".state" / "canonical-project.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["visual_events"][0]["annotation_provider"] = "llama.cpp-local"
    atomic_write_json(canonical_path, canonical)
    source_root = project_dir / ".state" / "vision" / "annotations"
    source_root.mkdir(parents=True)
    annotation_source = source_root / "V000001.annotation.json"
    observation_source = source_root / "V000001.observation.json"
    annotation_source.write_text('{"event_type":"visible_state_change"}\n', encoding="utf-8")
    observation_source.write_text('{"actor_label":"llama.cpp-local"}\n', encoding="utf-8")
    canonical_hash = sha256_file(canonical_path)

    default_bundle = create_review_bundle(
        project_dir,
        output_dir=tmp_path / "default-bundle",
        max_packets=1,
    )
    assert default_bundle["request_count"] == 0
    assert sha256_file(canonical_path) == canonical_hash

    legacy_bundle = create_review_bundle(
        project_dir,
        output_dir=tmp_path / "legacy-bundle",
        max_packets=1,
        include_annotation_providers=["llama.cpp-local"],
    )
    assert legacy_bundle["request_count"] == 1
    assert legacy_bundle["selection_mode"] == "annotation_provider_re_review"
    assert legacy_bundle["include_annotation_providers"] == ["llama.cpp-local"]
    entry = legacy_bundle["requests"][0]
    assert entry["source_annotation_provider"] == "llama.cpp-local"
    assert entry["source_annotation_sha256"] == sha256_file(annotation_source)
    assert entry["source_observation_sha256"] == sha256_file(observation_source)
    _verify_bundle_inputs(project_dir, Path(legacy_bundle["bundle_dir"]), legacy_bundle)
    assert sha256_file(canonical_path) == canonical_hash

    observation_source.write_text('{"actor_label":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValidationFailure, match="Legacy source observation changed"):
        _verify_bundle_inputs(project_dir, Path(legacy_bundle["bundle_dir"]), legacy_bundle)


def test_legacy_selector_loads_only_matching_provider_packets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider re-review must not parse unrelated packets in a long project."""

    project_dir, _unused_bundle, _unused_manifest = _bundle_fixture(tmp_path)
    canonical_path = project_dir / ".state" / "canonical-project.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["visual_events"] = [
        {"event_id": "V000001", "annotation_provider": "llama.cpp-local"},
        {"event_id": "V000002", "annotation_provider": "other-provider"},
    ]
    atomic_write_json(canonical_path, canonical)
    packet_dir = project_dir / ".state" / "vision" / "packets"
    packet = create_vision_packet(
        candidate_id="V000002",
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What is visible?"],
    )
    atomic_write_json(packet_dir / "V000002.json", packet.model_dump(mode="json"))

    loaded: list[str] = []
    original_load = subagent_review_module._load_packet

    def counted_load(path: Path) -> VisionPacket:
        loaded.append(path.name)
        return original_load(path)

    monkeypatch.setattr(subagent_review_module, "_load_packet", counted_load)
    selected, deferred = _select_legacy_review_packets(
        project_dir,
        canonical,
        provider_ids=["llama.cpp-local"],
        max_packets=8,
    )

    assert [packet.candidate_id for _path, packet, _provider in selected] == ["V000001"]
    assert deferred == []
    assert loaded == ["V000001.json"]


def test_legacy_selector_prioritizes_explicitly_weak_historical_claims(
    tmp_path: Path,
) -> None:
    project_dir, _unused_bundle, _unused_manifest = _bundle_fixture(tmp_path)
    canonical_path = project_dir / ".state" / "canonical-project.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["visual_events"] = [
        {
            "event_id": "V000001",
            "annotation_provider": "llama.cpp-local",
            "event_type": "visible_state_change",
            "confidence": 0.99,
        },
        {
            "event_id": "V000002",
            "annotation_provider": "llama.cpp-local",
            "event_type": "semantic_pending",
            "confidence": 0.0,
            "uncertainty": ["Historical visual claim was not defensible."],
        },
    ]
    atomic_write_json(canonical_path, canonical)
    packet_dir = project_dir / ".state" / "vision" / "packets"
    pending_packet = create_vision_packet(
        candidate_id="V000002",
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What is visible?"],
    )
    atomic_write_json(packet_dir / "V000002.json", pending_packet.model_dump(mode="json"))
    # The routine event has a stronger measured scene score, but the explicit
    # pending/risk markers must win the bounded host-agent frontier.
    routine_packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What is visible?"],
        scene_motion_metadata={"selection_reason": "hard-scene scene-cut consequential ocr adaptive"},
    )
    atomic_write_json(packet_dir / "V000001.json", routine_packet.model_dump(mode="json"))

    selected, deferred = _select_legacy_review_packets(
        project_dir,
        canonical,
        provider_ids=["llama.cpp-local"],
        max_packets=1,
    )

    assert [packet.candidate_id for _path, packet, _provider in selected] == ["V000002"]
    assert deferred == ["V000001"]


def test_legacy_source_archive_is_byte_preserving_and_scoped(tmp_path: Path) -> None:
    project_dir, _unused_bundle, _unused_manifest = _bundle_fixture(tmp_path)
    source_root = project_dir / ".state" / "vision" / "annotations"
    source_root.mkdir(parents=True)
    annotation_source = source_root / "V000001.annotation.json"
    observation_source = source_root / "V000001.observation.json"
    annotation_bytes = b'{"legacy":true}\n'
    observation_bytes = b'{"provider":"llama.cpp-local"}\n'
    annotation_source.write_bytes(annotation_bytes)
    observation_source.write_bytes(observation_bytes)
    manifest = {
        "bundle_id": "SBLEGACYTEST",
        "requests": [
            {
                "candidate_id": "V000001",
                "source_annotation_path": ".state/vision/annotations/V000001.annotation.json",
                "source_observation_path": ".state/vision/annotations/V000001.observation.json",
            }
        ],
    }
    archive = _archive_legacy_sources(
        project_dir,
        tmp_path / "external-bundle",
        manifest,
        ["V000001"],
    )
    assert (archive / annotation_source.name).read_bytes() == annotation_bytes
    assert (archive / observation_source.name).read_bytes() == observation_bytes
    assert (archive / "archive.json").is_file()


def test_legacy_bundle_apply_archives_old_sidecars_before_codex_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir, _unused_bundle, _unused_manifest = _bundle_fixture(tmp_path)
    canonical_path = project_dir / ".state" / "canonical-project.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["visual_events"][0]["annotation_provider"] = "llama.cpp-local"
    atomic_write_json(canonical_path, canonical)
    source_root = project_dir / ".state" / "vision" / "annotations"
    source_root.mkdir(parents=True)
    annotation_source = source_root / "V000001.annotation.json"
    observation_source = source_root / "V000001.observation.json"
    annotation_bytes = b'{"legacy":true}\n'
    observation_bytes = b'{"provider":"llama.cpp-local"}\n'
    annotation_source.write_bytes(annotation_bytes)
    observation_source.write_bytes(observation_bytes)
    bundle = create_review_bundle(
        project_dir,
        output_dir=tmp_path / "legacy-bundle",
        include_annotation_providers=["llama.cpp-local"],
    )
    request = bundle["requests"][0]
    packet = VisionPacket.model_validate(
        json.loads((project_dir / request["packet_path"]).read_text(encoding="utf-8"))
    )
    focus = packet.frames[0]
    response = VisionAnnotation(
        candidate_id=packet.candidate_id,
        factual_visible_description="The supplied frame is visible.",
        event_type="visible_state_change",
        evidence_frame_ids=[focus.frame_id],
        before_action_after_roles={focus.frame_id: "context"},
        exact_visible_text_candidates=[],
        consequential_changes=[],
        confidence=0.5,
        uncertainty=[],
        statements_not_inferred=["No identity or intent is inferred."],
    )
    response_path = Path(bundle["bundle_dir"]) / request["response_path"]
    response_path.write_text(response.model_dump_json(), encoding="utf-8")
    semantic_call: dict[str, object] = {}

    def fake_semantic_pass(*args: object, **kwargs: object) -> dict[str, object]:
        semantic_call.update(kwargs)
        return {
            "status": "review_required",
            "applied": [{"observation_id": "VA000001"}],
            "validation_errors": [],
            "validation": {},
        }

    monkeypatch.setattr(
        subagent_review_module,
        "run_semantic_pass",
        fake_semantic_pass,
    )

    result = apply_review_bundle(
        project_dir,
        Path(bundle["bundle_dir"]),
        semantic_workers=2,
    )

    assert result["legacy_re_review"] is True
    assert semantic_call["allow_observed_candidate_ids"] is True
    assert semantic_call["semantic_workers"] == 2
    archive_dir = Path(result["legacy_source_archive_dir"])
    assert (archive_dir / annotation_source.name).read_bytes() == annotation_bytes
    assert (archive_dir / observation_source.name).read_bytes() == observation_bytes


def test_bundle_apply_resumes_remaining_responses_from_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial Codex apply can continue without replaying stale packets."""

    project_dir, bundle_dir, manifest = _bundle_fixture(tmp_path)
    canonical_path = project_dir / ".state" / "canonical-project.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["visual_events"].append({"event_id": "V000002"})
    atomic_write_json(canonical_path, canonical)
    manifest["bundle_id"] = "SBRESUMETEST"
    manifest["canonical_project_sha256"] = sha256_file(canonical_path)

    packet_dir = project_dir / ".state" / "vision" / "packets"
    packet_one = VisionPacket.model_validate(
        json.loads((packet_dir / "V000001.json").read_text(encoding="utf-8"))
    )
    packet_two = packet_one.model_copy(update={"candidate_id": "V000002"})
    packet_two_path = packet_dir / "V000002.json"
    atomic_write_json(packet_two_path, packet_two.model_dump(mode="json"))
    response_two_rel = "responses/V000002.annotation.json"
    request_two = _request_payload(
        project_dir,
        canonical,
        packet_two,
        packet_path=packet_two_path,
        response_path=response_two_rel,
    )
    atomic_write_json(bundle_dir / "requests/V000002.json", request_two)
    manifest["requests"].append(
        {
            "candidate_id": "V000002",
            "request_path": "requests/V000002.json",
            "response_path": response_two_rel,
            "packet_path": request_two["packet_path"],
            "packet_sha256": request_two["packet_sha256"],
            "frame_files": request_two["frame_files"],
        }
    )
    atomic_write_json(bundle_dir / "bundle.json", manifest)

    def response_for(packet: VisionPacket) -> VisionAnnotation:
        frame_id = packet.frames[0].frame_id
        return VisionAnnotation(
            candidate_id=packet.candidate_id,
            factual_visible_description="The supplied frame is visible.",
            event_type="visible_state",
            evidence_frame_ids=[frame_id],
            before_action_after_roles={frame_id: "context"},
            exact_visible_text_candidates=[],
            consequential_changes=[],
            confidence=0.5,
            uncertainty=[],
            statements_not_inferred=["No identity, speech, motion, intent, or hidden state is inferred."],
        )

    response_one_path = bundle_dir / "responses/V000001.annotation.json"
    response_one_path.write_text(response_for(packet_one).model_dump_json(), encoding="utf-8")
    calls: list[set[str]] = []

    def fake_semantic_pass(*args: object, **kwargs: object) -> dict[str, object]:
        candidate_ids = {str(value) for value in kwargs["candidate_ids"]}
        calls.append(candidate_ids)
        project = json.loads(canonical_path.read_text(encoding="utf-8"))
        for event in project["visual_events"]:
            if event.get("event_id") in candidate_ids:
                event["annotation_provider"] = "codex-subagent"
                event["event_type"] = "visible_state"
        atomic_write_json(canonical_path, project)
        return {
            "status": "review_required",
            "applied": [{"candidate_id": value} for value in sorted(candidate_ids)],
            "validation_errors": [],
            "validation": {},
        }

    monkeypatch.setattr(subagent_review_module, "run_semantic_pass", fake_semantic_pass)

    first = apply_review_bundle(project_dir, bundle_dir)
    assert first["missing_candidate_ids"] == ["V000002"]
    assert calls == [{"V000001"}]
    receipt = json.loads((bundle_dir / "apply-result.json").read_text(encoding="utf-8"))
    assert receipt["post_apply_canonical_project_sha256"] == sha256_file(canonical_path)

    response_two_path = bundle_dir / response_two_rel
    response_two_path.write_text(response_for(packet_two).model_dump_json(), encoding="utf-8")
    second = apply_review_bundle(project_dir, bundle_dir)
    assert second["missing_candidate_ids"] == []
    assert second["pending_response_candidate_ids"] == ["V000002"]
    assert calls == [{"V000001"}, {"V000002"}]


def test_bundle_apply_reuses_packets_verified_before_response_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir, bundle_dir, _manifest = _bundle_fixture(tmp_path)
    packet_path = project_dir / ".state" / "vision" / "packets" / "V000001.json"
    packet = VisionPacket.model_validate(json.loads(packet_path.read_text(encoding="utf-8")))
    frame_id = packet.frames[0].frame_id
    response = VisionAnnotation(
        candidate_id=packet.candidate_id,
        factual_visible_description="A blue frame is visible.",
        event_type="visible_state",
        evidence_frame_ids=[frame_id],
        before_action_after_roles={frame_id: "context"},
        exact_visible_text_candidates=[],
        consequential_changes=[],
        confidence=0.5,
        uncertainty=[],
        statements_not_inferred=["No identity, speech, intent, or hidden state is inferred."],
    )
    (bundle_dir / "responses" / "V000001.annotation.json").write_text(
        response.model_dump_json(), encoding="utf-8"
    )
    real_load_packet = subagent_review_module._load_packet
    load_count = 0

    def counting_load_packet(path: Path) -> VisionPacket:
        nonlocal load_count
        load_count += 1
        return real_load_packet(path)

    monkeypatch.setattr(subagent_review_module, "_load_packet", counting_load_packet)
    monkeypatch.setattr(
        subagent_review_module,
        "run_semantic_pass",
        lambda *args, **kwargs: {
            "status": "review_required",
            "applied": [{"candidate_id": "V000001"}],
            "validation_errors": [],
            "validation": {},
        },
    )

    result = apply_review_bundle(project_dir, bundle_dir)

    assert result["response_candidate_ids"] == ["V000001"]
    # One load during immutable bundle verification; response validation uses
    # that verified packet object rather than reading/parsing it again.
    assert load_count == 1


def test_bundle_verification_binds_bounded_cumulative_metadata_context(tmp_path: Path) -> None:
    project_dir, bundle_dir, manifest = _bundle_fixture(tmp_path)
    canonical_path = project_dir / ".state" / "canonical-project.json"
    project = json.loads(canonical_path.read_text(encoding="utf-8"))
    project["frames"] = [
        {
            "frame_id": "F000001",
            "latest_revision_id": "MR000007",
            "metadata": {
                "analysis": {
                    "enrichment_level": "semantic",
                    "semantic_status": "observed",
                },
                "knowledge": {
                    "current_factual_description": "A blue status panel is visible.",
                    "supported_claim_ids": ["IC000001"],
                    "disputed_claim_ids": ["IC000002"],
                    "unresolved_claim_ids": ["IC000003"],
                    "explicit_unknowns": ["The small footer text is unclear."],
                    "statements_not_inferred": ["No identity is inferred."],
                    "claims": [
                        {
                            "claim_id": "IC000001",
                            "claim_class": "direct_visible",
                            "statement": "A blue status panel is visible.",
                            "status": "supported",
                            "confidence": 0.9,
                            "supporting_observation_ids": ["VA000001"],
                        }
                    ],
                },
            },
        }
    ]
    atomic_write_json(canonical_path, project)
    manifest["canonical_project_sha256"] = sha256_file(canonical_path)
    request_path = bundle_dir / manifest["requests"][0]["request_path"]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    # The request was created before the metadata was installed; rebuild only
    # this fixture request and keep the bundle's normal hash/path gates intact.
    packet_path = project_dir / manifest["requests"][0]["packet_path"]
    packet = VisionPacket.model_validate(json.loads(packet_path.read_text(encoding="utf-8")))
    request = _request_payload(
        project_dir,
        project,
        packet,
        packet_path=packet_path,
        response_path=manifest["requests"][0]["response_path"],
    )
    atomic_write_json(request_path, request)
    manifest["requests"][0]["packet_sha256"] = request["packet_sha256"]
    manifest["requests"][0]["frame_files"] = request["frame_files"]
    atomic_write_json(bundle_dir / "bundle.json", manifest)

    _verify_bundle_inputs(project_dir, bundle_dir, manifest)
    context = request["metadata_context"][0]
    assert context["semantic_status"] == "observed"
    assert context["claims"][0]["claim_id"] == "IC000001"
    assert context["disputed_claim_ids"] == ["IC000002"]

    request["metadata_context"][0]["semantic_status"] = "creation"
    atomic_write_json(request_path, request)
    with pytest.raises(ValidationFailure, match="metadata context changed"):
        _verify_bundle_inputs(project_dir, bundle_dir, manifest)


def test_bundle_verification_rejects_duplicate_and_unknown_candidates(tmp_path: Path) -> None:
    project_dir, bundle_dir, manifest = _bundle_fixture(tmp_path)
    manifest["requests"] = [manifest["requests"][0], manifest["requests"][0]]
    with pytest.raises(ValidationFailure, match="Duplicate subagent candidate ID"):
        _verify_bundle_inputs(project_dir, bundle_dir, manifest)

    project_dir, bundle_dir, manifest = _bundle_fixture(tmp_path / "unknown")
    manifest["requests"][0]["candidate_id"] = "V999999"
    with pytest.raises(ValidationFailure, match="Unknown subagent candidate ID"):
        _verify_bundle_inputs(project_dir, bundle_dir, manifest)


def test_filtered_semantic_pass_reconciles_global_deferred_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixtureProvider(VisionProvider):
        @property
        def descriptor(self) -> ProviderDescriptor:
            return ProviderDescriptor(
                provider_id="codex-subagent",
                route="host_agent",
                model="codex-subagent",
                model_version=None,
                adapter_version="1.0",
                network_required=False,
            )

        def annotate(self, packet: VisionPacket, *, project_root: Path) -> VisionAnnotation:
            raise AssertionError("filtered frontier test should not invoke a provider")

    project = {
        "frames": [],
        "visual_events": [],
        "review_items": [
            {
                "review_id": "R000001",
                "category": "semantic_budget_deferred",
                "decision": None,
                "event_ids": ["V000001"],
            }
        ],
    }
    monkeypatch.setattr(semantic_module, "_load_project", lambda _path: project)
    monkeypatch.setattr(semantic_module, "_load_ledger", lambda _path: {})
    monkeypatch.setattr(
        semantic_module,
        "_select_semantic_packet_files",
        lambda *_args, **_kwargs: ([], []),
    )
    monkeypatch.setattr(semantic_module, "_pending_semantic_event_ids", lambda *_args: ["V000002"])
    monkeypatch.setattr(semantic_module, "_finalize_semantic_project", lambda *_args, **_kwargs: None)

    summary = semantic_module.apply_vision_provider(
        tmp_path,
        FixtureProvider(),
        candidate_ids={"V000001"},
    )

    assert project["review_items"][0]["event_ids"] == ["V000002"]
    assert summary["semantic_deferred_event_ids"] == ["V000002"]


def test_codex_subagent_provider_is_public_and_file_only(tmp_path: Path) -> None:
    provider = CodexSubagentVisionProvider(response_root=tmp_path)
    assert provider.descriptor.route == "host_agent"
    assert provider.descriptor.network_required is False
    assert provider.semantic_cacheable is False


def test_bundle_request_context_includes_visual_and_claim_links(tmp_path: Path) -> None:
    project_dir, _bundle_dir, project = _bundle_fixture(tmp_path)
    project["script_blocks"] = [
        {
            "block_id": "B000001",
            "start_ms": 0,
            "end_ms": 1000,
            "frame_ids": ["F000001"],
            "visual_description": "A visible slide is pending review.",
            "on_screen_text": ["LOAD"],
            "visual_event_ids": ["V000001"],
            "image_claim_ids": ["IC000001"],
            "transcript_segment_ids": ["TS000001"],
        }
    ]
    packet_path = project_dir / ".state" / "vision" / "packets" / "V000001.json"
    packet = _load_packet_for_test(packet_path)
    request = _request_payload(
        project_dir,
        project,
        packet,
        packet_path=packet_path,
        response_path="responses/V000001.annotation.json",
    )

    context = request["script_context"]
    assert context[0]["visual_description"] == "A visible slide is pending review."
    assert context[0]["on_screen_text"] == ["LOAD"]
    assert context[0]["visual_event_ids"] == ["V000001"]
    assert context[0]["image_claim_ids"] == ["IC000001"]
    assert context[0]["transcript_segment_ids"] == ["TS000001"]


def test_script_context_prioritizes_uncertainty_and_bounds_values(tmp_path: Path) -> None:
    project_dir, _bundle_dir, project = _bundle_fixture(tmp_path)
    packet_path = project_dir / ".state" / "vision" / "packets" / "V000001.json"
    packet = _load_packet_for_test(packet_path)
    project["script_blocks"] = [
        {
            "block_id": f"B{i:02d}",
            "start_ms": 0,
            "end_ms": 2_000,
            "spoken_text": "r" * 2_000,
            "frame_ids": [],
        }
        for i in range(6)
    ]
    project["script_blocks"].append(
        {
            "block_id": "B99",
            "start_ms": 0,
            "end_ms": 2_000,
            "verification_status": "review_required",
            "uncertainty_items": ["u" * 1_000] * 20,
            "image_claim_ids": [f"IC{i}" for i in range(30)],
            "spoken_text": "x" * 2_000,
            "frame_ids": [],
        }
    )

    context = _script_context(project, packet)
    assert len(context) == 6
    assert context[0]["block_id"] == "B99"
    assert len(context[0]["spoken_text"]) == 1_200
    assert len(context[0]["uncertainty_items"]) == 8
    assert len(context[0]["image_claim_ids"]) == 12
    assert context == _script_context(project, packet)


def _load_packet_for_test(path: Path) -> VisionPacket:
    return VisionPacket.model_validate(json.loads(path.read_text(encoding="utf-8")))
