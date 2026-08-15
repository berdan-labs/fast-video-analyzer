from __future__ import annotations

import json
from pathlib import Path

import video_script_reconstructor.semantic_pipeline as semantic_pipeline
from video_script_reconstructor.providers.base import ProviderDescriptor, VisionProvider
from video_script_reconstructor.security import atomic_write_json
from video_script_reconstructor.semantic_pipeline import (
    _annotate_parallel_packet,
    _annotate_with_transport_identity,
    _deterministic_identical_frame_annotation,
    _deterministic_stable_frame_annotation,
    _is_retryable_semantic_pending,
    _packet_frame_hashes,
    _prune_unmirrored_generated_candidates,
    _read_semantic_cache,
    _read_semantic_content_cache,
    _remap_reused_annotation,
    _retryable_semantic_counts,
    _select_semantic_packet_files,
    _semantic_cache_key,
    _semantic_content_cache_key,
    _semantic_failure_is_provider_health_failure,
    _semantic_packet_score,
    _semantic_pending_retry_priority,
    _semantic_visual_content_key,
    _semantic_visual_reuse_key,
    _sync_semantic_budget_review_item,
    _upgrade_semantic_pending_annotation,
    _write_semantic_cache,
    _write_semantic_content_cache,
    _write_semantic_sidecars,
)
from video_script_reconstructor.vision_packets import (
    DifferenceRegion,
    VisionAnnotation,
    create_vision_packet,
)


def test_semantic_sidecars_use_compact_atomic_json(tmp_path: Path) -> None:
    class Payload:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"long": ["value"] * 3, "nested": {"ok": True}}

    _write_semantic_sidecars(
        tmp_path / "annotation.json",
        tmp_path / "observation.json",
        Payload(),
        Payload(),
    )

    for path in (tmp_path / "annotation.json", tmp_path / "observation.json"):
        text = path.read_text(encoding="utf-8")
        assert text.count("\n") == 1
        assert json.loads(text) == {"long": ["value"] * 3, "nested": {"ok": True}}


def test_pending_response_upgrades_only_from_pixel_identical_packet() -> None:
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/frame-a.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            },
            {
                "frame_id": "F000002",
                "path": "evidence/frame-b.png",
                "role": "after",
                "requested_ms": 1000,
                "actual_ms": 1000,
            },
        ],
        questions=["What is visible?"],
    )
    pending = VisionAnnotation(
        candidate_id=packet.candidate_id,
        factual_visible_description="No defensible visible semantic fact is established.",
        event_type="semantic_pending",
        evidence_frame_ids=["F000001"],
        before_action_after_roles={"F000001": "context"},
        exact_visible_text_candidates=[],
        consequential_changes=[],
        confidence=0.0,
        uncertainty=["The response was conservative."],
        statements_not_inferred=["No hidden state is inferred."],
    )
    frame_by_id = {
        frame_id: {"metadata": {"image": {"pixel_hash": {"value": "same"}}}}
        for frame_id in ("F000001", "F000002")
    }
    upgraded, did_upgrade = _upgrade_semantic_pending_annotation(
        pending, packet, frame_by_id
    )
    assert did_upgrade
    assert upgraded.event_type == "no_change"
    assert upgraded.confidence == 1.0
    assert upgraded.evidence_frame_ids == ["F000001", "F000002"]


def test_pending_response_stays_pending_when_pixels_are_not_stable() -> None:
    packet = create_vision_packet(
        candidate_id="V000002",
        frames=[
            {
                "frame_id": "F000003",
                "path": "evidence/frame-a.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            },
            {
                "frame_id": "F000004",
                "path": "evidence/frame-b.png",
                "role": "after",
                "requested_ms": 1000,
                "actual_ms": 1000,
            },
        ],
        questions=["What is visible?"],
    )
    pending = VisionAnnotation(
        candidate_id=packet.candidate_id,
        factual_visible_description="No defensible visible semantic fact is established.",
        event_type="semantic_pending",
        evidence_frame_ids=["F000003"],
        before_action_after_roles={"F000003": "context"},
        exact_visible_text_candidates=[],
        consequential_changes=[],
        confidence=0.0,
        uncertainty=["The response was conservative."],
        statements_not_inferred=["No hidden state is inferred."],
    )
    frame_by_id = {
        "F000003": {"metadata": {"image": {"pixel_hash": {"value": "a"}}}},
        "F000004": {"metadata": {"image": {"pixel_hash": {"value": "b"}}}},
    }
    upgraded, did_upgrade = _upgrade_semantic_pending_annotation(
        pending, packet, frame_by_id
    )
    assert not did_upgrade
    assert upgraded.event_type == "semantic_pending"
    assert upgraded.confidence == 0.0


def test_prune_unmirrored_candidates_removes_only_validator_named_files(tmp_path: Path) -> None:
    rejected = tmp_path / ".state" / "candidates" / "rejected-frames"
    rejected.mkdir(parents=True)
    stale = rejected / "F000001__00h00m01s000__full.png"
    mirrored = rejected / "F000002__00h00m02s000__full.png"
    stale.write_bytes(b"stale")
    mirrored.write_bytes(b"keep")

    removed = _prune_unmirrored_generated_candidates(
        tmp_path,
        [
            "image_metadata: hidden candidate "
            "F000001__00h00m01s000__full.png lacks a canonical ledger mirror",
            "image_metadata: hidden candidate "
            "F000002__00h00m02s000__full.png has a canonical ledger mirror",
        ],
    )

    assert removed == 1
    assert not stale.exists()
    assert mirrored.read_bytes() == b"keep"


def test_semantic_packet_score_normalizes_scene_reason_separators() -> None:
    base = create_vision_packet(
        candidate_id="V000001",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    )
    scene_cut = base.model_copy(
        update={"scene_motion_metadata": {"selection_reason": "scene_cut"}}
    )

    assert _semantic_packet_score(scene_cut) == _semantic_packet_score(base) + 8.0


def test_semantic_packet_score_prioritizes_event_importance_without_dropping_context() -> None:
    packet = create_vision_packet(
        candidate_id="V000001",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    )

    incidental = _semantic_packet_score(packet, {"importance": "incidental"})
    supporting = _semantic_packet_score(packet, {"importance": "supporting"})
    consequential = _semantic_packet_score(packet, {"importance": "consequential"})
    high_impact = _semantic_packet_score(packet, {"importance": "high_impact"})

    assert incidental < supporting < consequential < high_impact


def test_prune_unmirrored_candidates_rejects_path_traversal_and_other_errors(
    tmp_path: Path,
) -> None:
    rejected = tmp_path / ".state" / "candidates" / "rejected-frames"
    rejected.mkdir(parents=True)
    protected = tmp_path / "protected.png"
    protected.write_bytes(b"keep")

    removed = _prune_unmirrored_generated_candidates(
        tmp_path,
        [
            "image_metadata: hidden candidate ../protected.png "
            "lacks a canonical ledger mirror",
            "image_metadata: hidden candidate F000003.png lacks a canonical ledger mirror",
            "other validation failure",
        ],
    )

    assert removed == 0
    assert protected.read_bytes() == b"keep"


def test_semantic_visual_reuse_key_ignores_candidate_identity_but_keeps_scope() -> None:
    common = {
        "frames": [
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    }
    first = create_vision_packet(candidate_id="V000001", questions=["What is visible?"], **common)
    duplicate = create_vision_packet(
        candidate_id="V000002", questions=["What is visible?"], **common
    )
    different_scope = create_vision_packet(
        candidate_id="V000003", questions=["What changed?"], **common
    )

    assert _semantic_visual_reuse_key(first) == _semantic_visual_reuse_key(duplicate)
    assert _semantic_visual_reuse_key(first) != _semantic_visual_reuse_key(different_scope)


def test_semantic_circuit_classifies_shared_provider_health_failures_only() -> None:
    assert _semantic_failure_is_provider_health_failure(
        "Local llama.cpp vision request failed: HTTP Error 503: Service Unavailable"
    )
    assert _semantic_failure_is_provider_health_failure(
        "Local llama.cpp vision request failed: timed out"
    )
    assert not _semantic_failure_is_provider_health_failure(
        "Local llama.cpp vision request failed: HTTP Error 400: Bad Request"
    )
    assert not _semantic_failure_is_provider_health_failure(
        "Annotation must cite at least one focus/action/result frame from its packet"
    )
    assert not _semantic_failure_is_provider_health_failure(
        "Local vision server returned an invalid response: Unterminated string"
    )


def test_exact_visual_reuse_ignores_frame_ids_and_remaps_all_citations() -> None:
    first = create_vision_packet(
        candidate_id="V000001",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            },
            {
                "frame_id": "F000002",
                "path": "evidence/full/F000002.png",
                "role": "after",
                "requested_ms": 1000,
                "actual_ms": 1000,
            },
        ],
    )
    duplicate = create_vision_packet(
        candidate_id="V000002",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000101",
                "path": "evidence/full/F000101.png",
                "role": "focus",
                "requested_ms": 2000,
                "actual_ms": 2000,
            },
            {
                "frame_id": "F000102",
                "path": "evidence/full/F000102.png",
                "role": "after",
                "requested_ms": 3000,
                "actual_ms": 3000,
            },
        ],
    )
    frame_index = {
        "F000001": {"pixel_hash": {"value": "same-a"}},
        "F000002": {"pixel_hash": {"value": "same-b"}},
        "F000101": {"pixel_hash": {"value": "same-a"}},
        "F000102": {"pixel_hash": {"value": "same-b"}},
    }
    first_key = _semantic_visual_content_key(first, frame_index)
    duplicate_key = _semantic_visual_content_key(duplicate, frame_index)
    assert first_key is not None and duplicate_key is not None
    assert first_key[0] == duplicate_key[0]
    annotation = VisionAnnotation(
        candidate_id=first.candidate_id,
        factual_visible_description="A visible title card.",
        event_type="state",
        evidence_frame_ids=["F000001", "F000002"],
        before_action_after_roles={"F000001": "before", "F000002": "after"},
        confidence=0.9,
        uncertainty=[],
        statements_not_inferred=["Identity is not inferred."],
    )
    remapped = _remap_reused_annotation(
        annotation,
        source_frame_ids=first_key[1],
        packet=duplicate,
    )
    assert remapped.candidate_id == "V000002"
    assert remapped.evidence_frame_ids == ["F000101", "F000102"]
    assert dict(remapped.before_action_after_roles) == {
        "F000101": "before",
        "F000102": "after",
    }


def test_exact_visual_reuse_requires_every_frame_digest() -> None:
    packet = create_vision_packet(
        candidate_id="V000001",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    )
    assert _semantic_visual_content_key(packet, {"F000001": {}}) is None


def test_semantic_packet_cache_round_trip_includes_transport_profile(tmp_path: Path) -> None:
    packet = create_vision_packet(
        candidate_id="V000001",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    )
    provider = type(
        "FakeLlamaProvider",
        (),
        {
            "descriptor": ProviderDescriptor(
                provider_id="llama.cpp-local",
                route="local",
                model="fixture",
                model_version="1",
                adapter_version="1",
                network_required=False,
            ),
            "prompt_template_hash": "prompt-v1",
        },
    )()
    annotation = VisionAnnotation(
        candidate_id=packet.candidate_id,
        factual_visible_description="A blue field is visible.",
        event_type="visible_state",
        evidence_frame_ids=["F000001"],
        before_action_after_roles={"F000001": "context"},
        confidence=0.9,
        uncertainty=[],
        statements_not_inferred=["No hidden state is inferred."],
    )
    hashes = {"F000001": "pixel-a"}
    key = _semantic_cache_key(provider, packet, hashes)
    path = tmp_path / "packet-cache.json"
    assert _write_semantic_cache(
        path,
        key=key,
        provider=provider,
        packet=packet,
        annotation=annotation,
        frame_hashes=hashes,
        max_bytes=1024 * 1024,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cache_kind"] == "semantic-vision-v3-pixel-hash-transport-resize"
    assert payload["transport_profile"]["profile"] == "adaptive-no-ocr-single-v1"
    cache_text = path.read_text(encoding="utf-8")
    assert cache_text.endswith("\n")
    assert "\n  " not in cache_text
    reused = _read_semantic_cache(
        path,
        key=key,
        packet=packet,
        max_bytes=1024 * 1024,
    )
    assert reused is not None
    assert reused.candidate_id == packet.candidate_id


def test_packet_transport_identity_prefers_top_level_canonical_pixel_hash(tmp_path: Path) -> None:
    packet = create_vision_packet(
        candidate_id="V000001",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    )
    hashes = _packet_frame_hashes(
        {
            "frames": [
                {
                    "frame_id": "F000001",
                    "pixel_hash": {
                        "algorithm": "sha256-rgba8-srgb-v1",
                        "value": "canonical-pixels",
                    },
                    "file_hash": "container-with-project-specific-metadata",
                    "metadata": {"image": {"pixel_hash": {"value": "stale-mirror"}}},
                }
            ]
        },
        packet,
        tmp_path,
    )
    assert hashes == {"F000001": "canonical-pixels"}


def test_packet_transport_identity_reuses_prebuilt_frame_index(tmp_path: Path) -> None:
    """The semantic worker may supply its shared frame index without changing hashes."""

    packet = create_vision_packet(
        candidate_id="V000001",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    )
    prebuilt = {
        "F000001": {
            "frame_id": "F000001",
            "pixel_hash": {
                "algorithm": "sha256-rgba8-srgb-v1",
                "value": "shared-index-pixels",
            },
        }
    }

    # The project deliberately has no frames. Supplying the prebuilt index is
    # therefore both observable and sufficient to resolve the packet hash.
    hashes = _packet_frame_hashes({}, packet, tmp_path, frame_by_id=prebuilt)

    assert hashes == {"F000001": "shared-index-pixels"}


def test_local_transport_identity_hook_receives_hashes_without_changing_provider_contract(
    tmp_path: Path,
) -> None:
    packet = create_vision_packet(
        candidate_id="V000001",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    )
    annotation = VisionAnnotation(
        candidate_id=packet.candidate_id,
        factual_visible_description="A blue field is visible.",
        event_type="visible_state",
        evidence_frame_ids=["F000001"],
        before_action_after_roles={"F000001": "action"},
        confidence=0.8,
        uncertainty=[],
        statements_not_inferred=["No hidden state is inferred."],
    )

    class ContextualProvider:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def annotate(self, packet: object, *, project_root: Path) -> VisionAnnotation:
            del packet, project_root
            raise AssertionError("optional context hook should be preferred")

        def annotate_with_transport_context(
            self,
            packet: object,
            *,
            project_root: Path,
            transport_frame_hashes: dict[str, str],
        ) -> VisionAnnotation:
            del packet, project_root
            self.calls.append(dict(transport_frame_hashes))
            return annotation

    provider = ContextualProvider()
    result = _annotate_with_transport_identity(
        provider,  # type: ignore[arg-type]
        packet,
        project_dir=tmp_path,
        frame_hashes={"F000001": "canonical-pixels"},
    )
    assert result.candidate_id == "V000001"
    assert provider.calls == [{"F000001": "canonical-pixels"}]


def test_content_reuse_key_includes_ocr_and_prompt_context() -> None:
    base = create_vision_packet(
        candidate_id="V000001",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    )
    with_ocr = create_vision_packet(
        candidate_id="V000002",
        questions=["What exact text is visible?"],
        frames=[
            {
                "frame_id": "F000101",
                "path": "evidence/full/F000101.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        raw_ocr=[
            {
                "observation_id": "O000001",
                "frame_id": "F000101",
                "raw_engine_text": "ACCOUNT 42",
                "normalized_interpretation": "ACCOUNT 42",
                "confidence": 0.9,
                "uncertain_characters": [],
            }
        ],
    )
    index = {
        "F000001": {"pixel_hash": {"value": "same"}},
        "F000101": {"pixel_hash": {"value": "same"}},
    }
    first = _semantic_visual_content_key(base, index)
    second = _semantic_visual_content_key(with_ocr, index)
    assert first is not None and second is not None
    assert first[0] != second[0]


def test_pixel_identical_packet_uses_conservative_no_change_annotation() -> None:
    packet = create_vision_packet(
        candidate_id="V000001",
        questions=["What changed?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "before",
                "requested_ms": 0,
                "actual_ms": 0,
            },
            {
                "frame_id": "F000002",
                "path": "evidence/full/F000002.png",
                "role": "focus",
                "requested_ms": 500,
                "actual_ms": 500,
            },
            {
                "frame_id": "F000003",
                "path": "evidence/full/F000003.png",
                "role": "after",
                "requested_ms": 1000,
                "actual_ms": 1000,
            },
        ],
    )
    identical = {
        frame_id: {"metadata": {"image": {"pixel_hash": {"value": "same"}}}}
        for frame_id in ("F000001", "F000002", "F000003")
    }
    annotation = _deterministic_identical_frame_annotation(packet, identical)
    assert annotation is not None
    assert annotation.event_type == "no_change"
    assert annotation.confidence == 1.0
    assert annotation.evidence_frame_ids == ["F000001", "F000002", "F000003"]
    assert annotation.consequential_changes == []
    assert _deterministic_identical_frame_annotation(
        packet,
        {
            **identical,
            "F000003": {"metadata": {"image": {"pixel_hash": {"value": "changed"}}}},
        },
    ) is None


def test_perceptually_stable_packet_uses_conservative_stable_state_annotation() -> None:
    packet = create_vision_packet(
        candidate_id="V000001",
        questions=["What changed?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "before",
                "requested_ms": 0,
                "actual_ms": 0,
            },
            {
                "frame_id": "F000002",
                "path": "evidence/full/F000002.png",
                "role": "focus",
                "requested_ms": 500,
                "actual_ms": 500,
            },
        ],
        raw_ocr=[
            {
                "observation_id": "O000001",
                "frame_id": "F000001",
                "raw_engine_text": "Stable slide",
                "normalized_interpretation": "Stable slide",
                "confidence": 0.99,
                "uncertain_characters": [],
            },
            {
                "observation_id": "O000002",
                "frame_id": "F000002",
                "raw_engine_text": "Stable slide",
                "normalized_interpretation": "Stable slide",
                "confidence": 0.99,
                "uncertain_characters": [],
            },
        ],
    )
    frames = {
        "F000001": {
            "perceptual_hashes": {
                "dhash-8": "same",
                "dhash-8-algorithm": "dhash-8-v1",
                "dhash-8-verified": "true",
            }
        },
        "F000002": {
            "perceptual_hashes": {
                "dhash-8": "same",
                "dhash-8-algorithm": "dhash-8-v1",
                "dhash-8-verified": "true",
            }
        },
    }
    annotation = _deterministic_stable_frame_annotation(packet, frames)
    assert annotation is not None
    assert annotation.event_type == "stable_visible_state"
    assert annotation.confidence == 0.98
    assert annotation.evidence_frame_ids == ["F000001", "F000002"]
    assert _deterministic_stable_frame_annotation(
        packet,
        {
            "F000001": {"perceptual_hashes": {"dhash-8": "same"}},
            "F000002": {"perceptual_hashes": {"dhash-8": "same"}},
        },
    ) is None
    assert _deterministic_stable_frame_annotation(
        packet,
        {
            **frames,
            "F000002": {
                "perceptual_hashes": {
                    "dhash-8": "changed",
                    "dhash-8-algorithm": "dhash-8-v1",
                    "dhash-8-verified": "true",
                }
            },
        },
    ) is None


def test_semantic_budget_does_not_consume_vlm_slots_for_pixel_identical_packets(
    tmp_path: Path,
) -> None:
    packet_dir = tmp_path / ".state" / "vision" / "packets"
    packet_dir.mkdir(parents=True)

    def make_packet(candidate_id: str, frame_start: int, hashes: list[str]):
        frames = []
        for offset, (role, _digest) in enumerate(
            zip(("before", "focus", "after"), hashes, strict=True)
        ):
            frame_id = f"F{frame_start + offset:06d}"
            frames.append(
                {
                    "frame_id": frame_id,
                    "path": f"evidence/full/{frame_id}.png",
                    "role": role,
                    "requested_ms": offset * 500,
                    "actual_ms": offset * 500,
                }
            )
        packet = create_vision_packet(
            candidate_id=candidate_id,
            questions=["What changed?"],
            frames=frames,
        )
        atomic_write_json(packet_dir / f"{candidate_id}.json", packet.model_dump(mode="json"))
        return frames, hashes

    identical_frames, identical_hashes = make_packet("V000001", 1, ["same"] * 3)
    changed_frames, changed_hashes = make_packet("V000002", 10, ["a", "b", "c"])
    extra_frames, extra_hashes = make_packet("V000003", 20, ["d", "e", "f"])
    all_frames = identical_frames + changed_frames + extra_frames
    all_hashes = identical_hashes + changed_hashes + extra_hashes
    canonical_frames = [
        {
            "frame_id": frame["frame_id"],
            "metadata": {"analysis": {"semantic_status": "pending"}, "image": {"pixel_hash": {"value": digest}}},
        }
        for frame, digest in zip(all_frames, all_hashes, strict=True)
    ]
    project = {"frames": canonical_frames}
    selected, deferred = _select_semantic_packet_files(
        tmp_path,
        semantic_max_packets=1,
        project=project,
    )

    assert [path.stem for path in selected] == ["V000001", "V000002"]
    assert deferred == ["V000003"]


def test_bounded_semantic_selection_keeps_all_deterministic_scope_aliases(
    tmp_path: Path,
) -> None:
    packet_dir = tmp_path / ".state" / "vision" / "packets"
    packet_dir.mkdir(parents=True)
    shared_frames = [
        {
            "frame_id": "F000001",
            "path": "evidence/full/F000001.png",
            "role": "before",
            "requested_ms": 0,
            "actual_ms": 0,
        },
        {
            "frame_id": "F000002",
            "path": "evidence/full/F000002.png",
            "role": "focus",
            "requested_ms": 500,
            "actual_ms": 500,
        },
    ]
    for candidate_id in ("V000001", "V000002"):
        packet = create_vision_packet(
            candidate_id=candidate_id,
            questions=["What changed?"],
            frames=shared_frames,
        )
        atomic_write_json(packet_dir / f"{candidate_id}.json", packet.model_dump(mode="json"))
    changed_packet = create_vision_packet(
        candidate_id="V000003",
        questions=["What changed?"],
        frames=[
            {**shared_frames[0], "frame_id": "F000003", "path": "evidence/full/F000003.png"},
            {**shared_frames[1], "frame_id": "F000004", "path": "evidence/full/F000004.png"},
        ],
    )
    atomic_write_json(
        packet_dir / "V000003.json", changed_packet.model_dump(mode="json")
    )
    project = {
        "frames": [
            {
                "frame_id": frame_id,
                "metadata": {"image": {"pixel_hash": {"value": digest}}},
            }
            for frame_id, digest in (
                ("F000001", "same"),
                ("F000002", "same"),
                ("F000003", "different-a"),
                ("F000004", "different-b"),
            )
        ]
    }

    selected, deferred = _select_semantic_packet_files(
        tmp_path,
        semantic_max_packets=1,
        project=project,
    )

    assert [path.stem for path in selected] == ["V000001", "V000002", "V000003"]
    assert deferred == []


def test_deterministic_only_selection_excludes_nonidentical_packets(tmp_path: Path) -> None:
    packet_dir = tmp_path / ".state" / "vision" / "packets"
    packet_dir.mkdir(parents=True)

    def write_packet(candidate_id: str, frame_start: int, hashes: list[str]) -> None:
        frames = []
        for offset, role in enumerate(("before", "focus", "after")):
            frame_id = f"F{frame_start + offset:06d}"
            frames.append(
                {
                    "frame_id": frame_id,
                    "path": f"evidence/full/{frame_id}.png",
                    "role": role,
                    "requested_ms": offset * 500,
                    "actual_ms": offset * 500,
                }
            )
        packet = create_vision_packet(
            candidate_id=candidate_id,
            questions=["What changed?"],
            frames=frames,
        )
        atomic_write_json(packet_dir / f"{candidate_id}.json", packet.model_dump(mode="json"))
        return frames

    identical = write_packet("V000001", 1, ["same", "same", "same"])
    changed = write_packet("V000002", 10, ["a", "b", "c"])
    project = {
        "frames": [
            {
                "frame_id": frame["frame_id"],
                "metadata": {"image": {"pixel_hash": {"value": digest}}},
            }
            for frame, digest in zip(
                identical + changed,
                ["same", "same", "same", "a", "b", "c"],
                strict=True,
            )
        ]
    }

    selected, deferred = _select_semantic_packet_files(
        tmp_path,
        project=project,
        deterministic_only=True,
    )

    assert [path.stem for path in selected] == ["V000001"]
    assert deferred == []


def test_deterministic_only_selection_keeps_exact_scope_aliases(
    tmp_path: Path,
) -> None:
    packet_dir = tmp_path / ".state" / "vision" / "packets"
    packet_dir.mkdir(parents=True)
    frames = [
        {
            "frame_id": "F000001",
            "path": "evidence/full/F000001.png",
            "role": "before",
            "requested_ms": 0,
            "actual_ms": 0,
        },
        {
            "frame_id": "F000002",
            "path": "evidence/full/F000002.png",
            "role": "focus",
            "requested_ms": 500,
            "actual_ms": 500,
        },
        {
            "frame_id": "F000003",
            "path": "evidence/full/F000003.png",
            "role": "after",
            "requested_ms": 1000,
            "actual_ms": 1000,
        },
    ]
    for candidate_id in ("V000001", "V000002"):
        packet = create_vision_packet(
            candidate_id=candidate_id,
            questions=["What changed?"],
            frames=frames,
        )
        atomic_write_json(
            packet_dir / f"{candidate_id}.json",
            packet.model_dump(mode="json"),
        )
    project = {
        "frames": [
            {
                "frame_id": frame["frame_id"],
                "metadata": {"image": {"pixel_hash": {"value": "same"}}},
            }
            for frame in frames
        ]
    }

    selected, deferred = _select_semantic_packet_files(
        tmp_path,
        semantic_max_packets=1,
        project=project,
        deterministic_only=True,
    )

    assert [path.stem for path in selected] == ["V000001", "V000002"]
    assert deferred == []


def test_parallel_annotation_uses_deterministic_no_change_before_provider(
    tmp_path: Path,
) -> None:
    class FailingProvider(VisionProvider):
        @property
        def descriptor(self) -> ProviderDescriptor:
            return ProviderDescriptor(
                provider_id="fixture-vlm",
                route="local",
                model="fixture",
                model_version="1",
                adapter_version="1.0",
                network_required=False,
            )

        def annotate(self, packet, *, project_root):  # type: ignore[no-untyped-def]
            raise AssertionError("pixel-identical packets must not invoke the provider")

    packet = create_vision_packet(
        candidate_id="V000001",
        questions=["What changed?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "before",
                "requested_ms": 0,
                "actual_ms": 0,
            },
            {
                "frame_id": "F000002",
                "path": "evidence/full/F000002.png",
                "role": "focus",
                "requested_ms": 500,
                "actual_ms": 500,
            },
        ],
    )
    project = {
        "frames": [
            {
                "frame_id": frame_id,
                "metadata": {"image": {"pixel_hash": {"value": "same"}}},
            }
            for frame_id in ("F000001", "F000002")
        ]
    }

    prepared = _annotate_parallel_packet(
        packet,
        image_id="F000002",
        frame={"frame_id": "F000002"},
        observation_id="VA000001",
        project_dir=tmp_path,
        project=project,
        provider=FailingProvider(),
        semantic_cache_root=None,
        semantic_cache_limit=0,
    )

    assert prepared.deterministic_no_change is True
    assert prepared.annotation.event_type == "no_change"
    assert prepared.provider_failure is None


def test_semantic_budget_keeps_same_focus_packets_with_different_after_frames(
    tmp_path: Path,
) -> None:
    packet_dir = tmp_path / ".state" / "vision" / "packets"
    packet_dir.mkdir(parents=True)

    def write_packet(candidate_id: str, after_id: str) -> None:
        packet = create_vision_packet(
            candidate_id=candidate_id,
            questions=["Does the before/focus/after sequence show a consequential state change?"],
            frames=[
                {
                    "frame_id": "F000001",
                    "path": "evidence/full/F000001.png",
                    "role": "focus",
                    "requested_ms": 0,
                    "actual_ms": 0,
                },
                {
                    "frame_id": after_id,
                    "path": f"evidence/full/{after_id}.png",
                    "role": "after",
                    "requested_ms": 1000,
                    "actual_ms": 1000,
                },
            ],
        )
        atomic_write_json(packet_dir / f"{candidate_id}.json", packet.model_dump(mode="json"))

    write_packet("V000001", "F000002")
    write_packet("V000002", "F000003")
    canonical_frames = [
        {
            "frame_id": frame_id,
            "metadata": {
                "analysis": {
                    # The shared focus was observed by another packet.  The
                    # event-level markers below remain pending, so both
                    # distinct after-frame scopes must still be selectable.
                    "semantic_status": "observed" if frame_id == "F000001" else "pending"
                }
            },
        }
        for frame_id in ("F000001", "F000002", "F000003")
    ]

    selected, deferred = _select_semantic_packet_files(
        tmp_path,
        semantic_max_packets=2,
        project={
            "frames": canonical_frames,
            "visual_events": [
                {"event_id": "V000001", "annotation_provider": None},
                {"event_id": "V000002", "annotation_provider": None},
            ],
        },
    )

    assert [path.stem for path in selected] == ["V000001", "V000002"]
    assert deferred == []


def test_semantic_packet_selection_reuses_validated_packet_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet_dir = tmp_path / ".state" / "vision" / "packets"
    packet_dir.mkdir(parents=True)
    packet_path = packet_dir / "V000001.json"
    packet = create_vision_packet(
        candidate_id="V000001",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    )
    atomic_write_json(packet_path, packet.model_dump(mode="json"))

    original_load_packet = semantic_pipeline._load_packet
    loaded_paths: list[Path] = []

    def counted_load(path: Path):
        loaded_paths.append(path)
        return original_load_packet(path)

    monkeypatch.setattr(semantic_pipeline, "_load_packet", counted_load)
    selected, deferred = _select_semantic_packet_files(
        tmp_path,
        project={"frames": [{"frame_id": "F000001"}]},
    )

    assert selected == [packet_path]
    assert deferred == []
    assert loaded_paths == [packet_path]


def test_filtered_semantic_selection_loads_only_requested_packet_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet_dir = tmp_path / ".state" / "vision" / "packets"
    packet_dir.mkdir(parents=True)
    loaded_paths: list[Path] = []
    original_load_packet = semantic_pipeline._load_packet

    for candidate_id in ("V000001", "V000002", "V000003"):
        packet = create_vision_packet(
            candidate_id=candidate_id,
            questions=["What is visible?"],
            frames=[
                {
                    "frame_id": "F000001",
                    "path": "evidence/full/F000001.png",
                    "role": "focus",
                    "requested_ms": 0,
                    "actual_ms": 0,
                }
            ],
        )
        atomic_write_json(
            packet_dir / f"{candidate_id}.json", packet.model_dump(mode="json")
        )

    def counted_load(path: Path):
        loaded_paths.append(path)
        return original_load_packet(path)

    monkeypatch.setattr(semantic_pipeline, "_load_packet", counted_load)
    selected, deferred = _select_semantic_packet_files(
        tmp_path,
        project={"frames": [{"frame_id": "F000001"}]},
        candidate_ids={"V000002"},
    )

    assert [path.name for path in selected] == ["V000002.json"]
    assert deferred == []
    assert loaded_paths == [packet_dir / "V000002.json"]


def test_semantic_budget_selects_all_exact_packet_aliases_for_one_scope(
    tmp_path: Path,
) -> None:
    packet_dir = tmp_path / ".state" / "vision" / "packets"
    packet_dir.mkdir(parents=True)

    def write_packet(candidate_id: str) -> None:
        packet = create_vision_packet(
            candidate_id=candidate_id,
            questions=["What meaningful visible state is needed to understand this block?"],
            frames=[
                {
                    "frame_id": "F000010",
                    "path": "evidence/full/F000010.png",
                    "role": "focus",
                    "requested_ms": 0,
                    "actual_ms": 0,
                },
                {
                    "frame_id": "F000011",
                    "path": "evidence/full/F000011.png",
                    "role": "after",
                    "requested_ms": 1000,
                    "actual_ms": 1000,
                },
            ],
        )
        atomic_write_json(packet_dir / f"{candidate_id}.json", packet.model_dump(mode="json"))

    write_packet("V000010")
    write_packet("V000011")
    canonical_frames = [
        {
            "frame_id": frame_id,
            "metadata": {"analysis": {"semantic_status": "pending"}},
        }
        for frame_id in ("F000010", "F000011")
    ]

    selected, deferred = _select_semantic_packet_files(
        tmp_path,
        semantic_max_packets=1,
        project={"frames": canonical_frames},
    )

    assert [path.stem for path in selected] == ["V000010", "V000011"]
    assert deferred == []


def test_observed_candidate_re_review_is_explicitly_opt_in(tmp_path: Path) -> None:
    packet_dir = tmp_path / ".state" / "vision" / "packets"
    packet_dir.mkdir(parents=True)
    packet = create_vision_packet(
        candidate_id="V000001",
        questions=["Re-check the historical visual claim."],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    )
    atomic_write_json(packet_dir / "V000001.json", packet.model_dump(mode="json"))
    project = {
        "frames": [{"frame_id": "F000001"}],
        "visual_events": [
            {"event_id": "V000001", "annotation_provider": "llama.cpp-local"}
        ],
    }
    normal, normal_deferred = _select_semantic_packet_files(
        tmp_path,
        project=project,
        candidate_ids={"V000001"},
    )
    assert normal == []
    assert normal_deferred == []
    selected, deferred = _select_semantic_packet_files(
        tmp_path,
        project=project,
        candidate_ids={"V000001"},
        allow_observed_candidate_ids=True,
    )
    assert [path.stem for path in selected] == ["V000001"]
    assert deferred == []


def test_persistent_content_cache_remaps_and_validates_exact_visual_reuse(tmp_path: Path) -> None:
    source = create_vision_packet(
        candidate_id="V000001",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            },
            {
                "frame_id": "F000002",
                "path": "evidence/full/F000002.png",
                "role": "after",
                "requested_ms": 1000,
                "actual_ms": 1000,
            },
        ],
    )
    target = create_vision_packet(
        candidate_id="V000002",
        questions=["What is visible?"],
        frames=[
            {
                "frame_id": "F000101",
                "path": "evidence/full/F000101.png",
                "role": "focus",
                "requested_ms": 2000,
                "actual_ms": 2000,
            },
            {
                "frame_id": "F000102",
                "path": "evidence/full/F000102.png",
                "role": "after",
                "requested_ms": 3000,
                "actual_ms": 3000,
            },
        ],
    )
    frame_index = {
        "F000001": {"pixel_hash": {"value": "same-a"}},
        "F000002": {"pixel_hash": {"value": "same-b"}},
        "F000101": {"pixel_hash": {"value": "same-a"}},
        "F000102": {"pixel_hash": {"value": "same-b"}},
    }
    content = _semantic_visual_content_key(source, frame_index)
    assert content is not None
    provider = type(
        "FakeLocalProvider",
        (),
        {
            "descriptor": ProviderDescriptor(
                provider_id="fixture-local",
                route="local",
                model="fixture",
                model_version="1",
                adapter_version="1",
                network_required=False,
            ),
            "prompt_template_hash": "prompt-v1",
        },
    )()
    annotation = VisionAnnotation(
        candidate_id=source.candidate_id,
        factual_visible_description="A visible title card.",
        event_type="visible_state",
        evidence_frame_ids=["F000001", "F000002"],
        before_action_after_roles={"F000001": "before", "F000002": "after"},
        confidence=0.9,
        uncertainty=[],
        statements_not_inferred=["Identity is not inferred."],
    )
    key = _semantic_content_cache_key(provider, content[0])
    path = tmp_path / "content-cache.json"
    assert _write_semantic_content_cache(
        path,
        key=key,
        provider=provider,
        packet=source,
        annotation=annotation,
        source_frame_ids=content[1],
        max_bytes=1024 * 1024,
    )
    reused = _read_semantic_content_cache(
        path,
        key=key,
        packet=target,
        source_frame_ids=content[1],
        max_bytes=1024 * 1024,
    )
    assert reused is not None
    assert reused.candidate_id == target.candidate_id
    assert reused.evidence_frame_ids == ["F000101", "F000102"]
    assert dict(reused.before_action_after_roles) == {
        "F000101": "before",
        "F000102": "after",
    }
    cache_text = path.read_text(encoding="utf-8")
    assert cache_text.endswith("\n")
    assert "\n  " not in cache_text


def test_semantic_budget_review_frontier_is_consolidated_without_touching_decisions() -> None:
    decided = {
        "review_id": "R000003",
        "category": "semantic_budget_deferred",
        "event_ids": ["V000001"],
        "decision": "human_deferred_for_later_review",
    }
    old = {
        "review_id": "R000004",
        "category": "semantic_budget_deferred",
        "event_ids": ["V000002", "V000003"],
        "decision": None,
    }
    newest = {
        "review_id": "R000005",
        "category": "semantic_budget_deferred",
        "event_ids": ["V000004"],
        "decision": None,
    }
    project = {
        "media": {"duration_ms": 10_000},
        "review_items": [decided, old, newest],
    }

    assert _sync_semantic_budget_review_item(
        project,
        ["V000006", "V000007", "V000006"],
        provider_id="fixture-local",
    )
    live = [
        item
        for item in project["review_items"]
        if item.get("category") == "semantic_budget_deferred" and not item.get("decision")
    ]
    assert len(live) == 1
    assert live[0]["review_id"] == "R000005"
    assert live[0]["event_ids"] == ["V000006", "V000007"]
    assert decided["decision"] == "human_deferred_for_later_review"
    assert old not in project["review_items"]

    assert not _sync_semantic_budget_review_item(
        project,
        ["V000006", "V000007"],
        provider_id="fixture-local",
    )
    assert _sync_semantic_budget_review_item(
        project,
        [],
        provider_id="fixture-local",
    )
    assert decided in project["review_items"]
    assert not any(
        item.get("category") == "semantic_budget_deferred" and not item.get("decision")
        for item in project["review_items"]
    )


def test_semantic_pending_retry_requires_a_prompt_revision_change(tmp_path: Path) -> None:
    annotations = tmp_path / ".state" / "vision" / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "V000001.annotation.json").write_text(
        json.dumps({"event_type": "semantic_pending", "confidence": 0.0}),
        encoding="utf-8",
    )
    (annotations / "V000001.observation.json").write_text(
        json.dumps({"prompt_template_hash": "old-prompt"}),
        encoding="utf-8",
    )
    assert _is_retryable_semantic_pending(
        tmp_path, "V000001", prompt_template_hash="new-prompt"
    )
    assert not _is_retryable_semantic_pending(
        tmp_path, "V000001", prompt_template_hash="old-prompt"
    )
    (annotations / "V000002.annotation.json").write_text(
        json.dumps({"event_type": "semantic_pending", "confidence": 0.9}),
        encoding="utf-8",
    )
    (annotations / "V000002.observation.json").write_text(
        json.dumps({"prompt_template_hash": "old-prompt"}),
        encoding="utf-8",
    )
    assert _is_retryable_semantic_pending(
        tmp_path, "V000002", prompt_template_hash="new-prompt"
    )
    assert _semantic_pending_retry_priority(tmp_path, "V000001") == 1
    (annotations / "V000002.annotation.json").write_text(
        json.dumps({"event_type": "semantic_pending", "confidence": 0.9}),
        encoding="utf-8",
    )
    assert _semantic_pending_retry_priority(tmp_path, "V000002") == 2


def test_retryable_semantic_counts_decode_each_annotation_once(tmp_path: Path, monkeypatch) -> None:
    annotations = tmp_path / ".state" / "vision" / "annotations"
    annotations.mkdir(parents=True)
    for candidate_id, event_type, uncertainty, prompt_hash in (
        ("V000001", "semantic_pending", ["HTTP Error 400"], "old-prompt"),
        ("V000002", "semantic_pending", [], "new-prompt"),
        ("V000003", "visible_state", ["HTTP Error 400"], "old-prompt"),
    ):
        (annotations / f"{candidate_id}.annotation.json").write_text(
            json.dumps({"event_type": event_type, "uncertainty": uncertainty}),
            encoding="utf-8",
        )
        (annotations / f"{candidate_id}.observation.json").write_text(
            json.dumps({"prompt_template_hash": prompt_hash}),
            encoding="utf-8",
        )

    original_read_text = Path.read_text
    read_paths: list[Path] = []

    def counted_read_text(path: Path, *args, **kwargs):
        read_paths.append(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted_read_text)
    assert _retryable_semantic_counts(
        tmp_path,
        prompt_template_hash="new-prompt",
        include_http400=True,
        include_semantic_pending=True,
    ) == (1, 1)

    assert read_paths.count(annotations / "V000001.annotation.json") == 1
    assert read_paths.count(annotations / "V000002.annotation.json") == 1
    assert read_paths.count(annotations / "V000003.annotation.json") == 1
    assert read_paths.count(annotations / "V000001.observation.json") == 1
    assert read_paths.count(annotations / "V000002.observation.json") == 1
    assert (annotations / "V000003.observation.json") not in read_paths


def test_semantic_packet_score_prioritizes_measured_visual_change(tmp_path: Path) -> None:
    del tmp_path
    base = create_vision_packet(
        candidate_id="V000001",
        questions=["What changed?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    )
    changed = base.model_copy(
        update={
            "candidate_id": "V000002",
            "frames": [
                base.frames[0].model_copy(
                    update={
                        "frame_id": "F000002",
                        "difference_regions": [
                            DifferenceRegion(
                                xywh=(0, 0, 10, 10), changed_ratio=0.9
                            )
                        ],
                    }
                )
            ],
        }
    )
    assert _semantic_packet_score(changed) > _semantic_packet_score(base)


def test_semantic_packet_score_keeps_tiny_change_visible_in_bounded_queue() -> None:
    base = create_vision_packet(
        candidate_id="V000001",
        questions=["What changed?"],
        frames=[
            {
                "frame_id": "F000001",
                "path": "evidence/full/F000001.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
    )
    tiny = base.model_copy(
        update={
            "candidate_id": "V000002",
            "frames": [
                base.frames[0].model_copy(
                    update={
                        "frame_id": "F000002",
                        "difference_regions": [
                            DifferenceRegion(xywh=(0, 0, 2, 2), changed_ratio=0.01)
                        ],
                    }
                )
            ],
        }
    )

    assert _semantic_packet_score(tiny) > _semantic_packet_score(base)


def test_semantic_budget_mixes_temporal_anchors_with_high_change_packets(
    tmp_path: Path,
) -> None:
    packet_dir = tmp_path / ".state" / "vision" / "packets"
    packet_dir.mkdir(parents=True)
    canonical_frames: list[dict[str, object]] = []
    for index in range(8):
        candidate_id = f"V{index + 1:06d}"
        frame_id = f"F{index + 1:06d}"
        difference_regions = (
            [DifferenceRegion(xywh=(0, 0, 10, 10), changed_ratio=0.95)]
            if index == 5
            else []
        )
        packet = create_vision_packet(
            candidate_id=candidate_id,
            questions=["What changed?"],
            frames=[
                {
                    "frame_id": frame_id,
                    "path": f"evidence/full/{frame_id}.png",
                    "role": "focus",
                    "requested_ms": index * 1000,
                    "actual_ms": index * 1000,
                    "difference_regions": difference_regions,
                }
            ],
        )
        atomic_write_json(packet_dir / f"{candidate_id}.json", packet.model_dump(mode="json"))
        canonical_frames.append(
            {
                "frame_id": frame_id,
                "metadata": {"analysis": {"semantic_status": "pending"}},
            }
        )

    selected, _deferred = _select_semantic_packet_files(
        tmp_path,
        semantic_max_packets=4,
        project={"frames": canonical_frames},
    )

    selected_ids = {path.stem for path in selected}
    assert "V000006" in selected_ids
    assert len(selected_ids) == 4
