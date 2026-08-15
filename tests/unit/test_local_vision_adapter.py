from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest
from PIL import Image

import video_script_reconstructor.local_vision_adapter as adapter_module
from video_script_reconstructor.errors import SecurityError
from video_script_reconstructor.local_vision_adapter import (
    ADAPTIVE_MAX_IMAGE_EDGE,
    ADAPTIVE_MAX_TOKENS,
    DEFAULT_MAX_IMAGE_EDGE,
    DEFAULT_MAX_TOKENS,
    SINGLE_FRAME_MAX_IMAGE_EDGE,
    SINGLE_FRAME_MAX_TOKENS,
    _adaptive_transport_overrides,
    _clear_image_transport_cache,
    _focus_only_transport_request,
    _image_data_url,
    _image_transport_cache_stats,
    _local_vision_max_image_edge,
    _local_vision_max_tokens,
    _loopback_endpoint,
    _packet_has_textual_context,
    annotate_via_local_server,
    build_llama_request,
    normalize_annotation_evidence,
)
from video_script_reconstructor.vision_packets import VisionAnnotation, create_vision_packet


def test_local_vision_request_embeds_only_contained_images_and_schema(tmp_path: Path) -> None:
    Image.new("RGB", (32, 32), "blue").save(tmp_path / "frame.png")
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "frame.png",
                "role": "context",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What color is visible?"],
    )
    _, request = build_llama_request(
        {
            "packet": packet.model_dump(mode="json"),
            "project_root": str(tmp_path),
            "required_annotation_schema": VisionAnnotation.model_json_schema(),
        },
        model="local-fixture",
    )
    content = request["messages"][0]["content"]
    assert any(item.get("type") == "image_url" for item in content)
    assert request["response_format"]["json_schema"]["strict"] is True
    roles_schema = request["response_format"]["json_schema"]["schema"]["properties"][
        "before_action_after_roles"
    ]
    assert set(roles_schema["properties"]) == {"F000001"}
    assert roles_schema["additionalProperties"] is False
    assert request["temperature"] == 0
    assert request["max_tokens"] == 768
    assert request["chat_template_kwargs"] == {"enable_thinking": False}


def test_local_vision_prompt_compacts_ocr_transport_only(tmp_path: Path) -> None:
    Image.new("RGB", (32, 32), "blue").save(tmp_path / "frame.png")
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What exact number is visible?"],
        raw_ocr=[
            {
                "observation_id": "O000001",
                "frame_id": "F000001",
                "raw_engine_text": "RAW-HOCR-TOKEN " * 2000,
                "normalized_interpretation": "Visible account number: 1451039",
                "confidence": 0.4,
                "bounding_region": [0, 0, 32, 32],
                "uncertain_characters": [
                    {
                        "bounding_box": [1, 2, 3, 4],
                        "confidence": 0.1,
                        "reason": "low confidence",
                        "text": "9",
                    }
                ],
            }
        ],
    )

    _, request = build_llama_request(
        {
            "packet": packet.model_dump(mode="json"),
            "project_root": str(tmp_path),
            "required_annotation_schema": VisionAnnotation.model_json_schema(),
        },
        model="local-fixture",
    )

    prompt = str(request["messages"][0]["content"][0]["text"])
    assert len(prompt) < 20_000
    assert "RAW-HOCR-TOKEN" not in prompt
    assert "bounding_box" not in prompt
    assert "bounding_region" not in prompt
    assert "uncertain_character_count" in prompt
    assert "Visible account number: 1451039" in prompt
    # The returned packet remains the complete, unprojected canonical object.
    assert packet.raw_ocr[0].raw_engine_text.startswith("RAW-HOCR-TOKEN")
    assert packet.raw_ocr[0].uncertain_characters[0]["bounding_box"] == [1, 2, 3, 4]


def test_local_vision_prompt_bounds_transcript_linked_ocr_geometry(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (32, 32), "blue").save(tmp_path / "frame.png")
    geometry_rows = "\n".join(
        f"5 1 5 1 1 {index} 336 112 4 24 93.073463 Freight {index}"
        for index in range(1, 2_000)
    )
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What exact text is visible?"],
        nearby_transcript=[
            {
                "frame_ids": ["F000001"],
                "on_screen_text": ["Freight Courses 101", geometry_rows],
            }
        ],
    )
    original = packet.model_dump(mode="json")

    _, request = build_llama_request(
        {
            "packet": original,
            "project_root": str(tmp_path),
            "required_annotation_schema": VisionAnnotation.model_json_schema(),
        },
        model="local-fixture",
    )

    prompt = str(request["messages"][0]["content"][0]["text"])
    assert "Freight Courses 101" in prompt
    assert "5 1 5 1 1 1 336 112 4 24 93.073463" not in prompt
    assert "on_screen_text_transport_truncated" in prompt
    assert len(prompt) < 30_000
    # Prompt projection is transport-only; canonical packet data is unchanged.
    assert packet.model_dump(mode="json") == original


def test_local_vision_response_budget_is_bounded_and_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _local_vision_max_tokens() == 768
    monkeypatch.setenv("VSR_LOCAL_VISION_MAX_TOKENS", "128")
    assert _local_vision_max_tokens() == 256
    monkeypatch.setenv("VSR_LOCAL_VISION_MAX_TOKENS", "99999")
    assert _local_vision_max_tokens() == 4096


def test_local_vision_adapts_only_no_ocr_transport_and_respects_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Image.new("RGB", (32, 32), "blue").save(tmp_path / "frame.png")
    no_ocr_packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What is visible?"],
    )
    ocr_packet = create_vision_packet(
        candidate_id="V000002",
        frames=[
            {
                "frame_id": "F000001",
                "path": "frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What exact text is visible?"],
        raw_ocr=[
            {
                "observation_id": "O000001",
                "frame_id": "F000001",
                "raw_engine_text": "Visible text",
                "normalized_interpretation": "Visible text",
                "confidence": 0.9,
                "bounding_region": [0, 0, 32, 32],
                "uncertain_characters": [],
            }
        ],
    )

    assert _adaptive_transport_overrides(
        no_ocr_packet, image_edge=None, max_tokens=None
    ) == (SINGLE_FRAME_MAX_IMAGE_EDGE, SINGLE_FRAME_MAX_TOKENS)
    assert _adaptive_transport_overrides(
        ocr_packet, image_edge=None, max_tokens=None
    ) == (None, None)
    monkeypatch.setenv("VSR_LOCAL_VISION_MAX_IMAGE_EDGE", "1536")
    assert _adaptive_transport_overrides(
        no_ocr_packet, image_edge=None, max_tokens=None
    ) == (None, None)
    assert _adaptive_transport_overrides(
        no_ocr_packet, image_edge=900, max_tokens=300
    ) == (900, 300)

    multi_frame_packet = no_ocr_packet.model_copy(
        update={
            "frames": [
                *no_ocr_packet.frames,
                no_ocr_packet.frames[0].model_copy(
                    update={"frame_id": "F000002", "actual_ms": 1000, "requested_ms": 1000}
                ),
            ]
        }
    )
    monkeypatch.delenv("VSR_LOCAL_VISION_MAX_IMAGE_EDGE", raising=False)
    assert _adaptive_transport_overrides(
        multi_frame_packet, image_edge=None, max_tokens=None
    ) == (ADAPTIVE_MAX_IMAGE_EDGE, ADAPTIVE_MAX_TOKENS)


def test_local_vision_keeps_full_transport_when_neighbor_context_has_text(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (32, 32), "blue").save(tmp_path / "frame.png")
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What is visible?"],
        nearby_transcript=[{"on_screen_text": ["Freight Courses 101"]}],
    )

    assert _packet_has_textual_context(packet) is True
    assert _adaptive_transport_overrides(packet, image_edge=None, max_tokens=None) == (
        None,
        None,
    )

    unrelated = packet.model_copy(
        update={"nearby_transcript": [{"frame_ids": ["F000999"], "on_screen_text": ["old"]}]}
    )
    assert _packet_has_textual_context(unrelated) is False
    assert _adaptive_transport_overrides(unrelated, image_edge=None, max_tokens=None) == (
        SINGLE_FRAME_MAX_IMAGE_EDGE,
        SINGLE_FRAME_MAX_TOKENS,
    )


def test_local_vision_empty_ocr_marker_does_not_force_full_transport(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (32, 32), "blue").save(tmp_path / "frame.png")
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What is visible?"],
        raw_ocr=[
            {
                "observation_id": "O000001",
                "frame_id": "F000001",
                "raw_engine_text": "",
                "normalized_interpretation": "",
                "confidence": None,
                "bounding_region": None,
                "uncertain_characters": [],
            }
        ],
    )

    assert _packet_has_textual_context(packet) is False
    assert _adaptive_transport_overrides(packet, image_edge=None, max_tokens=None) == (
        SINGLE_FRAME_MAX_IMAGE_EDGE,
        SINGLE_FRAME_MAX_TOKENS,
    )


def test_local_vision_adaptive_request_uses_middle_profile_for_no_ocr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Image.new("RGB", (2400, 1200), "blue").save(tmp_path / "frame.png")
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What is visible?"],
    )
    request_data = {
        "packet": packet.model_dump(mode="json"),
        "project_root": str(tmp_path),
        "required_annotation_schema": {},
    }
    annotation = {
        "candidate_id": "V000001",
        "factual_visible_description": "A blue field is visible.",
        "event_type": "visible_state",
        "evidence_frame_ids": ["F000001"],
        "before_action_after_roles": {"F000001": "action"},
        "exact_visible_text_candidates": [],
        "consequential_changes": [],
        "confidence": 0.8,
        "uncertainty": [],
        "statements_not_inferred": ["No hidden state is inferred."],
    }
    calls: list[dict[str, object]] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(annotation)}}]}
            ).encode("utf-8")

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        del timeout
        data = request.data
        assert isinstance(data, bytes)
        calls.append(json.loads(data.decode("utf-8")))
        return Response()

    monkeypatch.delenv("VSR_LOCAL_VISION_MAX_IMAGE_EDGE", raising=False)
    monkeypatch.delenv("VSR_LOCAL_VISION_MAX_TOKENS", raising=False)
    monkeypatch.setattr(adapter_module, "urlopen", fake_urlopen)

    result = annotate_via_local_server(request_data, model="local-fixture")

    assert result.event_type == "visible_state"
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == SINGLE_FRAME_MAX_TOKENS
    image_item = next(
        item
        for item in calls[0]["messages"][0]["content"]
        if item.get("type") == "image_url"
    )
    encoded = str(image_item["image_url"]["url"]).split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as transported:
        assert max(transported.size) == SINGLE_FRAME_MAX_IMAGE_EDGE


def test_local_vision_escalates_compact_semantic_pending_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Image.new("RGB", (2400, 1200), "blue").save(tmp_path / "frame.png")
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What is visible?"],
    )
    provider_request = {
        "packet": packet.model_dump(mode="json"),
        "project_root": str(tmp_path),
        "required_annotation_schema": {},
    }
    pending = {
        "candidate_id": "V000001",
        "factual_visible_description": "The supplied pixels do not support a defensible visible fact.",
        "event_type": "semantic_pending",
        "evidence_frame_ids": ["F000001"],
        "before_action_after_roles": {"F000001": "action"},
        "exact_visible_text_candidates": [],
        "consequential_changes": [],
        "confidence": 0.0,
        "uncertainty": ["The compact transport was insufficient for a visible-fact decision."],
        "statements_not_inferred": ["No hidden state or cause is inferred."],
    }
    full = {
        **pending,
        "factual_visible_description": "A blue field is visible.",
        "event_type": "visible_state",
        "confidence": 0.8,
        "uncertainty": [],
    }
    calls: list[dict[str, object]] = []

    class Response:
        def __init__(self, annotation: dict[str, object]) -> None:
            self.annotation = annotation

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(self.annotation)}}]}
            ).encode("utf-8")

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        del timeout
        data = request.data
        assert isinstance(data, bytes)
        calls.append(json.loads(data.decode("utf-8")))
        return Response(pending if len(calls) == 1 else full)

    monkeypatch.delenv("VSR_LOCAL_VISION_MAX_IMAGE_EDGE", raising=False)
    monkeypatch.delenv("VSR_LOCAL_VISION_MAX_TOKENS", raising=False)
    monkeypatch.setattr(adapter_module, "urlopen", fake_urlopen)

    result = annotate_via_local_server(provider_request, model="local-fixture")

    assert result.event_type == "visible_state"
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == SINGLE_FRAME_MAX_TOKENS
    assert calls[1]["max_tokens"] == DEFAULT_MAX_TOKENS
    first_image = next(
        item
        for item in calls[0]["messages"][0]["content"]
        if item.get("type") == "image_url"
    )
    second_image = next(
        item
        for item in calls[1]["messages"][0]["content"]
        if item.get("type") == "image_url"
    )
    with Image.open(
        BytesIO(base64.b64decode(str(first_image["image_url"]["url"]).split(",", 1)[1]))
    ) as compact:
        with Image.open(
            BytesIO(base64.b64decode(str(second_image["image_url"]["url"]).split(",", 1)[1]))
        ) as full_transport:
            assert max(compact.size) == SINGLE_FRAME_MAX_IMAGE_EDGE
            assert max(full_transport.size) == DEFAULT_MAX_IMAGE_EDGE


def test_local_vision_transport_resize_does_not_change_source_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_image_transport_cache()
    source = tmp_path / "wide.png"
    Image.new("RGB", (2400, 1200), "blue").save(source)
    original = source.read_bytes()
    monkeypatch.setenv("VSR_LOCAL_VISION_MAX_IMAGE_EDGE", "1200")

    data_url = _image_data_url(source)
    assert _image_data_url(source) == data_url
    stats = _image_transport_cache_stats()
    assert stats["miss_count"] == 1
    assert stats["hit_count"] == 1
    assert stats["entry_count"] == 1
    encoded = data_url.split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as transported:
        assert max(transported.size) == 1200
    assert source.read_bytes() == original
    assert _local_vision_max_image_edge() == 1200


def test_local_vision_transport_cache_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_image_transport_cache()
    monkeypatch.setenv("VSR_LOCAL_VISION_TRANSPORT_CACHE_MAX_BYTES", "0")
    source = tmp_path / "small.png"
    Image.new("RGB", (32, 32), "green").save(source)
    first = _image_data_url(source)
    second = _image_data_url(source)
    assert first == second
    stats = _image_transport_cache_stats()
    assert stats["hit_count"] == 0
    assert stats["miss_count"] == 0
    assert stats["entry_count"] == 0


def test_local_vision_transport_cache_reuses_canonical_identity_across_paths(
    tmp_path: Path,
) -> None:
    """Decoded-pixel identity reuse avoids re-encoding copied evidence files."""

    _clear_image_transport_cache()
    first = tmp_path / "project-a" / "frame.png"
    second = tmp_path / "project-b" / "frame.png"
    first.parent.mkdir()
    second.parent.mkdir()
    Image.new("RGB", (1800, 900), "purple").save(first)
    second.write_bytes(first.read_bytes())
    identity = "sha256-rgba8-srgb-v1:fixture-pixel-digest"

    first_url = _image_data_url(first, max_edge=1024, cache_identity=identity)
    second_url = _image_data_url(second, max_edge=1024, cache_identity=identity)

    assert second_url == first_url
    stats = _image_transport_cache_stats()
    assert stats["miss_count"] == 1
    assert stats["hit_count"] == 1
    assert stats["identity_hit_count"] == 1
    assert stats["path_hit_count"] == 0


def test_local_vision_retries_http_400_with_smaller_transport_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "wide.png"
    Image.new("RGB", (2400, 1200), "blue").save(source)
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "wide.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What is visible?"],
    )
    provider_request = {
        "packet": packet.model_dump(mode="json"),
        "project_root": str(tmp_path),
        "required_annotation_schema": {},
    }
    annotation = {
        "candidate_id": "V000001",
        "factual_visible_description": "A blue field is visible.",
        "event_type": "visible_state",
        "evidence_frame_ids": ["F000001"],
        "before_action_after_roles": {"F000001": "action"},
        "exact_visible_text_candidates": [],
        "consequential_changes": [],
        "confidence": 0.8,
        "uncertainty": [],
        "statements_not_inferred": ["No hidden state is inferred."],
    }
    calls: list[dict[str, object]] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(annotation)}}]}
            ).encode("utf-8")

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        del timeout
        data = request.data
        assert isinstance(data, bytes)
        calls.append(json.loads(data.decode("utf-8")))
        if len(calls) <= 2:
            raise HTTPError("http://127.0.0.1:8187", 400, "Bad Request", {}, None)
        return Response()

    monkeypatch.setenv("VSR_LOCAL_VISION_MAX_IMAGE_EDGE", "1600")
    monkeypatch.setenv("VSR_LOCAL_VISION_MAX_TOKENS", "1536")
    monkeypatch.setattr(adapter_module, "urlopen", fake_urlopen)

    result = annotate_via_local_server(provider_request, model="local-fixture")

    assert result.candidate_id == "V000001"
    assert len(calls) == 3
    assert calls[0]["max_tokens"] == 1536
    assert calls[1]["max_tokens"] == 768
    assert calls[2]["max_tokens"] == 256
    first_url = calls[0]["messages"][0]["content"][2]["image_url"]["url"]
    second_url = calls[1]["messages"][0]["content"][2]["image_url"]["url"]
    third_url = calls[2]["messages"][0]["content"][2]["image_url"]["url"]
    with Image.open(BytesIO(base64.b64decode(str(first_url).split(",", 1)[1]))) as first:
        with Image.open(BytesIO(base64.b64decode(str(second_url).split(",", 1)[1]))) as second:
            with Image.open(BytesIO(base64.b64decode(str(third_url).split(",", 1)[1]))) as third:
                assert max(first.size) == 1600
                assert max(second.size) == 800
                assert max(third.size) == 768


def test_local_vision_http_400_drops_supplemental_crop_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name, color in (
        ("full-before.png", "blue"),
        ("crop-before.png", "cyan"),
        ("full-after.png", "green"),
        ("crop-after.png", "lime"),
    ):
        Image.new("RGB", (1200, 800), color).save(tmp_path / name)
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "full-before.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            },
            {
                "frame_id": "F000001-C01",
                "path": "crop-before.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            },
            {
                "frame_id": "F000002",
                "path": "full-after.png",
                "role": "after",
                "requested_ms": 1000,
                "actual_ms": 1000,
            },
            {
                "frame_id": "F000002-C01",
                "path": "crop-after.png",
                "role": "focus",
                "requested_ms": 1000,
                "actual_ms": 1000,
            },
        ],
        questions=["What changed?"],
    )
    provider_request = {
        "packet": packet.model_dump(mode="json"),
        "project_root": str(tmp_path),
        "required_annotation_schema": {},
    }
    annotation = {
        "candidate_id": "V000001",
        "factual_visible_description": "The visible state changes from blue to green.",
        "event_type": "visible_state_change",
        "evidence_frame_ids": ["F000001", "F000002"],
        "before_action_after_roles": {"F000001": "action", "F000002": "after"},
        "exact_visible_text_candidates": [],
        "consequential_changes": [],
        "confidence": 0.8,
        "uncertainty": [],
        "statements_not_inferred": ["No cause is inferred."],
    }
    calls: list[dict[str, object]] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(annotation)}}]}
            ).encode("utf-8")

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        del timeout
        data = request.data
        assert isinstance(data, bytes)
        calls.append(json.loads(data.decode("utf-8")))
        if len(calls) <= 3:
            raise HTTPError("http://127.0.0.1:8187", 400, "Bad Request", {}, None)
        return Response()

    monkeypatch.setattr(adapter_module, "urlopen", fake_urlopen)
    result = annotate_via_local_server(provider_request, model="local-fixture")

    assert result.evidence_frame_ids == ["F000001", "F000002"]
    assert any("omitted supplemental frame IDs" in item for item in result.uncertainty)
    assert len(calls) == 4
    image_counts = [
        sum(1 for item in call["messages"][0]["content"] if item.get("type") == "image_url")
        for call in calls
    ]
    assert image_counts == [4, 4, 2, 1]


def test_local_vision_retries_one_malformed_response_with_correction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "frame.png"
    Image.new("RGB", (32, 32), "blue").save(source)
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "frame.png",
                "role": "focus",
                "requested_ms": 0,
                "actual_ms": 0,
            }
        ],
        questions=["What is visible?"],
    )
    request_data = {
        "packet": packet.model_dump(mode="json"),
        "project_root": str(tmp_path),
        "required_annotation_schema": {},
    }
    annotation = {
        "candidate_id": "V000001",
        "factual_visible_description": "A blue field is visible.",
        "event_type": "visible_state",
        "evidence_frame_ids": ["F000001"],
        "before_action_after_roles": {"F000001": "action"},
        "exact_visible_text_candidates": [],
        "consequential_changes": [],
        "confidence": 0.8,
        "uncertainty": [],
        "statements_not_inferred": ["No hidden state is inferred."],
    }
    calls = 0

    class Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return self.body

    def fake_urlopen(_request: object, *, timeout: float) -> Response:
        nonlocal calls
        del timeout
        calls += 1
        if calls == 1:
            return Response(b"not-json")
        body = json.dumps(
            {"choices": [{"message": {"content": json.dumps(annotation)}}]}
        ).encode("utf-8")
        return Response(body)

    monkeypatch.setattr(adapter_module, "urlopen", fake_urlopen)
    result = annotate_via_local_server(request_data, model="local-fixture")

    assert result.candidate_id == "V000001"
    assert calls == 2


def test_local_vision_retries_missing_focus_citation_with_explicit_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name, color in (("before.png", "blue"), ("focus.png", "green"), ("after.png", "red")):
        Image.new("RGB", (32, 32), color).save(tmp_path / name)
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "before.png",
                "role": "before",
                "requested_ms": 0,
                "actual_ms": 0,
            },
            {
                "frame_id": "F000002",
                "path": "focus.png",
                "role": "focus",
                "requested_ms": 500,
                "actual_ms": 500,
            },
            {
                "frame_id": "F000003",
                "path": "after.png",
                "role": "after",
                "requested_ms": 1000,
                "actual_ms": 1000,
            },
        ],
        questions=["What changed?"],
    )
    provider_request = {
        "packet": packet.model_dump(mode="json"),
        "project_root": str(tmp_path),
        "required_annotation_schema": {},
    }

    def annotation(evidence_frame_id: str, role: str) -> dict[str, object]:
        return {
            "candidate_id": "V000001",
            "factual_visible_description": "The visible color changes across the sequence.",
            "event_type": "visible_state_change",
            "evidence_frame_ids": [evidence_frame_id],
            "before_action_after_roles": {evidence_frame_id: role},
            "exact_visible_text_candidates": [],
            "consequential_changes": [],
            "confidence": 0.8,
            "uncertainty": [],
            "statements_not_inferred": ["No cause is inferred."],
        }

    invalid = annotation("F000001", "before")
    valid = annotation("F000002", "action")
    calls: list[dict[str, object]] = []

    class Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return self.body

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        del timeout
        data = request.data
        assert isinstance(data, bytes)
        calls.append(json.loads(data.decode("utf-8")))
        payload = invalid if len(calls) <= 2 else valid
        return Response(
            json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]}).encode(
                "utf-8"
            )
        )

    monkeypatch.setattr(adapter_module, "urlopen", fake_urlopen)
    result = annotate_via_local_server(provider_request, model="local-fixture")

    assert result.evidence_frame_ids == ["F000002"]
    assert len(calls) == 3
    retry_text = str(calls[1]["messages"][0]["content"][-1]["text"])
    assert "focus/action/result IDs" in retry_text
    assert "F000002" in retry_text
    assert any("Citation recovery omitted supplemental frame IDs" in item for item in result.uncertainty)


def test_local_vision_endpoint_rejects_remote_or_credentialed_urls() -> None:
    assert _loopback_endpoint("http://127.0.0.1:8187/v1/chat/completions")
    with pytest.raises(SecurityError, match="loopback"):
        _loopback_endpoint("https://example.com/v1/chat/completions")
    with pytest.raises(SecurityError, match="credentials"):
        _loopback_endpoint("http://user:secret@localhost:8187/v1/chat/completions")


def test_local_vision_promotes_already_cited_role_frames_into_evidence(tmp_path: Path) -> None:
    Image.new("RGB", (32, 32), "blue").save(tmp_path / "before.png")
    Image.new("RGB", (32, 32), "green").save(tmp_path / "after.png")
    packet = create_vision_packet(
        candidate_id="V000001",
        frames=[
            {
                "frame_id": "F000001",
                "path": "before.png",
                "role": "before",
                "requested_ms": 0,
                "actual_ms": 0,
            },
            {
                "frame_id": "F000002",
                "path": "after.png",
                "role": "result",
                "requested_ms": 1000,
                "actual_ms": 1000,
            },
        ],
        questions=["What changed?"],
    )
    annotation = {
        "candidate_id": "V000001",
        "factual_visible_description": "The color changes from blue to green.",
        "event_type": "visible_state_change",
        "evidence_frame_ids": ["F000002"],
        "before_action_after_roles": {"F000001": "before", "F000002": "after"},
        "exact_visible_text_candidates": [],
        "consequential_changes": [],
        "confidence": 0.8,
        "uncertainty": [],
        "statements_not_inferred": ["No cause is inferred."],
    }
    normalized = normalize_annotation_evidence(annotation, packet)

    assert normalized["evidence_frame_ids"] == ["F000002", "F000001"]
    assert "normalized evidence_frame_ids" in normalized["uncertainty"][0]


def test_local_vision_focus_only_transport_recovery_preserves_primary_frames() -> None:
    request = {
        "packet": {
            "candidate_id": "V000001",
            "frames": [
                {
                    "frame_id": "F000001",
                    "path": "before.png",
                    "role": "before",
                    "requested_ms": 0,
                    "actual_ms": 0,
                },
                {
                    "frame_id": "F000002",
                    "path": "focus.png",
                    "role": "focus",
                    "requested_ms": 500,
                    "actual_ms": 500,
                },
            ],
            "questions": ["What changed?"],
        }
    }
    reduced = _focus_only_transport_request(request)
    assert reduced is not None
    variant, removed = reduced
    assert [frame["frame_id"] for frame in variant["packet"]["frames"]] == ["F000002"]
    assert removed == ("F000001",)
