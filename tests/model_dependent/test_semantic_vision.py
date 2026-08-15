from __future__ import annotations

import os
import shlex
from pathlib import Path

import pytest
from PIL import Image

from video_script_reconstructor.model_store import verify_model
from video_script_reconstructor.providers.llama_cpp import LlamaCppVisionProvider
from video_script_reconstructor.providers.local import LocalCommandVisionProvider
from video_script_reconstructor.vision_packets import create_vision_packet

pytestmark = [pytest.mark.model_dependent, pytest.mark.semantic_vision]


def test_configured_local_semantic_vision_backend(tmp_path: Path) -> None:
    command_value = os.environ.get("VSR_LOCAL_VISION_COMMAND")
    installed = verify_model("qwen3-vl-4b-q4")
    if not command_value and not installed.get("offline_ready"):
        pytest.skip(
            "semantic vision backend unavailable: set VSR_LOCAL_VISION_COMMAND to a real "
            "local adapter executable"
        )
    if command_value:
        command = shlex.split(command_value, posix=os.name != "nt")
        provider = LocalCommandVisionProvider(
            command,
            provider_id="model-dependent-local-smoke",
            model=os.environ.get("VSR_LOCAL_VISION_MODEL", "configured-local-model"),
        )
        if not provider.available():
            pytest.skip(f"semantic vision backend unavailable: executable not found: {command[0]}")
    else:
        provider = LlamaCppVisionProvider()
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (64, 64), "blue").save(image_path)
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
        questions=["What color is directly visible?"],
    )
    try:
        annotation = provider.annotate(packet, project_root=tmp_path)
        assert annotation.candidate_id == packet.candidate_id
        assert annotation.evidence_frame_ids == ["F000001"]
        assert annotation.factual_visible_description.strip()
        assert 0.0 <= annotation.confidence <= 1.0
    finally:
        if isinstance(provider, LlamaCppVisionProvider):
            provider.close()
