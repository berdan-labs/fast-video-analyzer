from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from ..errors import BlockedError, InputError, ValidationFailure
from ..security import safe_relative_path
from ..vision_packets import VisionAnnotation, VisionPacket, validate_annotation_for_packet
from .base import ProviderDescriptor, VisionProvider


class LocalCommandVisionProvider(VisionProvider):
    """Invoke a configured local multimodal adapter through JSON stdin/stdout."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        provider_id: str,
        model: str,
        model_version: str | None = None,
        adapter_version: str = "1.0",
        timeout_seconds: float = 300.0,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if not command or not all(isinstance(item, str) and item for item in command):
            raise InputError("Local vision command must be a non-empty argument sequence")
        if timeout_seconds <= 0 or max_response_bytes <= 0:
            raise InputError("Local provider timeout and response limit must be positive")
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self._descriptor = ProviderDescriptor(
            provider_id=provider_id,
            route="local",
            model=model,
            model_version=model_version,
            adapter_version=adapter_version,
            network_required=False,
        )

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor

    def available(self) -> bool:
        executable = self.command[0]
        return shutil.which(executable) is not None or Path(executable).is_file()

    def annotate(self, packet: VisionPacket, *, project_root: Path) -> VisionAnnotation:
        if not self.available():
            raise BlockedError(
                f"Configured local vision executable was not found: {self.command[0]}"
            )
        for frame in packet.frames:
            try:
                image = safe_relative_path(project_root.resolve(strict=True), frame.path)
            except Exception as exc:
                raise ValidationFailure(
                    f"Vision packet image path is unsafe: {frame.path}"
                ) from exc
            if not image.is_file():
                raise ValidationFailure(f"Vision packet image is missing: {frame.path}")
        request = {
            "packet": packet.model_dump(mode="json"),
            "project_root": str(project_root.resolve()),
            "required_annotation_schema": VisionAnnotation.model_json_schema(),
        }
        encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        try:
            completed = subprocess.run(
                list(self.command),
                input=encoded,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise BlockedError(
                f"Configured local vision executable was not found: {self.command[0]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BlockedError(
                f"Local visual annotation exceeded the {self.timeout_seconds:g}s timeout"
            ) from exc
        if completed.returncode != 0:
            detail = re.sub(r"\s+", " ", completed.stderr).strip()[-1000:]
            raise ValidationFailure(
                f"Local vision adapter failed with exit code {completed.returncode}: {detail}"
            )
        if len(completed.stdout.encode("utf-8")) > self.max_response_bytes:
            raise ValidationFailure("Local vision response exceeds the configured size limit")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValidationFailure(f"Local vision adapter returned invalid JSON: {exc}") from exc
        if isinstance(payload, dict) and "annotation" in payload:
            payload = payload["annotation"]
        return validate_annotation_for_packet(payload, packet)
