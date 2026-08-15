"""Managed, offline llama.cpp multimodal provider."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.request import urlopen

from ..errors import BlockedError, ValidationFailure
from ..local_vision_adapter import PROMPT_TEMPLATE_HASH, annotate_via_local_server
from ..model_store import model_directory, verify_model
from ..vision_packets import VisionAnnotation, VisionPacket
from .base import ProviderDescriptor, VisionProvider


class LlamaCppVisionProvider(VisionProvider):
    """Start a loopback-only llama.cpp server from verified local GGUF files."""

    semantic_cacheable = True

    def __init__(
        self,
        *,
        model_name: str = "qwen3-vl-4b-q4",
        model_root: Path | None = None,
        executable: str | None = None,
        port: int | None = None,
        # Qwen3-VL image patches plus compact OCR can legitimately exceed
        # 16k tokens on 4--5-frame packets.  A 32k slot prevents a server-side
        # HTTP 400 before generation; transport resizing remains available for
        # hosts that intentionally choose a smaller context.
        context_size: int = 32_768,
        gpu_layers: int = 99,
        parallel_slots: int = 1,
        startup_timeout_seconds: float = 180.0,
        request_timeout_seconds: float = 300.0,
        adapter_version: str = "1.0",
    ) -> None:
        if parallel_slots < 1 or parallel_slots > 2:
            raise ValueError("parallel_slots must be between 1 and 2")
        if parallel_slots > 1 and os.environ.get("VSR_ALLOW_UNSAFE_SEMANTIC_PARALLEL") != "1":
            raise BlockedError(
                "Two-slot local vision is experimental and can exhaust the shared KV cache; "
                "set VSR_ALLOW_UNSAFE_SEMANTIC_PARALLEL=1 only for an explicit benchmark"
            )
        status = verify_model(model_name, model_root)
        if not status.get("verified"):
            raise BlockedError(
                f"Local vision model is not hash-verified: {model_name}; run models fetch/verify"
            )
        directory = model_directory(model_name, model_root)
        model_files = sorted(
            path for path in directory.glob("*.gguf") if not path.name.startswith("mmproj-")
        )
        projector_files = sorted(directory.glob("mmproj-*.gguf"))
        if len(model_files) != 1 or len(projector_files) != 1:
            raise ValidationFailure(
                "The local vision model requires exactly one GGUF and one multimodal projector"
            )
        resolved_executable = executable or shutil.which("llama-server")
        if not resolved_executable or not Path(resolved_executable).is_file():
            raise BlockedError("llama-server executable is unavailable")
        self.executable = str(Path(resolved_executable).resolve())
        self.model_path = model_files[0]
        self.projector_path = projector_files[0]
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        # Two slots are deliberately the hard ceiling: the verified local
        # Qwen3-VL quantization already occupies most of a 16 GB host, and
        # higher fan-out can trade throughput for paging/OOM rather than
        # improving it.  The semantic scheduler remains single-worker by
        # default; callers opt into the second slot after benchmarking.
        self.parallel_slots = parallel_slots
        self.port = port or self._free_port()
        self.startup_timeout_seconds = startup_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.prompt_template_hash = PROMPT_TEMPLATE_HASH
        self._process: subprocess.Popen[bytes] | None = None
        self._model_id = str(self.model_path)
        self._descriptor = ProviderDescriptor(
            provider_id="llama.cpp-local",
            route="local",
            model=model_name,
            model_version=str(status.get("revision")),
            adapter_version=adapter_version,
            network_required=False,
        )

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/chat/completions"

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        command = [
            self.executable,
            "-m",
            str(self.model_path),
            "--mmproj",
            str(self.projector_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-c",
            str(self.context_size),
            "-ngl",
            str(self.gpu_layers),
            "-np",
            str(self.parallel_slots),
            "--cont-batching",
            "--no-webui",
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "NO_PROXY": "127.0.0.1,localhost",
            }
        )
        self._process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            env=environment,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
        deadline = time.monotonic() + self.startup_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise BlockedError(
                    f"llama-server exited during startup with code {self._process.returncode}"
                )
            try:
                with urlopen(  # noqa: S310 - fixed loopback URL.
                    f"http://127.0.0.1:{self.port}/v1/models", timeout=2
                ) as response:
                    payload = json.loads(response.read(1024 * 1024).decode("utf-8"))
                model_data = payload.get("data", []) if isinstance(payload, dict) else []
                if model_data and isinstance(model_data[0], dict):
                    self._model_id = str(model_data[0].get("id") or self.model_path)
                return
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(0.25)
        self.close()
        raise BlockedError(f"llama-server did not become ready: {last_error}")

    def annotate(self, packet: VisionPacket, *, project_root: Path) -> VisionAnnotation:
        return self._annotate_request(packet, project_root=project_root)

    def annotate_with_transport_context(
        self,
        packet: VisionPacket,
        *,
        project_root: Path,
        transport_frame_hashes: Mapping[str, str],
    ) -> VisionAnnotation:
        """Annotate with canonical frame digests for cross-project image reuse.

        The digests are transport-only cache identities. They are not added to
        the prompt and never influence the model's visible-fact claims.
        Providers that do not expose this optional hook retain the ordinary
        path/stat-bound transport cache behavior.
        """

        return self._annotate_request(
            packet,
            project_root=project_root,
            transport_frame_hashes=transport_frame_hashes,
        )

    def _annotate_request(
        self,
        packet: VisionPacket,
        *,
        project_root: Path,
        transport_frame_hashes: Mapping[str, str] | None = None,
    ) -> VisionAnnotation:
        self.start()
        request: dict[str, Any] = {
            "packet": packet.model_dump(mode="json"),
            "project_root": str(project_root.resolve(strict=True)),
            "required_annotation_schema": VisionAnnotation.model_json_schema(),
        }
        if transport_frame_hashes:
            request["transport_frame_hashes"] = {
                str(frame_id): str(digest)
                for frame_id, digest in transport_frame_hashes.items()
                if str(frame_id) and str(digest)
            }
        return annotate_via_local_server(
            request,
            endpoint=self.endpoint,
            model=self._model_id,
            timeout_seconds=self.request_timeout_seconds,
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def __enter__(self) -> LlamaCppVisionProvider:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["LlamaCppVisionProvider"]
