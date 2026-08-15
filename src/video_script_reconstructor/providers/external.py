from __future__ import annotations

import base64
import ipaddress
import json
import os
import socket
from collections.abc import Callable, Mapping
from http.client import HTTPMessage
from pathlib import Path
from typing import IO
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..errors import BlockedError, InputError, SecurityError, ValidationFailure
from ..security import redact, safe_relative_path, validate_remote_url
from ..vision_packets import VisionAnnotation, VisionPacket, validate_annotation_for_packet
from .base import ExternalProcessingPermission, ProviderDescriptor, VisionProvider


def _validate_public_resolution(url: str) -> str:
    validate_remote_url(url)
    parsed = urlparse(url)
    if parsed.hostname is None:
        raise SecurityError("External provider URL has no hostname")
    try:
        answers = socket.getaddrinfo(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        )
    except OSError as exc:
        raise SecurityError(f"Unable to resolve external provider host: {parsed.hostname}") from exc
    for answer in answers:
        address = ipaddress.ip_address(answer[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            raise SecurityError("External provider resolved to a non-public address")
    return url


class _ValidatingRedirectHandler(HTTPRedirectHandler):
    def __init__(self, permission: ExternalProcessingPermission, *, origin_host: str) -> None:
        super().__init__()
        self.permission = permission
        self.origin_host = origin_host.casefold()

    def redirect_request(
        self, req: Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage, newurl: str
    ) -> Request | None:
        resolved = _validate_public_resolution(urljoin(req.full_url, newurl))
        hostname = urlparse(resolved).hostname or ""
        explicitly_approved = {host.casefold() for host in self.permission.approved_hosts}
        allowed_redirect_hosts = explicitly_approved or {self.origin_host}
        if hostname.casefold() not in allowed_redirect_hosts or not self.permission.permits(
            hostname=hostname, uploads_media=True
        ):
            raise SecurityError(f"Redirect target is not explicitly permitted: {hostname}")
        redirected = super().redirect_request(req, fp, code, msg, headers, resolved)
        if redirected is not None and hostname.casefold() != self.origin_host:
            redirected.remove_header("Authorization")
        return redirected


class ExternalVisionProvider(VisionProvider):
    def __init__(
        self,
        *,
        endpoint: str,
        provider_id: str,
        model: str,
        permission: ExternalProcessingPermission,
        credential_env: str | None = None,
        model_version: str | None = None,
        adapter_version: str = "1.0",
        timeout_seconds: float = 60.0,
        max_upload_bytes: int = 32 * 1024 * 1024,
        max_response_bytes: int = 4 * 1024 * 1024,
        network_recorder: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme != "https":
            raise SecurityError("External visual providers require HTTPS")
        validate_remote_url(endpoint)
        if not parsed.hostname or not permission.permits(
            hostname=parsed.hostname, uploads_media=True
        ):
            raise SecurityError("External provider and media upload require explicit permission")
        if timeout_seconds <= 0 or max_upload_bytes <= 0 or max_response_bytes <= 0:
            raise InputError("External provider limits must be positive")
        self.endpoint = endpoint
        self.permission = permission
        self.credential_env = credential_env
        self.timeout_seconds = timeout_seconds
        self.max_upload_bytes = max_upload_bytes
        self.max_response_bytes = max_response_bytes
        self.network_recorder = network_recorder
        self._descriptor = ProviderDescriptor(
            provider_id=provider_id,
            route="external",
            model=model,
            model_version=model_version,
            adapter_version=adapter_version,
            network_required=True,
        )

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor

    def _request_payload(self, packet: VisionPacket, project_root: Path) -> bytes:
        root = project_root.resolve(strict=True)
        images: list[dict[str, str]] = []
        total = 0
        for frame in packet.frames:
            try:
                image_path = safe_relative_path(root, frame.path)
            except Exception as exc:
                raise ValidationFailure(
                    f"Vision packet image path is unsafe: {frame.path}"
                ) from exc
            if not image_path.is_file():
                raise ValidationFailure(f"Vision packet image is missing: {frame.path}")
            data = image_path.read_bytes()
            total += len(data)
            if total > self.max_upload_bytes:
                raise BlockedError(
                    "Visual packet media exceeds the explicitly configured upload limit"
                )
            suffix = image_path.suffix.casefold()
            media_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
            }.get(suffix)
            if media_type is None:
                raise ValidationFailure(
                    f"Unsupported visual-provider image format: {suffix or '<none>'}"
                )
            images.append(
                {
                    "frame_id": frame.frame_id,
                    "media_type": media_type,
                    "base64": base64.b64encode(data).decode("ascii"),
                }
            )
        payload = {
            "model": self.descriptor.model,
            "packet": packet.model_dump(mode="json"),
            "images": images,
            "required_annotation_schema": VisionAnnotation.model_json_schema(),
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > self.max_upload_bytes:
            raise BlockedError(
                "Encoded visual packet exceeds the explicitly configured upload limit"
            )
        return encoded

    def annotate(self, packet: VisionPacket, *, project_root: Path) -> VisionAnnotation:
        parsed = urlparse(self.endpoint)
        hostname = parsed.hostname or ""
        if not self.permission.permits(hostname=hostname, uploads_media=True):
            raise SecurityError(
                "External processing permission is absent or offline mode is active"
            )
        _validate_public_resolution(self.endpoint)
        body = self._request_payload(packet, project_root)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.credential_env:
            credential = os.environ.get(self.credential_env)
            if not credential:
                raise BlockedError(
                    f"Credential environment variable is not set: {self.credential_env}"
                )
            headers["Authorization"] = f"Bearer {credential}"
        if self.network_recorder:
            self.network_recorder(
                {
                    "action": "external_visual_annotation",
                    "provider": self.descriptor.provider_id,
                    "host": hostname,
                    "upload_bytes": len(body),
                    "candidate_id": packet.candidate_id,
                }
            )
        # The endpoint scheme/host and every redirect are validated above and by the handler.
        request = Request(self.endpoint, data=body, headers=headers, method="POST")  # noqa: S310
        opener = build_opener(_ValidatingRedirectHandler(self.permission, origin_host=hostname))
        try:
            with opener.open(request, timeout=self.timeout_seconds) as response:
                data = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            raise ValidationFailure(f"External vision provider returned HTTP {exc.code}") from exc
        except URLError as exc:
            safe_reason = redact(str(exc.reason))
            raise BlockedError(f"External vision provider request failed: {safe_reason}") from exc
        if len(data) > self.max_response_bytes:
            raise ValidationFailure("External vision response exceeds the configured size limit")
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationFailure(
                f"External vision provider returned invalid JSON: {exc}"
            ) from exc
        if isinstance(payload, dict) and "annotation" in payload:
            payload = payload["annotation"]
        return validate_annotation_for_packet(payload, packet)
