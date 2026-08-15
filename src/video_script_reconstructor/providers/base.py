from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..vision_packets import VisionAnnotation, VisionPacket


@dataclass(frozen=True)
class ProviderDescriptor:
    provider_id: str
    route: Literal["host_agent", "local", "external"]
    model: str | None
    model_version: str | None
    adapter_version: str
    network_required: bool


@dataclass(frozen=True)
class ExternalProcessingPermission:
    allow_network: bool = False
    allow_external_service: bool = False
    allow_media_upload: bool = False
    offline: bool = True
    approved_hosts: tuple[str, ...] = ()

    def permits(self, *, hostname: str, uploads_media: bool) -> bool:
        if self.offline or not self.allow_network or not self.allow_external_service:
            return False
        if uploads_media and not self.allow_media_upload:
            return False
        return not self.approved_hosts or hostname.casefold() in {
            host.casefold() for host in self.approved_hosts
        }


class VisionProvider(ABC):
    # Providers must opt into cross-project semantic caching explicitly.  A
    # local deterministic server can safely reuse an exact annotation, while
    # host-agent and external providers may depend on human state or changing
    # remote responses.
    semantic_cacheable: bool = False

    @property
    @abstractmethod
    def descriptor(self) -> ProviderDescriptor:
        raise NotImplementedError

    @abstractmethod
    def annotate(self, packet: VisionPacket, *, project_root: Path) -> VisionAnnotation:
        raise NotImplementedError
