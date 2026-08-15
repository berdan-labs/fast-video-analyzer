from __future__ import annotations

import platform
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from . import __version__
from .security import atomic_write_json, redact

# Runtime identity is immutable for one Python process.  Cache the relatively
# expensive platform probe once instead of recomputing it on every progress,
# checkpoint, and final manifest serialization.
_RUNTIME_PYTHON = sys.version
_RUNTIME_PLATFORM = platform.platform()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class StageRecord:
    status: str = "pending"
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    elapsed_seconds: float | None = None
    cache_key: str | None = None
    detail: str | None = None


@dataclass
class ManifestBuilder:
    run_id: str
    input_identity: dict[str, Any]
    config_hash: str
    commands: list[list[str]] = field(default_factory=list)
    stages: dict[str, StageRecord] = field(default_factory=dict)
    network_activity: list[dict[str, Any]] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    performance: dict[str, Any] = field(default_factory=dict)
    _started: dict[str, float] = field(default_factory=dict, repr=False)

    def start(self, stage: str, cache_key: str | None = None) -> None:
        self._started[stage] = time.monotonic()
        self.stages[stage] = StageRecord(
            status="processing", started_at_utc=utc_now(), cache_key=cache_key
        )

    def finish(self, stage: str, status: str, detail: str | None = None) -> None:
        if status not in {"completed", "blocked", "failed", "skipped"}:
            raise ValueError(f"Invalid stage status: {status}")
        record = self.stages.setdefault(stage, StageRecord())
        record.status = status
        record.completed_at_utc = utc_now()
        record.elapsed_seconds = round(
            time.monotonic() - self._started.get(stage, time.monotonic()), 6
        )
        record.detail = detail

    def update_performance(self, stage: str, payload: Mapping[str, Any]) -> None:
        """Persist the latest non-authoritative progress telemetry for a stage."""

        self.performance[stage] = dict(payload)

    def as_dict(self) -> dict[str, Any]:
        stage_records = [
            {
                "name": name,
                "status": record.status,
                "started_at_utc": record.started_at_utc,
                "ended_at_utc": record.completed_at_utc,
                "elapsed_ms": (
                    round(record.elapsed_seconds * 1000)
                    if record.elapsed_seconds is not None
                    else None
                ),
                "detail": record.detail,
            }
            for name, record in self.stages.items()
        ]
        source_hashes = {
            key: str(value)
            for key, value in self.input_identity.items()
            if key.endswith("hash") or key in {"sha256", "content_hash"}
        }
        return cast(
            dict[str, Any],
            redact(
                {
                    "schema_version": "1.0",
                    "run_id": self.run_id,
                    "input_identity": self.input_identity,
                    "source_hashes": source_hashes,
                    "source_config_hash": self.config_hash,
                    "configuration_hash": self.config_hash,
                    "code_version": __version__,
                    "runtime": {"python": _RUNTIME_PYTHON, "platform": _RUNTIME_PLATFORM},
                    "exact_commands": self.commands,
                    "commands": self.commands,
                    "stages": {name: vars(record) for name, record in self.stages.items()},
                    "stage_records": stage_records,
                    "tool_versions": {},
                    "dependency_versions": {},
                    "model_versions": {},
                    "prompt_versions": {},
                    "cache_keys": {
                        name: record.cache_key
                        for name, record in self.stages.items()
                        if record.cache_key
                    },
                    "checkpoints": [],
                    "network_activity": self.network_activity,
                    "provider_usage": [],
                    "degradations": self.degradations,
                    "performance": self.performance,
                    "generated_artifacts": sorted(set(self.artifacts)),
                    "reproducibility": {
                        "source_hashes": source_hashes,
                        "configuration_hash": self.config_hash,
                        "code_version": __version__,
                    },
                    "written_at_utc": utc_now(),
                }
            ),
        )

    def write(self, path: Path) -> None:
        atomic_write_json(path, self.as_dict())
