"""Apply semantic vision providers to persisted evidence packets."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .audit import audit_project
from .cache import cache_key
from .errors import ValidationFailure
from .evidence import ingest_project_observation
from .frame_quality import (
    PERCEPTUAL_DHASH_ALGORITHM,
    PERCEPTUAL_DHASH_VERIFIED,
    normalize_ocr_for_comparison,
)
from .ids import sequential_id
from .local_vision_adapter import local_vision_transport_profile
from .metadata_reconcile import SemanticBatchContext
from .providers.base import VisionProvider
from .providers.host_agent import HostAgentVisionProvider
from .render_markdown import render_to_path
from .retention import discover_projects
from .security import (
    JsonPatchState,
    atomic_update_json_fields,
    atomic_write_json,
    atomic_write_text,
    canonical_compact_for_payload,
    redact,
    sha256_file,
)
from .validate_output import ValidationResult, validate_project, write_validation_receipt
from .vision_observation import annotation_to_observation
from .vision_packets import (
    VisionAnnotation,
    VisionPacket,
    validate_annotation_for_packet,
)

LOGGER = logging.getLogger(__name__)
_SEMANTIC_CACHE_SCHEMA = "semantic-vision-v3-pixel-hash-transport-resize"
_SEMANTIC_CONTENT_CACHE_SCHEMA = "semantic-vision-v2-content-hash-remap-context"
_PACKET_CACHE_LIMIT = 4096
_PACKET_CACHE: dict[Path, tuple[tuple[int, int, int], VisionPacket]] = {}
_SEMANTIC_PRUNE_INTERVAL = 32
_SEMANTIC_PRUNE_STATE: dict[Path, tuple[int, int]] = {}
_STALE_CANDIDATE_ERROR = re.compile(
    r"^image_metadata: hidden candidate (?P<name>[^/\\]+) "
    r"lacks a canonical ledger mirror$"
)
_SEMANTIC_BATCH_JOURNAL_SCHEMA = "video-script-reconstructor.semantic-batch-journal"
_SEMANTIC_BATCH_JOURNAL_VERSION = "1.0"


def _journal_upsert(items: list[dict[str, Any]], value: Mapping[str, Any], key: str) -> None:
    """Replace one identity-bearing item or append it exactly once."""

    identity = value.get(key)
    if identity is None:
        return
    identity = str(identity)
    for index, existing in enumerate(items):
        if isinstance(existing, Mapping) and str(existing.get(key)) == identity:
            items[index] = dict(value)
            return
    items.append(dict(value))


def _journal_upsert_nested(
    items: list[dict[str, Any]], value: Mapping[str, Any], path: Sequence[str]
) -> None:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping):
            return
        current = current.get(part)
    if current is None:
        return
    identity = str(current)
    for index, existing in enumerate(items):
        if not isinstance(existing, Mapping):
            continue
        candidate: Any = existing
        for part in path:
            if not isinstance(candidate, Mapping):
                candidate = None
                break
            candidate = candidate.get(part)
        if candidate is not None and str(candidate) == identity:
            items[index] = dict(value)
            return
    items.append(dict(value))


class _SemanticBatchJournal:
    """Durable per-observation deltas for one semantic apply batch.

    Canonical and ledger JSON are intentionally not rewritten from the hot
    observation loop.  Each accepted packet appends a compact, identity-keyed
    delta and fsyncs it.  A final materialization (or restart recovery) folds
    the deltas into the canonical/ledger trees exactly once.  Upserts make a
    crash after canonical replacement but before journal cleanup idempotent.
    """

    defer_ledger = True

    def __init__(self, project_dir: Path, path: Path) -> None:
        self.project_dir = project_dir
        self.path = path
        self.count = 0

    @property
    def ledger_path(self) -> Path:
        return self.project_dir / ".state" / "vision" / "image-observations.json"

    @classmethod
    def start(cls, project_dir: Path) -> _SemanticBatchJournal:
        path = project_dir / ".state" / "checkpoints" / "semantic-batch-journal.jsonl"
        if path.exists() or path.is_symlink():
            raise ValidationFailure(f"Semantic batch journal already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "schema_name": _SEMANTIC_BATCH_JOURNAL_SCHEMA,
            "schema_version": _SEMANTIC_BATCH_JOURNAL_VERSION,
            "batch_id": f"{os.getpid()}-{time.time_ns()}",
            "base_canonical_project_sha256": sha256_file(
                project_dir / ".state" / "canonical-project.json"
            ),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        encoded = json.dumps(
            redact(header), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        atomic_write_text(path, encoded + "\n")
        return cls(project_dir, path)

    def append(self, entry: dict[str, Any]) -> None:
        if not self.path.is_file() or self.path.is_symlink():
            raise ValidationFailure("Semantic batch journal disappeared before append")
        encoded = json.dumps(
            redact(entry), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if len(encoded.encode("utf-8")) > 16 * 1024 * 1024:
            raise ValidationFailure("Semantic batch journal entry exceeds the size limit")
        with self.path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.count += 1

    def complete(self, ledger: Mapping[str, Any]) -> None:
        # Ledger materialization is separate from canonical finalization so a
        # crash in either atomic replace leaves the journal available for
        # replay.  The journal is removed only after both are durable.
        atomic_write_json(self.ledger_path, ledger, compact=True)
        self.path.unlink(missing_ok=True)

    @classmethod
    def _read(cls, project_dir: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
        path = project_dir / ".state" / "checkpoints" / "semantic-batch-journal.jsonl"
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if path.stat().st_size > 512 * 1024 * 1024:
            raise ValidationFailure("Semantic batch journal exceeds the bounded size limit")
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValidationFailure("Semantic batch journal is empty")
        try:
            header = json.loads(lines[0])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValidationFailure("Semantic batch journal header is invalid") from exc
        if not isinstance(header, dict) or (
            header.get("schema_name") != _SEMANTIC_BATCH_JOURNAL_SCHEMA
            or header.get("schema_version") != _SEMANTIC_BATCH_JOURNAL_VERSION
        ):
            raise ValidationFailure("Unsupported semantic batch journal schema")
        entries: list[dict[str, Any]] = []
        for line_index, raw in enumerate(lines[1:], start=1):
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                # A torn final append is ignored; every complete prior line is
                # independently fsynced and remains replayable.
                if line_index == len(lines) - 1:
                    break
                raise ValidationFailure("Semantic batch journal contains invalid JSON") from exc
            if not isinstance(entry, dict):
                raise ValidationFailure("Semantic batch journal entry is not an object")
            entries.append(entry)
        return path, header, entries

    @classmethod
    def recover(cls, project_dir: Path) -> list[str]:
        """Replay an interrupted batch, returning candidate IDs recovered."""

        path = project_dir / ".state" / "checkpoints" / "semantic-batch-journal.jsonl"
        if path.is_symlink():
            raise ValidationFailure("Semantic batch journal must not be a symlink")
        if not path.exists():
            return []
        _path, header, entries = cls._read(project_dir)
        if not entries:
            path.unlink(missing_ok=True)
            return []
        canonical_path = project_dir / ".state" / "canonical-project.json"
        base_digest = header.get("base_canonical_project_sha256")
        if not isinstance(base_digest, str) or len(base_digest) != 64:
            raise ValidationFailure("Semantic batch journal has no canonical base digest")
        current_digest = sha256_file(canonical_path)
        target_candidates: set[str] = set()
        target_observations: set[str] = set()
        for entry in entries:
            candidate_id = entry.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id:
                target_candidates.add(candidate_id)
            events = entry.get("visual_events")
            if isinstance(events, Mapping):
                target_candidates.update(
                    str(event_id) for event_id in events if event_id is not None
                )
            observation = entry.get("observation")
            if isinstance(observation, Mapping):
                observation_id = observation.get("observation_id")
                if isinstance(observation_id, str) and observation_id:
                    target_observations.add(observation_id)
        project = _load_project(project_dir)
        if current_digest != base_digest:
            present_candidates = {
                str(event.get("event_id"))
                for event in project.get("visual_events", [])
                if isinstance(event, Mapping) and event.get("event_id")
            }
            present_observations = {
                str(observation.get("observation_id"))
                for observation in project.get("visual_observations", [])
                if isinstance(observation, Mapping) and observation.get("observation_id")
            }
            # A differing digest is expected only when canonical replacement
            # succeeded immediately before a crash. Require all journal
            # targets to already be represented before accepting that state;
            # otherwise an external edit/stale journal must not be merged.
            if not (
                target_observations.issubset(present_observations)
                and target_candidates.issubset(present_candidates)
            ):
                raise ValidationFailure(
                    "Canonical project changed outside the semantic batch journal"
                )
        ledger = _load_ledger(project_dir)
        recovered: list[str] = []
        for entry in entries:
            _apply_semantic_batch_entry(project, ledger, entry)
            candidate_id = entry.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id:
                recovered.append(candidate_id)
        # The scheduling marker is intentionally not copied into each small
        # delta. Reconcile it from the recovered event frontier so a crash
        # after filtered-pass bookkeeping but before final materialization does
        # not resurrect stale deferred IDs.
        _sync_semantic_budget_review_item(
            project,
            _pending_semantic_event_ids(project_dir, project),
            provider_id="semantic-recovery",
        )
        # Ledger first is preferable: if canonical replacement is interrupted,
        # the still-present journal can complete canonical replay on restart.
        atomic_write_json(cls(project_dir, path).ledger_path, ledger, compact=True)
        _finalize_semantic_project(project_dir, project, force_full_write=True)
        path.unlink(missing_ok=True)
        return list(dict.fromkeys(recovered))


def _apply_semantic_batch_entry(
    project: dict[str, Any], ledger: dict[str, Any], entry: Mapping[str, Any]
) -> None:
    """Fold one journal delta into canonical and ledger state idempotently."""

    def array(name: str) -> list[dict[str, Any]]:
        value = project.setdefault(name, [])
        if not isinstance(value, list):
            raise ValidationFailure(f"Canonical {name} must be a list")
        return value

    observations = entry.get("observation")
    if isinstance(observations, Mapping):
        _journal_upsert(array("visual_observations"), observations, "observation_id")
    revision = entry.get("metadata_revision")
    if isinstance(revision, Mapping):
        _journal_upsert(array("metadata_revisions"), revision, "revision_id")
    decision = entry.get("sufficiency_decision")
    if isinstance(decision, Mapping):
        _journal_upsert(array("sufficiency_decisions"), decision, "decision_id")
    for claim in entry.get("claims", []):
        if isinstance(claim, Mapping):
            _journal_upsert(array("image_claims"), claim, "claim_id")
    for name, values in (
        ("frames", entry.get("frames")),
        ("evidence_image_metadata", entry.get("evidence_image_metadata")),
        ("script_blocks", entry.get("script_blocks")),
        ("visual_events", entry.get("visual_events")),
        ("review_items", entry.get("review_items")),
    ):
        if not isinstance(values, Mapping):
            continue
        identity = {
            "frames": "frame_id",
            "evidence_image_metadata": "image.image_id",
            "script_blocks": "block_id",
            "visual_events": "event_id",
            "review_items": "review_id",
        }[name]
        path = identity.split(".")
        for value in values.values():
            if isinstance(value, Mapping):
                _journal_upsert_nested(array(name), value, path)

    def ledger_array(name: str) -> list[dict[str, Any]]:
        value = ledger.setdefault(name, [])
        if not isinstance(value, list):
            raise ValidationFailure(f"Ledger {name} must be a list")
        return value

    history = entry.get("payload_history")
    if isinstance(history, Mapping):
        digest = history.get("integrity", {}).get("payload_digest") if isinstance(history.get("integrity"), Mapping) else None
        history_items = ledger_array("payload_history")
        if digest is None or not any(
            isinstance(item, Mapping)
            and isinstance(item.get("integrity"), Mapping)
            and item.get("integrity", {}).get("payload_digest") == digest
            for item in history_items
        ):
            history_items.append(dict(history))
    payload = entry.get("payload")
    if isinstance(payload, Mapping):
        _journal_upsert_nested(ledger_array("payloads"), payload, ("image", "image_id"))
    if isinstance(observations, Mapping):
        _journal_upsert(ledger_array("observations"), observations, "observation_id")
    if isinstance(revision, Mapping):
        _journal_upsert(ledger_array("revisions"), revision, "revision_id")
    for claim in entry.get("claims", []):
        if isinstance(claim, Mapping):
            _journal_upsert(ledger_array("claims"), claim, "claim_id")


def _write_semantic_sidecars(
    annotation_path: Path,
    observation_path: Path,
    annotation: VisionAnnotation,
    observation: Any,
) -> None:
    """Persist machine-readable semantic sidecars in compact JSON form.

    These files are resumability/provenance artifacts, not hand-edited
    reports.  Compact UTF-8 keeps the per-observation write path small while
    preserving the parsed payload and the atomic-write durability contract.
    """

    atomic_write_json(annotation_path, annotation.model_dump(mode="json"), compact=True)
    atomic_write_json(observation_path, observation.model_dump(mode="json"), compact=True)


def _prune_unmirrored_generated_candidates(
    project_dir: Path,
    errors: list[str],
) -> int:
    """Remove only validator-proven, unmirrored generated candidate PNGs.

    An explicit rebuild can leave rejected-frame files from an older visual
    survey in ``.state/candidates/rejected-frames``.  They are not canonical
    evidence, and the validator intentionally rejects them when no ledger
    mirror exists.  Restrict cleanup to the exact basenames reported by that
    validator and to the generated candidate directory; source media, final
    evidence, and mirrored candidates are never touched.
    """

    candidate_root = (project_dir / ".state" / "candidates" / "rejected-frames").resolve()
    if not candidate_root.is_dir():
        return 0
    targets: list[Path] = []
    for error in errors:
        match = _STALE_CANDIDATE_ERROR.fullmatch(str(error))
        if match is None:
            continue
        target = (candidate_root / match.group("name")).resolve()
        if target.parent != candidate_root or not target.is_file() or target.is_symlink():
            continue
        targets.append(target)
    removed = 0
    for target in dict.fromkeys(targets):
        try:
            target.unlink()
        except OSError:
            LOGGER.warning("Unable to prune stale generated candidate %s", target)
        else:
            removed += 1
    if removed:
        LOGGER.info("Pruned %d validator-proven stale generated candidates", removed)
    return removed


def _load_project(project_dir: Path) -> dict[str, Any]:
    path = project_dir / ".state" / "canonical-project.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValidationFailure("Canonical project root is not an object")
    return loaded


def _load_ledger(project_dir: Path) -> dict[str, Any]:
    path = project_dir / ".state" / "vision" / "image-observations.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        loaded = None
    if not isinstance(loaded, dict):
        loaded = {
            "schema_version": "1.0",
            "payloads": [],
            "payload_history": [],
            "observations": [],
            "claims": [],
            "revisions": [],
        }
    return loaded


def _semantic_cache_enabled(provider: VisionProvider) -> bool:
    """Allow only explicitly deterministic local providers to share results."""

    disabled = os.environ.get("VSR_DISABLE_SEMANTIC_SHARED_CACHE", "").strip().casefold()
    if disabled in {"1", "true", "yes", "on"}:
        return False
    visual_disabled = os.environ.get("VSR_DISABLE_VISUAL_SHARED_CACHE", "").strip().casefold()
    if visual_disabled in {"1", "true", "yes", "on"}:
        return False
    return bool(
        getattr(provider, "semantic_cacheable", False)
        and provider.descriptor.route == "local"
    )


def _semantic_cache_root() -> Path | None:
    """Resolve the bounded local cache for deterministic semantic annotations."""

    configured = os.environ.get("VSR_SEMANTIC_SHARED_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "video-script-reconstructor" / "visual-cache" / "semantic").resolve()


def _semantic_cache_limit() -> int:
    default_limit = 256 * 1024 * 1024
    raw_limit = os.environ.get("VSR_SEMANTIC_SHARED_CACHE_MAX_BYTES", "").strip()
    if not raw_limit:
        return default_limit
    try:
        return max(0, int(raw_limit))
    except ValueError:
        LOGGER.warning(
            "Ignoring invalid VSR_SEMANTIC_SHARED_CACHE_MAX_BYTES=%r", raw_limit
        )
        return default_limit


def _provider_cache_descriptor(provider: VisionProvider) -> dict[str, Any]:
    descriptor = provider.descriptor
    return {
        "provider_id": descriptor.provider_id,
        "route": descriptor.route,
        "model": descriptor.model,
        "model_version": descriptor.model_version,
        "adapter_version": descriptor.adapter_version,
        "network_required": descriptor.network_required,
        "prompt_template_hash": getattr(provider, "prompt_template_hash", None),
        "context_size": getattr(provider, "context_size", None),
        "gpu_layers": getattr(provider, "gpu_layers", None),
    }


def _semantic_transport_profile(
    provider: VisionProvider,
    packet: VisionPacket | None,
) -> dict[str, Any]:
    """Return transport identity for cache keys without constraining other providers."""

    if packet is None or provider.descriptor.provider_id != "llama.cpp-local":
        return {"profile": "provider-default-v1"}
    return local_vision_transport_profile(packet)


def _packet_frame_hashes(
    project: Mapping[str, Any],
    packet: VisionPacket,
    project_dir: Path,
    *,
    frame_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Bind a semantic cache entry to the exact evidence PNG bytes."""

    # ``apply_vision_provider`` already builds this index once for packet
    # selection and commit/link work. Reusing it avoids rebuilding an O(F)
    # dictionary for every packet in the semantic hot loop. Keep the
    # project-derived fallback for direct callers and older integrations.
    canonical_frames = frame_by_id
    if canonical_frames is None:
        canonical_frames = {
            str(frame.get("frame_id")): frame
            for frame in project.get("frames", [])
            if isinstance(frame, Mapping) and frame.get("frame_id")
        }
    hashes: dict[str, str] = {}
    for frame in packet.frames:
        canonical = canonical_frames.get(frame.frame_id, {})
        pixel_hash = canonical.get("pixel_hash") if isinstance(canonical, Mapping) else None
        if not isinstance(pixel_hash, Mapping):
            metadata = canonical.get("metadata") if isinstance(canonical, Mapping) else None
            image_metadata = metadata.get("image") if isinstance(metadata, Mapping) else None
            pixel_hash = (
                image_metadata.get("pixel_hash")
                if isinstance(image_metadata, Mapping)
                else None
            )
        if isinstance(pixel_hash, Mapping) and isinstance(pixel_hash.get("value"), str):
            # PNG containers may differ across runs because canonical metadata
            # is embedded after extraction.  The VLM sees decoded pixels, so a
            # stable RGBA pixel digest is the correct reuse identity and avoids
            # re-running expensive local inference for byte-identical imagery.
            hashes[frame.frame_id] = str(pixel_hash["value"])
            continue
        stored = canonical.get("file_hash") if isinstance(canonical, Mapping) else None
        if isinstance(stored, str) and stored:
            hashes[frame.frame_id] = stored
            continue
        path = project_dir / frame.path
        hashes[frame.frame_id] = sha256_file(path)
    return hashes


def _annotate_with_transport_identity(
    provider: VisionProvider,
    packet: VisionPacket,
    *,
    project_dir: Path,
    frame_hashes: Mapping[str, str] | None,
) -> VisionAnnotation:
    """Call an optional local provider hook with canonical image identities.

    The hook is deliberately discovered at runtime so host-agent/external
    providers keep their existing interface and never receive filesystem or
    cache-only data. Local llama.cpp can use the decoded pixel digest to reuse
    its prepared data URL when the same evidence is present in another
    project, without changing the prompt or semantic cache contract.
    """

    contextual = getattr(provider, "annotate_with_transport_context", None)
    if callable(contextual) and frame_hashes:
        return VisionAnnotation.model_validate(
            contextual(
                packet,
                project_root=project_dir,
                transport_frame_hashes=frame_hashes,
            )
        )
    return provider.annotate(packet, project_root=project_dir)


def _semantic_cache_key(
    provider: VisionProvider,
    packet: VisionPacket,
    frame_hashes: Mapping[str, str],
) -> str:
    return cache_key(
        _SEMANTIC_CACHE_SCHEMA,
        __version__,
        _provider_cache_descriptor(provider),
        _semantic_transport_profile(provider, packet),
        packet.model_dump(mode="json"),
        dict(sorted(frame_hashes.items())),
    )


def _semantic_content_cache_key(
    provider: VisionProvider,
    content_key: str,
    *,
    packet: VisionPacket | None = None,
) -> str:
    """Key a semantic result by immutable visual content, not candidate IDs.

    A project can contain repeated packets for the same decoded pixels (for
    example adjacent visual events that retain an identical before/after pair).
    The ordinary packet cache deliberately includes packet IDs and therefore
    cannot reuse those results.  This second key is still bound to the exact
    provider/model/prompt descriptor and to :func:`_semantic_visual_content_key`;
    only the transient candidate/frame IDs are omitted so a stored annotation
    can be remapped and validated against the target packet before commit.
    """

    return cache_key(
        _SEMANTIC_CONTENT_CACHE_SCHEMA,
        __version__,
        _provider_cache_descriptor(provider),
        _semantic_transport_profile(provider, packet),
        content_key,
    )


def _read_semantic_cache(
    path: Path,
    *,
    key: str,
    packet: VisionPacket,
    max_bytes: int,
) -> VisionAnnotation | None:
    if max_bytes <= 0 or not path.is_file() or path.is_symlink():
        return None
    try:
        if path.stat().st_size > max_bytes:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return None
        if (
            payload.get("schema_version") != "1.0"
            or payload.get("cache_kind") != _SEMANTIC_CACHE_SCHEMA
            or payload.get("cache_key") != key
        ):
            return None
        annotation = payload.get("annotation")
        if not isinstance(annotation, Mapping):
            return None
        return validate_annotation_for_packet(annotation, packet)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        LOGGER.info("Ignoring invalid semantic cache entry: %s", path)
        return None


def _read_semantic_content_cache(
    path: Path,
    *,
    key: str,
    packet: VisionPacket,
    source_frame_ids: Sequence[str],
    max_bytes: int,
) -> VisionAnnotation | None:
    """Read a content-addressed annotation and remap it to ``packet`` safely."""

    if max_bytes <= 0 or not path.is_file() or path.is_symlink():
        return None
    try:
        if path.stat().st_size > max_bytes:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return None
        if (
            payload.get("schema_version") != "1.0"
            or payload.get("cache_kind") != _SEMANTIC_CONTENT_CACHE_SCHEMA
            or payload.get("cache_key") != key
        ):
            return None
        stored_ids = payload.get("source_frame_ids")
        if not isinstance(stored_ids, list) or tuple(str(item) for item in stored_ids) != tuple(
            source_frame_ids
        ):
            return None
        annotation = payload.get("annotation")
        if not isinstance(annotation, Mapping):
            return None
        parsed = VisionAnnotation.model_validate(annotation)
        if parsed.candidate_id != str(payload.get("source_candidate_id", "")):
            return None
        # The ordinary packet validator below proves every remapped citation is
        # in the target packet.  Validate the source shape first so malformed or
        # stale entries cannot silently survive a frame-ID remap.
        return _remap_reused_annotation(
            parsed,
            source_frame_ids=tuple(source_frame_ids),
            packet=packet,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        LOGGER.info("Ignoring invalid semantic content cache entry: %s", path)
        return None


def _prune_semantic_cache(root: Path, *, current: Path, max_bytes: int) -> int:
    if max_bytes <= 0 or not root.is_dir():
        return 0
    try:
        entries = [
            item
            for item in root.glob("*.json")
            if item.is_file() and not item.is_symlink()
        ]
        total = sum(item.stat().st_size for item in entries)
        if total <= max_bytes:
            return total
        current_resolved = current.resolve()
        for item in sorted(entries, key=lambda item: (item.stat().st_mtime_ns, item.name)):
            if total <= max_bytes:
                break
            if item.resolve() == current_resolved:
                continue
            try:
                size = item.stat().st_size
                item.unlink()
                total -= size
            except OSError:
                LOGGER.warning("Unable to prune semantic cache entry: %s", item)
        return total
    except OSError:
        LOGGER.info("Skipping semantic cache pruning after receipt inspection failure")
        return 0


def _semantic_cache_prune_due(path: Path, *, encoded_size: int, max_bytes: int) -> bool:
    """Track cache growth without rescanning the directory on every write.

    Cache writes are normally unique files, so a stat-bound approximate ledger
    is exact for the common case and still handles overwrites by subtracting
    the previous size.  A full directory scan remains bounded to every 32
    writes, or happens immediately when the configured byte limit is crossed.
    The state is process-local; a new process initializes from the directory
    once and therefore remains safe when another process has changed the cache.
    """

    root = path.parent.resolve()
    try:
        existing_size = path.stat().st_size if path.is_file() and not path.is_symlink() else 0
    except OSError:
        existing_size = 0
    state = _SEMANTIC_PRUNE_STATE.get(root)
    if state is None:
        try:
            total = sum(
                item.stat().st_size
                for item in root.glob("*.json")
                if item.is_file() and not item.is_symlink()
            )
        except OSError:
            total = 0
        writes = 0
    else:
        writes, total = state
    total = max(0, total - existing_size) + encoded_size
    writes += 1
    _SEMANTIC_PRUNE_STATE[root] = (writes, total)
    return total > max_bytes or writes >= _SEMANTIC_PRUNE_INTERVAL


def _write_semantic_cache(
    path: Path,
    *,
    key: str,
    provider: VisionProvider,
    packet: VisionPacket,
    annotation: VisionAnnotation,
    frame_hashes: Mapping[str, str],
    max_bytes: int,
) -> bool:
    if max_bytes <= 0:
        return False
    payload = {
        "schema_version": "1.0",
        "cache_kind": _SEMANTIC_CACHE_SCHEMA,
        "cache_key": key,
        "provider": _provider_cache_descriptor(provider),
        "transport_profile": _semantic_transport_profile(provider, packet),
        "candidate_id": packet.candidate_id,
        "frame_hashes": dict(sorted(frame_hashes.items())),
        "annotation": annotation.model_dump(mode="json"),
    }
    try:
        encoded_size = len(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        if encoded_size > max_bytes:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        prune_due = _semantic_cache_prune_due(
            path,
            encoded_size=encoded_size,
            max_bytes=max_bytes,
        )
        # The size guard above measures compact JSON, so persist the same
        # representation.  Pretty-printing here would make a bounded cache
        # consume more SSD than the admission check predicts.
        atomic_write_json(path, payload, compact=True)
        if prune_due:
            total = _prune_semantic_cache(path.parent, current=path, max_bytes=max_bytes)
            _SEMANTIC_PRUNE_STATE[path.parent.resolve()] = (0, total)
        return True
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.warning("Unable to persist semantic cache entry %s: %s", path, exc)
        return False


def _write_semantic_content_cache(
    path: Path,
    *,
    key: str,
    provider: VisionProvider,
    packet: VisionPacket,
    annotation: VisionAnnotation,
    source_frame_ids: Sequence[str],
    max_bytes: int,
) -> bool:
    """Persist a provider result for exact visual-content reuse.

    The cache stores only schema-validated annotation data and the ordered
    source frame IDs needed for citation remapping.  It is bounded by the same
    shared semantic-cache budget and is never treated as canonical evidence:
    every hit is remapped and passed through normal packet validation first.
    """

    if max_bytes <= 0:
        return False
    payload = {
        "schema_version": "1.0",
        "cache_kind": _SEMANTIC_CONTENT_CACHE_SCHEMA,
        "cache_key": key,
        "provider": _provider_cache_descriptor(provider),
        "transport_profile": _semantic_transport_profile(provider, packet),
        "source_candidate_id": packet.candidate_id,
        "source_frame_ids": list(source_frame_ids),
        "annotation": annotation.model_dump(mode="json"),
    }
    try:
        encoded_size = len(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        if encoded_size > max_bytes:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        prune_due = _semantic_cache_prune_due(
            path,
            encoded_size=encoded_size,
            max_bytes=max_bytes,
        )
        # Keep the on-disk representation aligned with the compact size guard
        # while preserving the exact JSON payload and cache identity.
        atomic_write_json(path, payload, compact=True)
        if prune_due:
            total = _prune_semantic_cache(path.parent, current=path, max_bytes=max_bytes)
            _SEMANTIC_PRUNE_STATE[path.parent.resolve()] = (0, total)
        return True
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.warning("Unable to persist semantic content cache entry %s: %s", path, exc)
        return False


def _next_observation_number(project: dict[str, Any]) -> int:
    numbers = [
        int(value.removeprefix("VA"))
        for item in project.get("visual_observations", [])
        if (value := str(item.get("observation_id", ""))).startswith("VA")
        and value.removeprefix("VA").isdigit()
    ]
    return max(numbers, default=0) + 1


def _load_packet(path: Path) -> VisionPacket:
    """Load one packet through a bounded stat-keyed process-local cache."""

    info = path.stat()
    signature = (int(info.st_size), int(info.st_mtime_ns), int(info.st_ino))
    cached = _PACKET_CACHE.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    packet = VisionPacket.model_validate(payload)
    if len(_PACKET_CACHE) >= _PACKET_CACHE_LIMIT:
        _PACKET_CACHE.pop(next(iter(_PACKET_CACHE)))
    _PACKET_CACHE[path] = (signature, packet)
    return packet


def _loaded_packet_files(project_dir: Path) -> list[tuple[Path, VisionPacket]]:
    """Return valid semantic packets without reloading each packet twice.

    Packet discovery historically loaded every packet to validate its schema,
    then the caller loaded those same paths again.  ``_load_packet`` avoids
    JSON/Pydantic work on a cache hit, but each call still performs a ``stat``
    and a cache lookup.  Keeping the validated packet alongside its path lets
    selection and preflight share the one load while preserving the existing
    path-only helper for callers that only need filenames.
    """

    result: list[tuple[Path, VisionPacket]] = []
    for path in sorted((project_dir / ".state" / "vision" / "packets").glob("V*.json")):
        try:
            packet = _load_packet(path)
        except (
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            TypeError,
            ValueError,
            ValidationFailure,
        ):
            continue
        if packet.schema_name == "video-script-reconstructor.vision-packet":
            result.append((path, packet))
    return result


def _candidate_packet_paths(
    project_dir: Path,
    candidate_ids: set[str],
) -> list[Path] | None:
    """Resolve a filtered packet set without scanning unrelated packet JSON.

    A filtered semantic pass is used by host-agent bundle application, where
    ``candidate_ids`` usually contains only a handful of responses from a
    project with thousands of packets.  Generated packets use the candidate
    ID as their filename; resolve those exact safe names and return ``None``
    for malformed/missing IDs so callers can retain the complete legacy scan
    behavior for unusual layouts.
    """

    if not candidate_ids:
        return []
    packet_dir = project_dir / ".state" / "vision" / "packets"
    paths: list[Path] = []
    for candidate_id in candidate_ids:
        if (
            not candidate_id
            or "/" in candidate_id
            or "\\" in candidate_id
            or candidate_id in {".", ".."}
        ):
            return None
        path = packet_dir / f"{candidate_id}.json"
        try:
            if path.is_symlink() or not path.is_file():
                return None
        except OSError:
            return None
        paths.append(path)
    return sorted(paths, key=lambda path: (path.name.casefold(), path.name))


def _packet_files(project_dir: Path) -> list[Path]:
    """Return valid semantic packet paths in deterministic order."""

    return [path for path, _packet in _loaded_packet_files(project_dir)]


def _semantic_packet_budget(override: int | None = None) -> int | None:
    """Return an optional deterministic semantic-observation budget."""

    raw = str(override) if override is not None else os.environ.get("VSR_SEMANTIC_MAX_PACKETS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning("Ignoring invalid VSR_SEMANTIC_MAX_PACKETS=%r", raw)
        return None
    return None if value <= 0 else min(value, 100_000)


def _semantic_packet_score(
    packet: VisionPacket,
    event: Mapping[str, Any] | None = None,
) -> float:
    """Rank packets for bounded semantic review without changing evidence selection.

    Measured change regions are intentionally part of the priority score.  The
    bounded scheduler otherwise sees hundreds of packets with identical role,
    OCR, and question signals and spends early slots on timestamp order.  A
    high changed-ratio packet is more likely to contain a consequential visual
    transition, so resolving it earlier improves useful coverage without
    changing which packets exist or how canonical evidence is selected.
    """

    score = 0.0
    # Event-level importance is a stronger signal than packet order when a
    # bounded continuation is used.  Keep low-importance context eligible for
    # eventual review, but let consequential/high-impact evidence reach a
    # reviewer first.  This changes scheduling only; it never removes a
    # packet or changes the evidence selected by the visual stage.
    if isinstance(event, Mapping):
        importance = str(event.get("importance", "supporting")).casefold()
        score += {
            "incidental": 0.0,
            "supporting": 1.0,
            "consequential": 8.0,
            "high_impact": 12.0,
        }.get(importance, 1.0)
        if event.get("scene_or_state_id"):
            score += 2.0
        claim_ids = event.get("image_claim_ids")
        if isinstance(claim_ids, (list, tuple)):
            score += min(2.0, len(claim_ids) * 0.25)
    roles = {frame.role for frame in packet.frames}
    score += 2.0 * len(roles.intersection({"focus", "action", "result"}))
    # Survey/candidate producers use both ``scene_cut`` and ``scene-cut``
    # spellings across persisted projects. Normalize separators before
    # scoring so measured scene/adaptive/OCR signals affect bounded review
    # order without changing the selected evidence itself.
    reasons = str(packet.scene_motion_metadata.get("selection_reason", "")).casefold()
    reasons = reasons.replace("_", "-")
    for token, weight in (
        ("hard-scene", 8.0),
        ("scene-cut", 8.0),
        ("consequential", 5.0),
        ("ocr", 4.0),
        ("adaptive", 2.0),
    ):
        if token in reasons:
            score += weight
    if any(item.uncertain_characters for item in packet.raw_ocr):
        score += 4.0
    if any(item.normalized_interpretation.strip() for item in packet.raw_ocr):
        score += 3.0
    if any("consequential" in question.casefold() for question in packet.questions):
        score += 3.0
    change_ratios = [
        float(region.changed_ratio)
        for frame in packet.frames
        for region in frame.difference_regions
    ]
    if change_ratios:
        max_change = max(change_ratios)
        score += min(6.0, max_change * 6.0)
        # A small measured change can be more important than a large static
        # region: one changed character, badge, or pixel cluster may carry the
        # only consequential state transition. Keep a bounded floor for tiny
        # non-zero regions so they are not starved by low-value context.
        if 0.0 < max_change <= 0.05:
            score += 3.0
    return score


def _semantic_visual_reuse_key(packet: VisionPacket) -> str:
    """Key only the visual evidence/question scope for safe in-run deduplication."""

    payload = {
        "frames": [
            {"frame_id": frame.frame_id, "path": frame.path, "role": frame.role}
            for frame in packet.frames
        ],
        "questions": list(packet.questions),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _semantic_packet_scope_key(packet: VisionPacket) -> str:
    """Return an exact packet-scope key for bounded scheduler deduplication.

    Candidate IDs are event identities rather than evidence scope.  Every
    other packet field remains in the key, including the complete ordered
    frame set (IDs, paths, roles, timing, and crops), questions, OCR, and
    neighbouring/context events.  In particular, packets sharing a focus
    frame but carrying a different after/action frame must not be collapsed:
    their state-change question is different even when the focus pixels match.
    """

    payload = packet.model_dump(mode="json")
    payload.pop("candidate_id", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _semantic_event_is_observed(event: Mapping[str, Any] | None) -> bool:
    """Return whether the packet's own event has a persisted observer result.

    A frame can be cited by several packets.  Its ``semantic_status`` therefore
    cannot prove that every packet sharing that focus was observed: a different
    after/action/crop/context scope may still need independent semantic work.
    The event-level provider link is the authoritative marker for this skip.
    """

    return isinstance(event, Mapping) and bool(str(event.get("annotation_provider") or "").strip())


_SEMANTIC_CONTEXT_ID_KEYS = {
    "candidate_id",
    "event_id",
    "frame_id",
    "image_id",
    "observation_id",
    "segment_id",
    "block_id",
    "claim_id",
    "revision_id",
}


def _semantic_content_context(value: Any, *, key: str | None = None) -> Any:
    """Project prompt context while removing run-specific identity values.

    Content reuse intentionally ignores candidate/frame IDs so exact visual
    pixels can be reused across projects.  It must still include the textual,
    OCR, scene, and uncertainty context the model sees; otherwise a cache hit
    could silently answer a different question.  IDs are replaced with stable
    placeholders, preserving list lengths and semantic text without binding the
    result to one project's sequential identifiers.
    """

    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            field = str(raw_key)
            lowered = field.casefold()
            if lowered in _SEMANTIC_CONTEXT_ID_KEYS:
                projected[field] = "<id>"
            elif lowered.endswith("_ids") and isinstance(raw_value, Sequence) and not isinstance(
                raw_value, (str, bytes, bytearray)
            ):
                projected[field] = ["<id>" for _ in raw_value]
            else:
                projected[field] = _semantic_content_context(raw_value, key=field)
        return projected
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_semantic_content_context(item) for item in value]
    return value


def _semantic_visual_content_key(
    packet: VisionPacket,
    frame_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]] | None:
    """Build a content-addressed key for exact visual reuse within one pass.

    Candidate IDs and extracted frame IDs are intentionally absent from this
    key.  Reuse is nevertheless conservative: every packet frame must have a
    canonical decoded-pixel (or whole-file) digest, frame roles and order must
    match, and the observer questions must be identical.  A caller can then
    remap the source annotation's cited frame IDs to the target packet and run
    the normal packet validator before committing it.
    """

    frame_signatures: list[dict[str, str]] = []
    source_ids: list[str] = []
    for reference in packet.frames:
        source_ids.append(reference.frame_id)
        frame = frame_by_id.get(reference.frame_id)
        if not isinstance(frame, Mapping):
            return None
        pixel_hash = frame.get("pixel_hash")
        if not isinstance(pixel_hash, Mapping):
            metadata = frame.get("metadata")
            image_metadata = metadata.get("image") if isinstance(metadata, Mapping) else None
            pixel_hash = (
                image_metadata.get("pixel_hash")
                if isinstance(image_metadata, Mapping)
                else None
            )
        digest = pixel_hash.get("value") if isinstance(pixel_hash, Mapping) else None
        if not isinstance(digest, str) or not digest:
            digest = frame.get("file_hash")
        if not isinstance(digest, str) or not digest:
            return None
        frame_signatures.append({"role": reference.role, "digest": digest})
    payload = {
        "frames": frame_signatures,
        "questions": list(packet.questions),
        "max_span_ms": packet.max_span_ms,
        "nearby_transcript": _semantic_content_context(packet.nearby_transcript),
        "raw_ocr": _semantic_content_context(
            [item.model_dump(mode="json") for item in packet.raw_ocr]
        ),
        "scene_motion_metadata": _semantic_content_context(packet.scene_motion_metadata),
        "prior_event_context": _semantic_content_context(packet.prior_event_context),
        "next_event_context": _semantic_content_context(packet.next_event_context),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), tuple(source_ids)


def _deterministic_identical_frame_annotation(
    packet: VisionPacket,
    frame_by_id: Mapping[str, Mapping[str, Any]],
) -> VisionAnnotation | None:
    """Return a conservative no-change annotation for pixel-identical packets.

    Exact decoded-pixel hashes are stronger evidence for a no-change decision
    than a second multimodal request.  This fast path is deliberately narrow:
    every supplied frame must expose a canonical pixel hash and at least two
    frames must match exactly.  It makes no claim about text, identity, cause,
    or hidden state, and therefore cannot turn uncertain OCR into a supported
    fact.  Missing hashes or even one changed frame fall through to the VLM.
    """

    if len(packet.frames) < 2:
        return None
    hashes: list[str] = []
    for reference in packet.frames:
        frame = frame_by_id.get(reference.frame_id)
        if not isinstance(frame, Mapping):
            return None
        metadata = frame.get("metadata")
        image_metadata = metadata.get("image") if isinstance(metadata, Mapping) else None
        pixel_hash = image_metadata.get("pixel_hash") if isinstance(image_metadata, Mapping) else None
        value = pixel_hash.get("value") if isinstance(pixel_hash, Mapping) else None
        if not isinstance(value, str) or not value:
            return None
        hashes.append(value)
    if len(set(hashes)) != 1:
        return None
    return VisionAnnotation(
        candidate_id=packet.candidate_id,
        factual_visible_description=(
            "The supplied evidence frames are pixel-identical; no visible state "
            "change is observed across the sequence."
        ),
        event_type="no_change",
        evidence_frame_ids=[frame.frame_id for frame in packet.frames],
        before_action_after_roles={},
        exact_visible_text_candidates=[],
        consequential_changes=[],
        confidence=1.0,
        uncertainty=[
            "Pixel-identical evidence establishes no visible pixel change; it does not "
            "establish hidden state, causality, or off-screen events."
        ],
        statements_not_inferred=[
            "No text, identity, intent, cause, or hidden state is inferred from pixel identity."
        ],
    )


def _deterministic_stable_frame_annotation(
    packet: VisionPacket,
    frame_by_id: Mapping[str, Mapping[str, Any]],
) -> VisionAnnotation | None:
    """Conservatively classify stable visible state without a provider call.

    This weaker fast path requires at least two frames, matching persisted
    dHashes, no packet difference regions, and stable/uncertain-free OCR. It
    deliberately makes no claim of pixel identity, text completeness, speech,
    intent, causality, or hidden state; any missing or conflicting signal falls
    through to normal semantic review.
    """

    if len(packet.frames) < 2:
        return None
    dhashes: list[str] = []
    for reference in packet.frames:
        frame = frame_by_id.get(reference.frame_id)
        if not isinstance(frame, Mapping) or reference.difference_regions:
            return None
        perceptual = frame.get("perceptual_hashes")
        if not isinstance(perceptual, Mapping):
            return None
        if (
            perceptual.get("dhash-8-algorithm") != PERCEPTUAL_DHASH_ALGORITHM
            or perceptual.get("dhash-8-verified") != PERCEPTUAL_DHASH_VERIFIED
        ):
            # Legacy projects may contain hashes written before the frame-local
            # dHash enrichment fix. Never use those unverifiable values to
            # bypass provider review.
            return None
        dhash = perceptual.get("dhash-8")
        if not isinstance(dhash, str) or not dhash:
            return None
        dhashes.append(dhash)
    if len(set(dhashes)) != 1:
        return None
    normalized_ocr = [
        normalize_ocr_for_comparison(item.normalized_interpretation)
        for item in packet.raw_ocr
    ]
    if any(item.uncertain_characters for item in packet.raw_ocr):
        return None
    if normalized_ocr and len(set(normalized_ocr)) != 1:
        return None
    return VisionAnnotation(
        candidate_id=packet.candidate_id,
        factual_visible_description=(
            "The supplied evidence frames show a stable visible state across the "
            "sequence; no measured scene-difference region is recorded."
        ),
        event_type="stable_visible_state",
        evidence_frame_ids=[frame.frame_id for frame in packet.frames],
        before_action_after_roles={},
        exact_visible_text_candidates=[],
        consequential_changes=[],
        confidence=0.98,
        uncertainty=[
            "Perceptual and OCR stability support a stable visible state but do not "
            "establish pixel identity, text completeness, causality, or hidden state."
        ],
        statements_not_inferred=[
            "No identity, intent, cause, speech, or hidden state is inferred from stable evidence."
        ],
    )


def _upgrade_semantic_pending_annotation(
    annotation: VisionAnnotation,
    packet: VisionPacket,
    frame_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[VisionAnnotation, bool]:
    """Replace a pending response only when deterministic pixels prove stability.

    File-based Codex/subagent review responses are intentionally allowed to be
    conservative ``semantic_pending`` markers.  A bounded deterministic proof
    can still be stronger than that marker when every packet frame is
    pixel-identical or has verified stable dHash/OCR evidence.  The proof never
    fabricates text, identity, motion, or causality; it cites the packet frames
    generated by the existing strict deterministic helpers.
    """

    if annotation.event_type != "semantic_pending" or annotation.confidence != 0:
        return annotation, False
    identical = _deterministic_identical_frame_annotation(packet, frame_by_id)
    if identical is not None:
        return identical, True
    stable = _deterministic_stable_frame_annotation(packet, frame_by_id)
    if stable is not None:
        return stable, True
    return annotation, False


def _remap_reused_annotation(
    annotation: VisionAnnotation,
    *,
    source_frame_ids: Sequence[str],
    packet: VisionPacket,
) -> VisionAnnotation:
    """Move an exact-content annotation onto a packet with new frame IDs."""

    target_frame_ids = tuple(frame.frame_id for frame in packet.frames)
    if len(source_frame_ids) != len(target_frame_ids):
        raise ValidationFailure("Exact visual reuse frame cardinality differs")
    frame_map = dict(zip(source_frame_ids, target_frame_ids, strict=True))
    payload = annotation.model_dump(mode="json")
    payload["candidate_id"] = packet.candidate_id
    payload["evidence_frame_ids"] = [
        frame_map.get(str(frame_id), str(frame_id))
        for frame_id in payload.get("evidence_frame_ids", [])
    ]
    payload["before_action_after_roles"] = {
        frame_map.get(str(frame_id), str(frame_id)): role
        for frame_id, role in payload.get("before_action_after_roles", {}).items()
    }
    for item in payload.get("exact_visible_text_candidates", []):
        if isinstance(item, dict) and item.get("frame_id") is not None:
            item["frame_id"] = frame_map.get(str(item["frame_id"]), str(item["frame_id"]))
    for change in payload.get("consequential_changes", []):
        if not isinstance(change, dict):
            continue
        change["action_frame_ids"] = [
            frame_map.get(str(frame_id), str(frame_id))
            for frame_id in change.get("action_frame_ids", [])
        ]
        if change.get("before_frame_id") is not None:
            change["before_frame_id"] = frame_map.get(
                str(change["before_frame_id"]), str(change["before_frame_id"])
            )
        change["after_frame_ids"] = [
            frame_map.get(str(frame_id), str(frame_id))
            for frame_id in change.get("after_frame_ids", [])
        ]
    return validate_annotation_for_packet(payload, packet)


def _is_retryable_http400_fallback(project_dir: Path, candidate_id: str) -> bool:
    """Return whether a persisted fallback records one transient local HTTP 400."""

    if not candidate_id or "/" in candidate_id or "\\" in candidate_id:
        return False
    annotation_path = (
        project_dir
        / ".state"
        / "vision"
        / "annotations"
        / f"{candidate_id}.annotation.json"
    )
    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(annotation, Mapping) or annotation.get("event_type") != "semantic_pending":
        return False
    uncertainty = annotation.get("uncertainty", [])
    return any("HTTP Error 400" in str(item) for item in uncertainty)


def _semantic_failure_is_provider_health_failure(error: BaseException | str) -> bool:
    """Return whether a semantic failure indicates a shared provider outage.

    A semantic packet can fail for reasons that are local to that packet: a
    multimodal context 400, truncated JSON, or an annotation that omitted a
    required frame citation.  Opening a global circuit for those errors turns
    the remaining queue into unattempted review-only fallbacks even though the
    loopback model may be healthy for the next packet.  Only failures that
    indicate the provider itself is unavailable or repeatedly unhealthy should
    trip the circuit breaker.  The conservative fallback is still persisted
    for every packet-local failure, and the existing explicit retry queue can
    revisit transient 400s later.
    """

    text = str(error).casefold()
    health_markers = (
        "connection refused",
        "connection reset",
        "connection aborted",
        "connection timed out",
        "timed out",
        "timeout",
        "http error 500",
        "http error 502",
        "http error 503",
        "http error 504",
        "server disconnected",
        "server exited",
        "provider unavailable",
    )
    return any(marker in text for marker in health_markers)


def _is_retryable_semantic_pending(
    project_dir: Path,
    candidate_id: str,
    *,
    prompt_template_hash: str | None,
) -> bool:
    """Select old semantic-pending markers after a prompt revision.

    This is opt-in because a semantic-pending marker is a valid conservative
    result.  Reprocessing is justified only when the persisted observation was
    produced by a different prompt revision, so a deterministic provider does
    not repeatedly spend inference time on the same unresolved packet.
    """

    if not candidate_id or "/" in candidate_id or "\\" in candidate_id:
        return False
    if not prompt_template_hash:
        return False
    annotation_path = (
        project_dir
        / ".state"
        / "vision"
        / "annotations"
        / f"{candidate_id}.annotation.json"
    )
    observation_path = (
        project_dir
        / ".state"
        / "vision"
        / "annotations"
        / f"{candidate_id}.observation.json"
    )
    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(annotation, Mapping) or not isinstance(observation, Mapping):
        return False
    if annotation.get("event_type") != "semantic_pending":
        return False
    return str(observation.get("prompt_template_hash") or "") != prompt_template_hash


def _retryable_semantic_counts(
    project_dir: Path,
    *,
    prompt_template_hash: str | None,
    include_http400: bool,
    include_semantic_pending: bool,
) -> tuple[int, int]:
    """Count retryable markers with one annotation-directory scan.

    Batch preflight may need both retry queues.  The old independent counters
    each globbed and decoded every annotation, and the prompt-refresh counter
    decoded the same annotation once more before reading its observation.  A
    single pass preserves the exact predicates while avoiding duplicate JSON
    reads when both retry modes are enabled.
    """

    if not include_http400 and not include_semantic_pending:
        return 0, 0
    pending_enabled = include_semantic_pending and bool(prompt_template_hash)
    if not pending_enabled and not include_http400:
        return 0, 0

    annotation_root = project_dir / ".state" / "vision" / "annotations"
    fallback_count = 0
    pending_count = 0
    for path in annotation_root.glob("V*.annotation.json"):
        candidate_id = path.name.removesuffix(".annotation.json")
        if not candidate_id or "/" in candidate_id or "\\" in candidate_id:
            continue
        try:
            annotation = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(annotation, Mapping):
            continue
        if annotation.get("event_type") != "semantic_pending":
            continue
        if include_http400:
            uncertainty = annotation.get("uncertainty", [])
            if any("HTTP Error 400" in str(item) for item in uncertainty):
                fallback_count += 1
        if pending_enabled:
            observation_path = annotation_root / f"{candidate_id}.observation.json"
            try:
                observation = json.loads(observation_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(observation, Mapping):
                continue
            if str(observation.get("prompt_template_hash") or "") != prompt_template_hash:
                pending_count += 1
    return fallback_count, pending_count


def _retryable_semantic_pending_count(
    project_dir: Path,
    *,
    prompt_template_hash: str | None,
) -> int:
    """Count markers eligible for an explicit prompt refresh."""

    return _retryable_semantic_counts(
        project_dir,
        prompt_template_hash=prompt_template_hash,
        include_http400=False,
        include_semantic_pending=True,
    )[1]


def _retryable_http400_fallback_count(project_dir: Path) -> int:
    """Count persisted local HTTP-400 fallbacks eligible for explicit retry."""

    return _retryable_semantic_counts(
        project_dir,
        prompt_template_hash=None,
        include_http400=True,
        include_semantic_pending=False,
    )[0]


def _semantic_pending_retry_priority(project_dir: Path, candidate_id: str) -> int:
    """Rank stale pending records so contract violations are repaired first.

    Older adapters occasionally persisted ``semantic_pending`` with a positive
    confidence even though the event is explicitly unresolved.  Both that
    record and a confidence-zero pending marker are safe to refresh after a
    prompt revision, but the former is a higher-value repair and should not be
    left behind while a bounded scheduler spends its budget on already-valid
    conservative markers.
    """

    if not candidate_id or "/" in candidate_id or "\\" in candidate_id:
        return 0
    path = (
        project_dir
        / ".state"
        / "vision"
        / "annotations"
        / f"{candidate_id}.annotation.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, Mapping) or payload.get("event_type") != "semantic_pending":
        return 0
    try:
        return 2 if float(payload.get("confidence", 0.0)) > 0 else 1
    except (TypeError, ValueError):
        return 1


def _prior_claim_context_ids(project: Mapping[str, Any], packet: VisionPacket) -> list[str]:
    """Return bounded claim IDs visible to a cumulative semantic observer."""

    frame_ids = {frame.frame_id for frame in packet.frames}
    claim_ids: list[str] = []
    for frame in project.get("frames", []):
        if not isinstance(frame, Mapping) or str(frame.get("frame_id")) not in frame_ids:
            continue
        metadata = frame.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        knowledge = metadata.get("knowledge")
        knowledge = knowledge if isinstance(knowledge, Mapping) else {}
        for key in (
            "supported_claim_ids",
            "disputed_claim_ids",
            "unresolved_claim_ids",
        ):
            values = knowledge.get(key)
            if isinstance(values, (list, tuple)):
                claim_ids.extend(str(value) for value in values if value)
        claims = knowledge.get("claims")
        if isinstance(claims, list):
            for claim in claims:
                if isinstance(claim, Mapping) and claim.get("claim_id"):
                    claim_ids.append(str(claim["claim_id"]))
    # A project may have current claim records even when an older frame payload
    # has not yet been fully mirrored. Include only claims explicitly tied to a
    # packet frame, preserving the same bounded context contract.
    for claim in project.get("image_claims", []):
        if not isinstance(claim, Mapping) or not claim.get("claim_id"):
            continue
        supporting = claim.get("supporting_image_ids", [])
        if isinstance(supporting, list) and frame_ids.intersection(str(item) for item in supporting):
            claim_ids.append(str(claim["claim_id"]))
    return list(dict.fromkeys(claim_ids))[:128]


def _semantic_worker_count(override: int | None = None) -> int:
    """Resolve the bounded provider fan-out used by continuation passes.

    A single worker is intentionally the safe default for small-memory hosts.
    The local Qwen3-VL server is only configured for one or two slots, so an
    accidental environment value cannot turn a long run into an OOM storm.
    """

    raw = str(override) if override is not None else os.environ.get("VSR_SEMANTIC_WORKERS", "1")
    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning("Ignoring invalid semantic worker count %r", raw)
        return 1
    return max(1, min(2, value))


@dataclass(frozen=True)
class _ParallelSemanticAnnotation:
    """Provider-side result kept separate from canonical commit ordering."""

    packet: VisionPacket
    image_id: str
    frame: dict[str, Any]
    observation_id: str
    annotation: VisionAnnotation
    cache_hit: bool
    cache_miss: int
    cache_write: int
    deterministic_no_change: bool
    provider_failure: dict[str, Any] | None


def _annotate_parallel_packet(
    packet: VisionPacket,
    *,
    image_id: str,
    frame: dict[str, Any],
    observation_id: str,
    project_dir: Path,
    project: dict[str, Any],
    provider: VisionProvider,
    allow_deterministic_stable: bool = False,
    semantic_cache_root: Path | None,
    semantic_cache_limit: int,
    frame_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> _ParallelSemanticAnnotation:
    """Run one independent provider/cache operation for the bounded worker pool."""

    cache_key_value: str | None = None
    cache_path: Path | None = None
    frame_hashes: dict[str, str] = {}
    cache_hit = False
    cache_miss = 0
    cache_write = 0
    deterministic_no_change = False
    annotation: VisionAnnotation | None = None
    provider_failure: dict[str, Any] | None = None
    if frame_by_id is None:
        # Preserve the helper's historical direct-call surface for integrations
        # and unit fixtures. Production batch callers pass the shared index once
        # so this O(frame-count) construction is not repeated per packet.
        frame_by_id = {
            str(item.get("frame_id")): item
            for item in project.get("frames", [])
            if isinstance(item, Mapping) and item.get("frame_id")
        }
    if semantic_cache_root is not None:
        try:
            frame_hashes = _packet_frame_hashes(
                project,
                packet,
                project_dir,
                frame_by_id=frame_by_id,
            )
            cache_key_value = _semantic_cache_key(provider, packet, frame_hashes)
            cache_path = semantic_cache_root / f"{cache_key_value}.json"
            annotation = _read_semantic_cache(
                cache_path,
                key=cache_key_value,
                packet=packet,
                max_bytes=semantic_cache_limit,
            )
            if annotation is not None:
                cache_hit = True
            else:
                cache_miss = 1
        except (OSError, ValidationFailure, TypeError, ValueError) as exc:
            # Cache inspection is an acceleration layer only. A malformed or
            # unavailable cache must never prevent a fresh observation.
            LOGGER.info("Semantic cache lookup skipped for %s: %s", packet.candidate_id, exc)
            cache_miss = 1
    if annotation is None:
        annotation = _deterministic_identical_frame_annotation(packet, frame_by_id)
        if annotation is None and allow_deterministic_stable:
            annotation = _deterministic_stable_frame_annotation(packet, frame_by_id)
        deterministic_no_change = annotation is not None
    if annotation is None:
        try:
            annotation = _annotate_with_transport_identity(
                provider,
                packet,
                project_dir=project_dir,
                frame_hashes=frame_hashes,
            )
        except ValidationFailure as exc:
            provider_failure = {
                "candidate_id": packet.candidate_id,
                "error": str(exc)[:1000],
                "provider_attempted": True,
                "fallback": True,
            }
            annotation = _fallback_annotation(packet, exc)
        if provider_failure is None and cache_path is not None and cache_key_value is not None:
            if not frame_hashes:
                frame_hashes = _packet_frame_hashes(
                    project,
                    packet,
                    project_dir,
                    frame_by_id=frame_by_id,
                )
            if _write_semantic_cache(
                cache_path,
                key=cache_key_value,
                provider=provider,
                packet=packet,
                annotation=annotation,
                frame_hashes=frame_hashes,
                max_bytes=semantic_cache_limit,
            ):
                cache_write = 1
    return _ParallelSemanticAnnotation(
        packet=packet,
        image_id=image_id,
        frame=frame,
        observation_id=observation_id,
        annotation=annotation,
        cache_hit=cache_hit,
        cache_miss=cache_miss,
        cache_write=cache_write,
        deterministic_no_change=deterministic_no_change,
        provider_failure=provider_failure,
    )


def _select_semantic_packet_files(
    project_dir: Path,
    *,
    semantic_max_packets: int | None = None,
    project: Mapping[str, Any] | None = None,
    retry_fallbacks: bool = False,
    retry_semantic_pending: bool = False,
    prompt_template_hash: str | None = None,
    candidate_ids: set[str] | None = None,
    deterministic_only: bool = False,
    allow_observed_candidate_ids: bool = False,
    allow_deterministic_stable: bool = False,
) -> tuple[list[Path], list[str]]:
    """Select pending packets and bound only the expensive provider work.

    Pixel-identical packets are deterministic no-change observations, so they
    are selected outside the VLM budget.  This prevents a long static section
    from consuming the same inference quota as genuinely changing evidence;
    the normal packet validation and canonical commit path still applies.
    """

    loaded_project = project if project is not None else _load_project(project_dir)
    frame_by_id = {
        str(frame.get("frame_id")): frame
        for frame in loaded_project.get("frames", [])
        if isinstance(frame, Mapping) and frame.get("frame_id")
    }
    event_by_id = {
        str(event.get("event_id")): event
        for event in loaded_project.get("visual_events", [])
        if isinstance(event, Mapping) and event.get("event_id")
    }
    parsed: list[tuple[Path, VisionPacket, bool, bool]] = []
    candidate_paths = (
        _candidate_packet_paths(project_dir, candidate_ids)
        if candidate_ids is not None
        else None
    )
    loaded_packets: list[tuple[Path, VisionPacket]]
    if candidate_paths is None:
        loaded_packets = _loaded_packet_files(project_dir)
    else:
        loaded_packets = []
        for path in candidate_paths:
            try:
                packet = _load_packet(path)
            except (
                OSError,
                json.JSONDecodeError,
                UnicodeDecodeError,
                TypeError,
                ValueError,
                ValidationFailure,
            ):
                continue
            if packet.schema_name == "video-script-reconstructor.vision-packet":
                loaded_packets.append((path, packet))
    for path, packet in loaded_packets:
        if candidate_ids is not None and packet.candidate_id not in candidate_ids:
            continue
        retryable_fallback = retry_fallbacks and _is_retryable_http400_fallback(
            project_dir, packet.candidate_id
        )
        retryable_pending = retry_semantic_pending and _is_retryable_semantic_pending(
            project_dir,
            packet.candidate_id,
            prompt_template_hash=prompt_template_hash,
        )
        retryable = retryable_fallback or retryable_pending
        event = event_by_id.get(packet.candidate_id)
        selected_for_review = (
            candidate_ids is not None
            and packet.candidate_id in candidate_ids
            and (
                allow_observed_candidate_ids
                or (
                    isinstance(event, Mapping)
                    and event.get("event_type") == "semantic_pending"
                )
            )
        )
        if _semantic_event_is_observed(event) and not retryable and not selected_for_review:
            continue
        deterministic_no_change = (
            _deterministic_identical_frame_annotation(packet, frame_by_id) is not None
            or (
                allow_deterministic_stable
                and _deterministic_stable_frame_annotation(packet, frame_by_id) is not None
            )
        )
        if deterministic_only and not deterministic_no_change:
            continue
        parsed.append((path, packet, retryable, deterministic_no_change))
    if not parsed:
        return [], []
    if deterministic_only:
        # Deterministic no-change observations do not consume a provider
        # budget.  Commit every pending event, including exact packet-scope
        # aliases: scope deduplication is useful for expensive semantic work,
        # but each event still needs its own persisted observer link.
        return [path for path, _packet, _retryable, _deterministic in parsed], []
    packet_scope_keys = [
        _semantic_packet_scope_key(packet)
        for _path, packet, _retryable, _deterministic in parsed
    ]
    budget = _semantic_packet_budget(semantic_max_packets)
    if budget is None or budget >= len(parsed):
        return [path for path, _packet, _retryable, _deterministic in parsed], []
    # Evenly spaced anchors preserve temporal coverage; the remaining slots are
    # filled by the highest-value scene/OCR/change packets with stable ties.
    # Deterministic no-change packets are included in addition to that budget.
    deterministic_indices = {
        index
        for index, (_path, _packet, _retryable, deterministic) in enumerate(parsed)
        if deterministic
    }
    # Deterministic no-change packets do not consume provider budget, so keep
    # every pending event—even exact-scope aliases—rather than deferring them
    # for another continuation round.  The expensive budget remains reserved
    # for one representative per packet scope below; aliases that require a
    # provider call still share a validated annotation at commit time.
    scope_indices: dict[str, list[int]] = {}
    for index, scope_key in enumerate(packet_scope_keys):
        scope_indices.setdefault(scope_key, []).append(index)
    deterministic_scope_keys = {
        packet_scope_keys[index] for index in deterministic_indices
    }
    retryable_indices = {
        index
        for index, (_path, _packet, retryable, deterministic) in enumerate(parsed)
        if retryable and not deterministic
    }
    selected_indices: set[int] = set()
    selected_scope_keys: set[str] = set()
    selected_expensive_scope_keys: set[str] = set()

    def add_scope(index: int) -> bool:
        """Select one scope and all aliases, returning whether it was new."""

        scope_key = packet_scope_keys[index]
        if scope_key in selected_scope_keys:
            return False
        selected_scope_keys.add(scope_key)
        selected_indices.update(scope_indices[scope_key])
        if scope_key not in deterministic_scope_keys:
            selected_expensive_scope_keys.add(scope_key)
        return True

    for index in sorted(deterministic_indices):
        # An exact-scope alias is still an independent event ledger entry. It
        # must receive its own deterministic observation even though the
        # expensive semantic scope is identical.
        selected_indices.add(index)
        selected_scope_keys.add(packet_scope_keys[index])

    retryable_order = sorted(
        retryable_indices,
        key=lambda index: (
            -_semantic_pending_retry_priority(project_dir, parsed[index][1].candidate_id),
            -_semantic_packet_score(
                parsed[index][1], event_by_id.get(parsed[index][1].candidate_id)
            ),
            index,
        ),
    )
    for index in retryable_order:
        if len(selected_expensive_scope_keys) >= budget:
            break
        add_scope(index)
    if len(selected_expensive_scope_keys) < budget:
        # When no retry work consumes the budget, reserve roughly half the
        # expensive slots for temporal anchors and let measured visual-change
        # ranking choose the other half.  The previous implementation filled
        # every slot with anchors, making the ranking branch unreachable for
        # ordinary continuation passes.
        remaining = budget - len(selected_expensive_scope_keys)
        anchor_count = min(remaining, max(1, budget // 2))
        anchors = [
            round(index * (len(parsed) - 1) / max(1, anchor_count - 1))
            for index in range(anchor_count)
        ]
        for anchor in anchors:
            add_scope(anchor)
            if len(selected_expensive_scope_keys) >= budget:
                break
    ranked = sorted(
        range(len(parsed)),
        key=lambda index: (
            -_semantic_packet_score(
                parsed[index][1], event_by_id.get(parsed[index][1].candidate_id)
            ),
            index,
        ),
    )
    for index in ranked:
        if len(selected_expensive_scope_keys) >= budget:
            break
        add_scope(index)
    selected = [parsed[index][0] for index in sorted(selected_indices)]
    deferred = [
        parsed[index][1].candidate_id
        for index in range(len(parsed))
        if index not in selected_indices
    ]
    return selected, deferred


def pending_packet_count(project_dir: Path) -> int:
    """Count packet events that still need their own semantic observation.

    This preflight is intentionally the same deterministic status check used
    by :func:`apply_vision_provider`. It lets the public pipeline avoid
    starting a multi-gigabyte local vision server when a changed/retried run
    already contains complete semantic observations; no canonical files are
    modified by the check.
    """

    return len(_pending_semantic_event_ids(project_dir))


def _pending_semantic_event_ids(
    project_dir: Path,
    project: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return packet IDs whose event has no persisted semantic observer link.

    Filtered continuation passes (for example a partially completed subagent
    review bundle) must not mistake an empty *filtered* deferred list for an
    empty project-wide frontier.  Keeping this helper alongside
    :func:`pending_packet_count` gives those passes an inexpensive, event-scoped
    reconciliation source after their selected observations commit.
    """

    loaded_project = project if project is not None else _load_project(project_dir)
    event_by_id = {
        str(event.get("event_id")): event
        for event in loaded_project.get("visual_events", [])
        if isinstance(event, Mapping) and event.get("event_id")
    }
    pending: list[str] = []
    for _packet_path, packet in _loaded_packet_files(project_dir):
        if not _semantic_event_is_observed(event_by_id.get(packet.candidate_id)):
            pending.append(packet.candidate_id)
    return pending


def prepare_host_agent_handoffs(project_dir: Path) -> list[str]:
    provider = HostAgentVisionProvider()
    written: list[str] = []
    for _packet_path, packet in _loaded_packet_files(project_dir):
        request = provider.prepare_handoff(packet, project_root=project_dir)
        written.append(request.relative_to(project_dir).as_posix())
    return written


def _apply_semantic_links(
    project: dict[str, Any],
    *,
    provider: VisionProvider,
    packet: VisionPacket,
    annotation_event_type: str,
    annotation_description: str,
    annotation_roles: dict[str, str],
    annotation_confidence: float,
    annotation_uncertainty: list[str],
    image_id: str,
    revision_id: str,
    supported_claim_ids: list[str],
    sufficiency_status: str,
    event_by_id: Mapping[str, dict[str, Any]] | None = None,
    blocks_by_frame_id: Mapping[str, Sequence[dict[str, Any]]] | None = None,
    reviews_by_frame_id: Mapping[str, Sequence[dict[str, Any]]] | None = None,
) -> None:
    """Apply provider links to an already-loaded canonical project.

    Keeping this mutation pure lets transactional observation ingestion fold
    semantic links into the same canonical write.  The legacy wrapper below
    still provides the public/private single-observation behavior.
    """
    descriptor = provider.descriptor
    roles: dict[str, list[str]] = {}
    for frame_id, role in annotation_roles.items():
        roles.setdefault(role, []).append(frame_id)
    event_candidates: Sequence[dict[str, Any]] = (
        [event_by_id[packet.candidate_id]]
        if event_by_id is not None and packet.candidate_id in event_by_id
        else []
        if event_by_id is not None
        else project.get("visual_events", [])
    )
    for event in event_candidates:
        if event_by_id is None and str(event.get("event_id")) != packet.candidate_id:
            continue
        event.update(
            {
                "event_type": annotation_event_type,
                "factual_grounded_description": annotation_description,
                "before_action_after_roles": roles,
                "confidence": annotation_confidence,
                "uncertainty": annotation_uncertainty,
                "annotation_provider": descriptor.provider_id,
                "review_status": (
                    "review_required"
                    if annotation_event_type == "semantic_pending"
                    and annotation_confidence <= 0
                    and any(
                        "no visual fact is asserted" in item.casefold()
                        for item in annotation_uncertainty
                    )
                    else "automatically_checked"
                ),
                "image_claim_ids": supported_claim_ids,
            }
        )
        revision_ids = event.setdefault("metadata_revision_ids", [])
        if revision_id not in revision_ids:
            revision_ids.append(revision_id)
    block_candidates: Sequence[dict[str, Any]] = (
        blocks_by_frame_id.get(image_id, [])
        if blocks_by_frame_id is not None
        else project.get("script_blocks", [])
    )
    for block in block_candidates:
        if blocks_by_frame_id is None and image_id not in block.get("frame_ids", []):
            continue
        block["uncertainty"] = [
            item
            for item in block.get("uncertainty", [])
            if "semantic visual analysis is pending" not in str(item).casefold()
        ]
        if supported_claim_ids and sufficiency_status == "sufficient":
            block["verification_status"] = "automatically_checked"
    if supported_claim_ids and sufficiency_status == "sufficient":
        review_candidates: Sequence[dict[str, Any]] = (
            reviews_by_frame_id.get(image_id, [])
            if reviews_by_frame_id is not None
            else project.get("review_items", [])
        )
        for review in review_candidates:
            if (
                review.get("category") == "visual_semantic_annotation"
                and (reviews_by_frame_id is not None or image_id in review.get("frame_ids", []))
                and not review.get("decision")
            ):
                review.update(
                    {
                        "decision": "resolved_by_semantic_observation",
                        "reviewer": descriptor.provider_id,
                        "decision_timestamp_utc": project.get("generated_at_utc"),
                        "rationale": (
                            "A schema-validated local semantic observation produced current "
                            "pixel-grounded claims that answered the scoped evidence question."
                        ),
                    }
                )


def _sync_semantic_budget_review_item(
    project: dict[str, Any],
    deferred_event_ids: Sequence[str],
    *,
    provider_id: str,
) -> bool:
    """Keep one live review marker for the moving semantic budget frontier.

    A bounded continuation changes the set of packet IDs still awaiting an
    observer.  Treating every frontier as a new unresolved review item makes
    canonical state grow quadratically and leaves already-observed packet IDs
    looking unresolved.  Budget markers are operational, automatic work
    scheduling records rather than observations or human decisions, so the
    latest unresolved marker is updated in place.  Existing decided markers
    remain untouched; all factual observations and review corrections stay
    append-only elsewhere in the canonical project.
    """

    review_items = project.setdefault("review_items", [])
    if not isinstance(review_items, list):
        raise ValidationFailure("Canonical review_items must be a list")
    active = [
        item
        for item in review_items
        if isinstance(item, dict)
        and item.get("category") == "semantic_budget_deferred"
        and not item.get("decision")
    ]
    deferred = list(dict.fromkeys(str(value) for value in deferred_event_ids))
    if not deferred:
        # Once the frontier is empty, any remaining unresolved budget marker is
        # stale.  Remove only this automatic scheduling category; observation-
        # specific semantic uncertainty remains represented by its own review
        # items and continues to block factual verification as appropriate.
        if not active:
            return False
        review_items[:] = [item for item in review_items if item not in active]
        return True

    changed = False
    if active:
        target = active[-1]
        # Collapse duplicate unresolved markers produced by older versions or
        # interrupted runs while preserving any user-decided history.
        if len(active) > 1:
            review_items[:] = [
                item for item in review_items if item not in active[:-1]
            ]
            changed = True
    else:
        review_numbers = [
            int(str(item.get("review_id", "R0")).removeprefix("R"))
            for item in review_items
            if isinstance(item, dict)
            and str(item.get("review_id", "")).removeprefix("R").isdigit()
        ]
        target = {
            "review_id": f"R{max(review_numbers, default=0) + 1:06d}",
            "severity": "medium",
            "category": "semantic_budget_deferred",
            "start_ms": 0,
            "end_ms": project.get("media", {}).get("duration_ms"),
            "block_ids": [],
            "segment_ids": [],
            "event_ids": [],
            "frame_ids": [],
            "ocr_observation_ids": [],
            "image_claim_ids": [],
            "metadata_revision_ids": [],
            "sufficiency_decision_ids": [],
            "problem": (
                "A bounded semantic budget deferred packet-grounded visual review; "
                "deterministic frames and OCR remain preserved."
            ),
            "alternatives": [],
            "required_action": (
                "Run a later semantic pass with a larger budget or an approved local "
                "observer before consuming deferred visual facts."
            ),
            "blocking": False,
            "decision": None,
            "reviewer": None,
            "decision_timestamp_utc": None,
            "rationale": None,
        }
        review_items.append(target)
        changed = True

    if target.get("event_ids") != deferred:
        changed = True
    target["event_ids"] = deferred
    expected_rationale = (
        f"Current bounded semantic frontier maintained by {provider_id}; "
        f"{len(deferred)} packet(s) remain for a later pass."
    )
    if target.get("rationale") != expected_rationale:
        changed = True
    target["decision"] = None
    target["reviewer"] = None
    target["decision_timestamp_utc"] = None
    target["rationale"] = expected_rationale
    return changed


def _commit_semantic_observation(
    *,
    project_dir: Path,
    annotations_dir: Path,
    provider: VisionProvider,
    project: dict[str, Any],
    ledger: dict[str, Any],
    canonical_patch_state: JsonPatchState,
    packet: VisionPacket,
    image_id: str,
    frame: Mapping[str, Any],
    observation_id: str,
    annotation: VisionAnnotation,
    cache_hit: bool,
    deterministic_no_change: bool = False,
    batch_journal: _SemanticBatchJournal | None = None,
    batch_context: SemanticBatchContext | None = None,
    frame_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    allow_pending_upgrade: bool = True,
) -> dict[str, Any]:
    """Persist one validated annotation while retaining deterministic commit order."""

    base_revision = str(frame.get("latest_revision_id") or "")
    if not base_revision:
        raise ValidationFailure(f"Frame {image_id} lacks a metadata base revision")
    if frame_by_id is None:
        frame_by_id = {
            str(item.get("frame_id")): item
            for item in project.get("frames", [])
            if isinstance(item, Mapping) and item.get("frame_id")
        }
    if allow_pending_upgrade:
        annotation, deterministic_upgrade = _upgrade_semantic_pending_annotation(
            annotation,
            packet,
            frame_by_id,
        )
    else:
        deterministic_upgrade = False
    deterministic_no_change = deterministic_no_change or deterministic_upgrade
    observation = annotation_to_observation(
        annotation,
        packet,
        provider.descriptor,
        project_root=project_dir,
        observation_id=observation_id,
        prompt_template_hash=getattr(provider, "prompt_template_hash", None),
        deterministic=deterministic_no_change,
        prior_metadata_visible=bool(getattr(provider, "prior_metadata_visible", False)),
        prior_claim_context_ids=(
            _prior_claim_context_ids(project, packet)
            if getattr(provider, "prior_metadata_visible", False)
            else ()
        ),
    )
    annotation_path = annotations_dir / f"{packet.candidate_id}.annotation.json"
    observation_path = annotations_dir / f"{packet.candidate_id}.observation.json"
    _write_semantic_sidecars(annotation_path, observation_path, annotation, observation)

    def append_batch_journal(entry: dict[str, Any]) -> None:
        if batch_journal is None:
            return
        journal_entry = dict(entry)
        journal_entry["candidate_id"] = packet.candidate_id
        batch_journal.append(journal_entry)

    def apply_links(
        project_state: dict[str, Any],
        committed_revision_id: str,
        *,
        _image_id: str = image_id,
        _packet: VisionPacket = packet,
        _annotation: VisionAnnotation = annotation,
    ) -> None:
        current_frame = (
            batch_context.frames_by_id.get(_image_id)
            if batch_context is not None
            else frame_by_id.get(_image_id)
        )
        if current_frame is None:
            current_frame = next(
                (
                    item
                    for item in project_state.get("frames", [])
                    if item.get("frame_id") == _image_id
                ),
                None,
            )
        if current_frame is None:
            raise ValidationFailure(f"Vision packet focus frame is absent: {_image_id}")
        _apply_semantic_links(
            project_state,
            provider=provider,
            packet=_packet,
            annotation_event_type=_annotation.event_type,
            annotation_description=_annotation.factual_visible_description,
            annotation_roles=dict(_annotation.before_action_after_roles),
            annotation_confidence=_annotation.confidence,
            annotation_uncertainty=_annotation.uncertainty,
            image_id=_image_id,
            revision_id=committed_revision_id,
            supported_claim_ids=[
                str(item) for item in current_frame.get("supported_claim_ids", [])
            ],
            sufficiency_status=str(current_frame.get("metadata_sufficiency_state", "")),
            event_by_id=(batch_context.events_by_id if batch_context is not None else None),
            blocks_by_frame_id=(
                batch_context.blocks_by_frame_id if batch_context is not None else None
            ),
            reviews_by_frame_id=(
                batch_context.reviews_by_frame_id if batch_context is not None else None
            ),
        )

    result = ingest_project_observation(
        project_dir,
        observation_path,
        base_revision=base_revision,
        finalize=False,
        project_mutator=apply_links,
        incremental_fields=(
            "frames",
            "evidence_image_metadata",
            "visual_observations",
            "image_claims",
            "metadata_revisions",
            "sufficiency_decisions",
            "script_blocks",
            "visual_events",
            "review_items",
        ),
        project_state=project,
        ledger_state=ledger,
        canonical_patch_state=canonical_patch_state,
        canonical_journal=append_batch_journal if batch_journal is not None else None,
        defer_ledger=batch_journal is not None,
        batch_context=batch_context,
    )
    if not isinstance(result, dict):
        raise ValidationFailure("Semantic observation ingestion returned a non-object result")
    committed = dict(result)
    committed["semantic_cache_hit"] = cache_hit
    committed["semantic_deterministic_upgrade"] = deterministic_upgrade
    return committed


def _update_semantic_links(
    project_dir: Path,
    *,
    provider: VisionProvider,
    packet: VisionPacket,
    annotation_event_type: str,
    annotation_description: str,
    annotation_roles: dict[str, str],
    annotation_confidence: float,
    annotation_uncertainty: list[str],
    image_id: str,
    revision_id: str,
    supported_claim_ids: list[str],
    sufficiency_status: str,
    finalize: bool = True,
) -> None:
    project = _load_project(project_dir)
    _apply_semantic_links(
        project,
        provider=provider,
        packet=packet,
        annotation_event_type=annotation_event_type,
        annotation_description=annotation_description,
        annotation_roles=annotation_roles,
        annotation_confidence=annotation_confidence,
        annotation_uncertainty=annotation_uncertainty,
        image_id=image_id,
        revision_id=revision_id,
        supported_claim_ids=supported_claim_ids,
        sufficiency_status=sufficiency_status,
    )
    canonical_path = project_dir / ".state" / "canonical-project.json"
    atomic_write_json(
        canonical_path,
        project,
        compact=canonical_compact_for_payload(canonical_path, project),
    )
    if finalize:
        _finalize_semantic_project(project_dir, project)


def _fallback_annotation(
    packet: VisionPacket,
    error: ValidationFailure,
) -> VisionAnnotation:
    """Create a conservative, packet-valid marker after a bad provider response.

    A malformed local VLM response is an evidence-quality problem, not a reason to
    discard the deterministic transcript/OCR/frame work already completed.  The
    fallback deliberately contains no visual assertion and cites one real packet
    frame solely so the observation can be persisted and routed to review.
    """

    focus_ids = [
        frame.frame_id for frame in packet.frames if frame.role in {"focus", "action", "result"}
    ]
    evidence_id = focus_ids[0] if focus_ids else packet.frames[0].frame_id
    detail = str(error).strip().replace("\n", " ")[:500]
    annotation = VisionAnnotation(
        candidate_id=packet.candidate_id,
        factual_visible_description="Semantic visual evidence is pending independent review.",
        event_type="semantic_pending",
        evidence_frame_ids=[evidence_id],
        before_action_after_roles={},
        exact_visible_text_candidates=[],
        consequential_changes=[],
        confidence=0.0,
        uncertainty=[
            f"The semantic provider response failed deterministic validation: {detail}",
            "No visual fact is asserted by this fallback annotation.",
        ],
        statements_not_inferred=[
            "Any visual state, text, identity, or change not independently reviewed."
        ],
    )
    return validate_annotation_for_packet(annotation, packet)


def _finalize_semantic_project(
    project_dir: Path,
    project: dict[str, Any],
    *,
    canonical_patch_state: JsonPatchState | None = None,
    force_full_write: bool = False,
) -> ValidationResult:
    """Recompute project-wide state once after a semantic observation batch."""

    supported_claims = {
        str(claim.get("claim_id")): claim
        for claim in project.get("image_claims", [])
        if claim.get("status") == "supported"
    }
    for block in project.get("script_blocks", []):
        claim_ids = [
            str(claim_id)
            for claim_id in block.get("image_claim_ids", [])
            if str(claim_id) in supported_claims
        ]
        block["image_claim_ids"] = claim_ids
        if claim_ids:
            block["visual_description"] = " ".join(
                str(supported_claims[claim_id].get("statement", "")).strip()
                for claim_id in claim_ids
            )

    # A prior interrupted semantic pass may have persisted a blocking review
    # item that only described its own transient post-link validation failure.
    # Once the generated candidate artifacts are reconciled, that diagnostic is
    # stale and must not keep a now-valid project permanently blocked.  Keep
    # provider failures and other genuine prerequisite blockers intact.
    project["review_items"] = [
        item
        for item in project.get("review_items", [])
        if not (
            item.get("category") == "blocked_prerequisite"
            and "Post-semantic-link validation failed" in str(item.get("problem", ""))
        )
    ]

    project["audit"] = audit_project(project)
    project["project_status"] = project["audit"]["final_project_status"]
    project["status_reason"] = (
        "Consequential visual or wording uncertainty remains in the review queue."
        if project["project_status"] == "review_required"
        else "All mandatory automatic checks passed; no human verification is implied."
    )
    canonical_path = project_dir / ".state" / "canonical-project.json"
    if force_full_write:
        # A staged semantic batch has never patched canonical root arrays in
        # its hot loop.  Materialize the complete in-memory project exactly
        # once before rendering/validation; the journal remains until the
        # caller confirms this proof succeeded.
        atomic_write_json(
            canonical_path,
            project,
            compact=canonical_compact_for_payload(canonical_path, project),
        )
    else:
        atomic_update_json_fields(
            canonical_path,
            {
                "audit": project["audit"],
                "project_status": project["project_status"],
                "status_reason": project["status_reason"],
                "script_blocks": project.get("script_blocks", []),
                # Budget-frontier reconciliation can remove or replace the
                # project-level semantic_budget_deferred review item after the
                # packet transaction has already committed. Persist that list in
                # canonical state before rendering, or Markdown will correctly
                # omit the stale item while validation still requires its anchor.
                "review_items": project.get("review_items", []),
            },
            fallback_payload=project,
            patch_state=canonical_patch_state,
        )
    atomic_write_json(project_dir / ".state" / "audit.json", project["audit"])
    atomic_write_json(project_dir / ".state" / "review-queue.json", project["review_items"])
    markdown = next(project_dir.glob("*.md"))
    render_to_path(project, markdown)
    validation = validate_project(project_dir, use_cached_file_hash=True)
    if not validation.valid:
        # Explicit rebuilds can leave old rejected-frame files behind.  Repair
        # only the exact generated candidates that the validator proves lack
        # canonical mirrors, then perform one bounded recheck.
        if _prune_unmirrored_generated_candidates(project_dir, validation.errors):
            validation = validate_project(project_dir, use_cached_file_hash=True)
    if not validation.valid:
        raise ValidationFailure(
            "Post-semantic-link validation failed: " + "; ".join(validation.errors)
        )
    return validation


def _semantic_validation_payload(validation: ValidationResult | None) -> dict[str, Any] | None:
    """Serialize the final proof for reuse after manifest-only bookkeeping.

    ``run_semantic_pass`` appends timing/provider telemetry to ``manifest``
    after semantic finalization.  Validation deliberately excludes that
    volatile field from its canonical digest and inventory, so the proof is
    still valid; keeping its small result payload avoids a second full project
    validation on every continuation batch.
    """

    if validation is None:
        return None
    return {
        "valid": bool(validation.valid),
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
        "checks": dict(validation.checks),
    }


def _apply_parallel_vision_provider(
    *,
    project_dir: Path,
    provider: VisionProvider,
    project: dict[str, Any],
    frame_by_id: Mapping[str, Mapping[str, Any]],
    ledger: dict[str, Any],
    canonical_patch_state: JsonPatchState,
    annotations_dir: Path,
    packet_paths: list[Path],
    semantic_deferred: list[str],
    semantic_cache_enabled: bool,
    semantic_cache_root: Path | None,
    semantic_cache_limit: int,
    retry_fallbacks: bool,
    retry_semantic_pending: bool,
    semantic_max_packets: int | None,
    semantic_workers: int,
    failure_limit: int,
    allow_deterministic_stable: bool,
    candidate_ids: set[str] | None = None,
    batch_journal: _SemanticBatchJournal | None = None,
    recovered_batch_candidate_ids: Sequence[str] = (),
    batch_context: SemanticBatchContext | None = None,
) -> dict[str, Any]:
    """Annotate packets concurrently, then commit them in packet order.

    Provider calls are independent, but canonical writes are not.  This two-
    phase shape keeps IDs, review ordering, and crash recovery deterministic
    while allowing an explicitly requested local server slot to overlap image
    encoding/generation.  A provider exception still aborts before any of this
    batch is committed, so the conservative default (one worker) retains the
    historical per-observation resume behavior.
    """

    work: list[tuple[VisionPacket, str, dict[str, Any], str]] = []
    next_observation_number = _next_observation_number(project)
    for index, packet_path in enumerate(packet_paths):
        packet = _load_packet(packet_path)
        focus_ids = [
            frame.frame_id for frame in packet.frames if frame.role in {"focus", "action", "result"}
        ]
        image_id = focus_ids[0] if focus_ids else packet.frames[0].frame_id
        frame = frame_by_id.get(image_id)
        if frame is None:
            raise ValidationFailure(f"Vision packet focus frame is absent: {image_id}")
        work.append(
            (
                packet,
                image_id,
                dict(frame),
                sequential_id("visual_analysis", next_observation_number + index),
            )
        )

    starter = getattr(provider, "start", None)
    if callable(starter):
        starter()
    with ThreadPoolExecutor(
        max_workers=semantic_workers,
        thread_name_prefix="vsr-semantic-provider",
    ) as executor:
        futures = [
            executor.submit(
                _annotate_parallel_packet,
                packet,
                image_id=image_id,
                frame=frame,
                observation_id=observation_id,
                project_dir=project_dir,
                project=project,
                frame_by_id=frame_by_id,
                provider=provider,
                allow_deterministic_stable=allow_deterministic_stable,
                semantic_cache_root=semantic_cache_root if semantic_cache_enabled else None,
                semantic_cache_limit=semantic_cache_limit,
            )
            for packet, image_id, frame, observation_id in work
        ]
        # Reading results in submission order, rather than completion order,
        # makes annotation and metadata IDs stable across worker counts.
        prepared = [future.result() for future in futures]

    applied: list[dict[str, Any]] = []
    semantic_provider_failures = [
        item.provider_failure for item in prepared if item.provider_failure is not None
    ]
    semantic_provider_attempt_failures = len(semantic_provider_failures)
    # Futures are already collected in deterministic packet order. Recompute
    # the same health-only streak used by the sequential path; packet-local
    # 400/JSON/citation failures must not make parallel telemetry claim that a
    # shared provider circuit opened. All submitted calls have completed by
    # this point, so this is a truthful diagnostic rather than a cancellation
    # mechanism.
    health_failure_streak = 0
    semantic_circuit_breaker_triggered = False
    for item in prepared:
        failure = item.provider_failure
        if failure is None:
            health_failure_streak = 0
            continue
        if _semantic_failure_is_provider_health_failure(str(failure.get("error", ""))):
            health_failure_streak += 1
            if health_failure_streak >= failure_limit:
                semantic_circuit_breaker_triggered = True
        else:
            health_failure_streak = 0
    semantic_cache_hits = sum(item.cache_hit for item in prepared)
    semantic_cache_misses = sum(item.cache_miss for item in prepared)
    semantic_cache_writes = sum(item.cache_write for item in prepared)
    semantic_deterministic_no_change_count = sum(
        item.deterministic_no_change for item in prepared
    )
    for item in prepared:
        applied.append(
            _commit_semantic_observation(
                project_dir=project_dir,
                annotations_dir=annotations_dir,
                provider=provider,
                project=project,
                ledger=ledger,
                canonical_patch_state=canonical_patch_state,
                packet=item.packet,
                image_id=item.image_id,
                frame=item.frame,
                observation_id=item.observation_id,
                annotation=item.annotation,
                cache_hit=item.cache_hit,
                deterministic_no_change=item.deterministic_no_change,
                batch_journal=batch_journal,
                batch_context=batch_context,
                frame_by_id=frame_by_id,
                allow_pending_upgrade=item.provider_failure is None,
            )
        )
    semantic_deterministic_no_change_count += sum(
        bool(item.get("semantic_deterministic_upgrade")) for item in applied
    )
    if candidate_ids is not None:
        # A filtered pass (for example a partially completed review bundle)
        # must reconcile the *project-wide* frontier.  The selector's deferred
        # list only covers candidate_ids, so syncing it directly would remove
        # an existing budget marker while unrelated packets remain pending.
        semantic_deferred = _pending_semantic_event_ids(project_dir, project)
        _sync_semantic_budget_review_item(
            project,
            semantic_deferred,
            provider_id=provider.descriptor.provider_id,
        )
    final_validation: ValidationResult | None = None
    if applied or semantic_deferred:
        final_validation = _finalize_semantic_project(
            project_dir,
            project,
            canonical_patch_state=canonical_patch_state,
            force_full_write=batch_journal is not None,
        )
        if batch_journal is not None:
            batch_journal.complete(ledger)
    return {
        "provider": provider.descriptor.provider_id,
        "model": provider.descriptor.model,
        "model_version": provider.descriptor.model_version,
        "applied": applied,
        "skipped_event_ids": [],
        "semantic_cache_enabled": semantic_cache_enabled,
        "semantic_cache_hit_count": semantic_cache_hits,
        "semantic_cache_miss_count": semantic_cache_misses,
        "semantic_cache_write_count": semantic_cache_writes,
        "semantic_content_cache_hit_count": 0,
        "semantic_content_cache_write_count": 0,
        "semantic_visual_reuse_hit_count": 0,
        "semantic_visual_content_reuse_hit_count": 0,
        "semantic_deterministic_no_change_count": semantic_deterministic_no_change_count,
        "semantic_cache_path": str(semantic_cache_root) if semantic_cache_root else None,
        "semantic_provider_failures": semantic_provider_failures,
        "semantic_provider_attempt_failure_count": semantic_provider_attempt_failures,
        "semantic_fallback_annotation_count": len(semantic_provider_failures),
        # All in-flight calls have already been issued when a failure is seen;
        # expose the health-only threshold without claiming requests stopped.
        "semantic_circuit_breaker_triggered": semantic_circuit_breaker_triggered,
        "semantic_deferred_event_ids": semantic_deferred,
        "semantic_max_packets": semantic_max_packets
        if semantic_max_packets is not None
        else _semantic_packet_budget(),
        "semantic_retry_fallbacks": retry_fallbacks,
        "semantic_retry_semantic_pending": retry_semantic_pending,
        "semantic_worker_count": semantic_workers,
        "semantic_validation": _semantic_validation_payload(final_validation),
        "semantic_batch_journal_recovered_candidate_ids": list(
            dict.fromkeys(str(value) for value in recovered_batch_candidate_ids)
        ),
        "semantic_batch_journal_recovered_count": len(
            set(str(value) for value in recovered_batch_candidate_ids)
        ),
    }


def apply_vision_provider(
    project_dir: Path,
    provider: VisionProvider,
    *,
    semantic_max_packets: int | None = None,
    retry_fallbacks: bool = False,
    retry_semantic_pending: bool = False,
    semantic_workers: int | None = None,
    candidate_ids: set[str] | None = None,
    deterministic_only: bool = False,
    allow_observed_candidate_ids: bool = False,
) -> dict[str, Any]:
    """Annotate persisted events and ingest them transactionally.

    ``deterministic_only`` is used by the offline host-agent route to commit
    conservative pixel-identical no-change observations before creating a
    Codex review bundle. It never invokes the provider for a non-identical
    packet and therefore cannot consume a semantic model/API budget.
    """

    project_dir = project_dir.resolve(strict=True)
    # Complete any prior journal before reading packet state.  This is the
    # restart gate for a provider/validation failure that occurred after one
    # or more observations had been durably staged.
    recovered_batch_candidate_ids = _SemanticBatchJournal.recover(project_dir)
    annotations_dir = project_dir / ".state" / "vision" / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    applied: list[dict[str, Any]] = []
    skipped: list[str] = []
    project = _load_project(project_dir)
    ledger = _load_ledger(project_dir)
    batch_context: SemanticBatchContext | None = None
    frame_by_id = {
        str(frame.get("frame_id")): frame
        for frame in project.get("frames", [])
        if isinstance(frame, Mapping) and frame.get("frame_id")
    }
    event_by_id = {
        str(event.get("event_id")): event
        for event in project.get("visual_events", [])
        if isinstance(event, Mapping) and event.get("event_id")
    }
    next_observation_number = _next_observation_number(project)
    canonical_patch_state = JsonPatchState()
    semantic_cache_enabled = _semantic_cache_enabled(provider)
    semantic_cache_root = _semantic_cache_root() if semantic_cache_enabled else None
    semantic_cache_limit = _semantic_cache_limit() if semantic_cache_enabled else 0
    semantic_cache_hits = 0
    semantic_cache_misses = 0
    semantic_cache_writes = 0
    semantic_content_cache_hits = 0
    semantic_content_cache_writes = 0
    semantic_visual_reuse_hits = 0
    semantic_visual_reuse: dict[str, VisionAnnotation] = {}
    semantic_visual_content_reuse_hits = 0
    semantic_visual_content_reuse: dict[
        str, tuple[VisionAnnotation, tuple[str, ...]]
    ] = {}
    semantic_provider_failures: list[dict[str, Any]] = []
    semantic_provider_attempt_failures = 0
    semantic_provider_failure_streak = 0
    semantic_circuit_breaker_triggered = False
    semantic_deterministic_no_change_count = 0
    try:
        failure_limit = max(
            1,
            min(20, int(os.environ.get("VSR_SEMANTIC_FAILURE_LIMIT", "3"))),
        )
    except ValueError:
        failure_limit = 3
    packet_paths, semantic_deferred = _select_semantic_packet_files(
        project_dir,
        semantic_max_packets=semantic_max_packets,
        project=project,
        retry_fallbacks=retry_fallbacks,
        retry_semantic_pending=retry_semantic_pending,
        prompt_template_hash=getattr(provider, "prompt_template_hash", None),
        candidate_ids=candidate_ids,
        deterministic_only=deterministic_only,
        allow_observed_candidate_ids=allow_observed_candidate_ids,
        allow_deterministic_stable=(
            candidate_ids is None
            and not deterministic_only
            and provider.descriptor.route == "host_agent"
        ),
    )
    if packet_paths:
        # Parse append-only observation/revision history once for this semantic
        # batch. Provider calls may run in parallel, but commits remain ordered
        # and therefore update this process-local context deterministically.
        batch_context = SemanticBatchContext.from_project(project)
    batch_journal = (
        _SemanticBatchJournal.start(project_dir) if packet_paths else None
    )
    # When candidate_ids filters the pass, ``semantic_deferred`` intentionally
    # excludes every other pending packet.  Do not clear a project-wide budget
    # frontier before those selected observations commit; it is reconciled from
    # the full event set below after the filtered batch finishes.
    filtered_pass = candidate_ids is not None or deterministic_only
    budget_review_changed = (
        False
        if filtered_pass
        else _sync_semantic_budget_review_item(
            project,
            semantic_deferred,
            provider_id=provider.descriptor.provider_id,
        )
    )
    resolved_workers = 1 if deterministic_only else _semantic_worker_count(semantic_workers)
    if resolved_workers > 1 and packet_paths:
        return _apply_parallel_vision_provider(
            project_dir=project_dir,
            provider=provider,
            project=project,
            frame_by_id=frame_by_id,
            ledger=ledger,
            canonical_patch_state=canonical_patch_state,
            annotations_dir=annotations_dir,
            packet_paths=packet_paths,
            semantic_deferred=semantic_deferred,
            semantic_cache_enabled=semantic_cache_enabled,
            semantic_cache_root=semantic_cache_root,
            semantic_cache_limit=semantic_cache_limit,
            retry_fallbacks=retry_fallbacks,
            retry_semantic_pending=retry_semantic_pending,
            semantic_max_packets=semantic_max_packets,
            semantic_workers=resolved_workers,
            failure_limit=failure_limit,
            allow_deterministic_stable=(
                candidate_ids is None
                and not deterministic_only
                and provider.descriptor.route == "host_agent"
            ),
            candidate_ids=candidate_ids,
            batch_journal=batch_journal,
            recovered_batch_candidate_ids=recovered_batch_candidate_ids,
            batch_context=batch_context,
        )
    for packet_path in packet_paths:
        packet = _load_packet(packet_path)
        focus_ids = [
            frame.frame_id for frame in packet.frames if frame.role in {"focus", "action", "result"}
        ]
        image_id = focus_ids[0] if focus_ids else packet.frames[0].frame_id
        frame = frame_by_id.get(image_id)
        if frame is None:
            raise ValidationFailure(f"Vision packet focus frame is absent: {image_id}")
        retryable_fallback = retry_fallbacks and _is_retryable_http400_fallback(
            project_dir, packet.candidate_id
        )
        retryable_pending = retry_semantic_pending and _is_retryable_semantic_pending(
            project_dir,
            packet.candidate_id,
            prompt_template_hash=getattr(provider, "prompt_template_hash", None),
        )
        selected_for_review = (
            candidate_ids is not None
            and packet.candidate_id in candidate_ids
            and (
                allow_observed_candidate_ids
                or event_by_id.get(packet.candidate_id, {}).get("event_type")
                == "semantic_pending"
            )
        )
        if _semantic_event_is_observed(event_by_id.get(packet.candidate_id)) and not (
            retryable_fallback or retryable_pending or selected_for_review
        ):
            skipped.append(packet.candidate_id)
            continue
        observation_id = sequential_id("visual_analysis", next_observation_number)
        next_observation_number += 1
        cache_key_value: str | None = None
        cache_path: Path | None = None
        content_cache_key_value: str | None = None
        content_cache_path: Path | None = None
        cache_hit = False
        frame_hashes: dict[str, str] = {}
        annotation: VisionAnnotation | None = None
        annotation_from_provider = False
        annotation_from_fallback = False
        annotation_from_deterministic_no_change = False
        reuse_key = _semantic_visual_reuse_key(packet)
        content_reuse = _semantic_visual_content_key(packet, frame_by_id)
        if semantic_cache_root is not None:
            try:
                frame_hashes = _packet_frame_hashes(project, packet, project_dir)
                cache_key_value = _semantic_cache_key(provider, packet, frame_hashes)
                cache_path = semantic_cache_root / f"{cache_key_value}.json"
                annotation = _read_semantic_cache(
                    cache_path,
                    key=cache_key_value,
                    packet=packet,
                    max_bytes=semantic_cache_limit,
                )
                if annotation is not None:
                    cache_hit = True
                    annotation_from_provider = True
                    semantic_cache_hits += 1
                else:
                    semantic_cache_misses += 1
            except (OSError, ValidationFailure, TypeError, ValueError) as exc:
                # Cache inspection is an acceleration layer only. A malformed
                # or unavailable cache must never prevent a fresh observation.
                LOGGER.info("Semantic cache lookup skipped for %s: %s", packet.candidate_id, exc)
                semantic_cache_misses += 1
        if annotation is None and semantic_cache_root is not None and content_reuse is not None:
            try:
                content_key, source_frame_ids = content_reuse
                content_cache_key_value = _semantic_content_cache_key(
                    provider,
                    content_key,
                    packet=packet,
                )
                content_cache_path = semantic_cache_root / f"content-{content_cache_key_value}.json"
                annotation = _read_semantic_content_cache(
                    content_cache_path,
                    key=content_cache_key_value,
                    packet=packet,
                    source_frame_ids=source_frame_ids,
                    max_bytes=semantic_cache_limit,
                )
                if annotation is not None:
                    cache_hit = True
                    # Preserve the public aggregate hit count for all
                    # deterministic cache paths; the content-specific count
                    # below remains available for detailed telemetry.
                    semantic_cache_hits += 1
                    if semantic_cache_misses:
                        semantic_cache_misses -= 1
                    semantic_content_cache_hits += 1
            except (OSError, ValidationFailure, TypeError, ValueError) as exc:
                LOGGER.info(
                    "Semantic content cache lookup skipped for %s: %s",
                    packet.candidate_id,
                    exc,
                )
        if annotation is None:
            reused = semantic_visual_reuse.get(reuse_key)
            if reused is not None:
                annotation = validate_annotation_for_packet(
                    reused.model_copy(update={"candidate_id": packet.candidate_id}),
                    packet,
                )
                semantic_visual_reuse_hits += 1
        if annotation is None and content_reuse is not None:
            content_key, source_frame_ids = content_reuse
            reused_content = semantic_visual_content_reuse.get(content_key)
            if reused_content is not None:
                annotation = _remap_reused_annotation(
                    reused_content[0],
                    source_frame_ids=reused_content[1],
                    packet=packet,
                )
                semantic_visual_content_reuse_hits += 1
        if annotation is None:
            annotation = _deterministic_identical_frame_annotation(packet, frame_by_id)
            if (
                annotation is None
                and candidate_ids is None
                and not deterministic_only
                and provider.descriptor.route == "host_agent"
            ):
                annotation = _deterministic_stable_frame_annotation(packet, frame_by_id)
            if annotation is not None:
                annotation_from_deterministic_no_change = True
                semantic_deterministic_no_change_count += 1
        if annotation is None:
            if semantic_circuit_breaker_triggered:
                fallback_error = ValidationFailure(
                    "Semantic provider circuit breaker open after repeated invalid responses"
                )
                semantic_provider_failures.append(
                    {
                        "candidate_id": packet.candidate_id,
                        "error": str(fallback_error),
                        "provider_attempted": False,
                        "fallback": True,
                    }
                )
                annotation = _fallback_annotation(packet, fallback_error)
                annotation_from_fallback = True
            else:
                try:
                    annotation = _annotate_with_transport_identity(
                        provider,
                        packet,
                        project_dir=project_dir,
                        frame_hashes=frame_hashes,
                    )
                    annotation_from_provider = True
                    semantic_provider_failure_streak = 0
                except ValidationFailure as exc:
                    semantic_provider_attempt_failures += 1
                    # Do not let packet-local model/context failures suppress
                    # every later packet.  The breaker is reserved for shared
                    # provider-health failures such as a refused connection,
                    # timeout, or server-side 5xx response.
                    if _semantic_failure_is_provider_health_failure(exc):
                        semantic_provider_failure_streak += 1
                        if semantic_provider_failure_streak >= failure_limit:
                            semantic_circuit_breaker_triggered = True
                    else:
                        semantic_provider_failure_streak = 0
                    semantic_provider_failures.append(
                        {
                            "candidate_id": packet.candidate_id,
                            "error": str(exc)[:1000],
                            "provider_attempted": True,
                            "fallback": True,
                        }
                    )
                    annotation = _fallback_annotation(packet, exc)
                    annotation_from_fallback = True
            if annotation_from_provider and cache_path is not None and cache_key_value is not None:
                if not frame_hashes:
                    frame_hashes = _packet_frame_hashes(project, packet, project_dir)
                if _write_semantic_cache(
                    cache_path,
                    key=cache_key_value,
                    provider=provider,
                    packet=packet,
                    annotation=annotation,
                    frame_hashes=frame_hashes,
                    max_bytes=semantic_cache_limit,
                ):
                    semantic_cache_writes += 1
            if annotation_from_provider and content_reuse is not None:
                try:
                    content_key, source_frame_ids = content_reuse
                    if content_cache_key_value is None:
                        content_cache_key_value = _semantic_content_cache_key(
                            provider,
                            content_key,
                            packet=packet,
                        )
                    if content_cache_path is None and semantic_cache_root is not None:
                        content_cache_path = (
                            semantic_cache_root / f"content-{content_cache_key_value}.json"
                        )
                    if content_cache_path is not None and _write_semantic_content_cache(
                        content_cache_path,
                        key=content_cache_key_value,
                        provider=provider,
                        packet=packet,
                        annotation=annotation,
                        source_frame_ids=source_frame_ids,
                        max_bytes=semantic_cache_limit,
                    ):
                        semantic_content_cache_writes += 1
                except (OSError, TypeError, ValueError) as exc:
                    LOGGER.info(
                        "Semantic content cache write skipped for %s: %s",
                        packet.candidate_id,
                        exc,
                    )
        if annotation_from_provider and semantic_provider_failure_streak == 0:
            semantic_visual_reuse.setdefault(reuse_key, annotation)
            if content_reuse is not None:
                content_key, source_frame_ids = content_reuse
                semantic_visual_content_reuse.setdefault(
                    content_key, (annotation, source_frame_ids)
                )
        if not annotation_from_fallback:
            annotation, deterministic_upgrade = _upgrade_semantic_pending_annotation(
                annotation,
                packet,
                frame_by_id,
            )
        else:
            deterministic_upgrade = False
        annotation_from_deterministic_no_change = (
            annotation_from_deterministic_no_change or deterministic_upgrade
        )
        observation = annotation_to_observation(
            annotation,
            packet,
            provider.descriptor,
            project_root=project_dir,
            observation_id=observation_id,
            prompt_template_hash=getattr(provider, "prompt_template_hash", None),
            deterministic=annotation_from_deterministic_no_change,
            prior_metadata_visible=bool(getattr(provider, "prior_metadata_visible", False)),
            prior_claim_context_ids=(
                _prior_claim_context_ids(project, packet)
                if getattr(provider, "prior_metadata_visible", False)
                else ()
            ),
        )
        base_revision = str(frame.get("latest_revision_id") or "")
        if not base_revision:
            raise ValidationFailure(f"Frame {image_id} lacks a metadata base revision")
        annotation_path = annotations_dir / f"{packet.candidate_id}.annotation.json"
        observation_path = annotations_dir / f"{packet.candidate_id}.observation.json"
        _write_semantic_sidecars(annotation_path, observation_path, annotation, observation)

        def append_batch_journal(
            entry: dict[str, Any], *, _candidate_id: str = packet.candidate_id
        ) -> None:
            if batch_journal is None:
                return
            journal_entry = dict(entry)
            journal_entry["candidate_id"] = _candidate_id
            batch_journal.append(journal_entry)

        def apply_links(
            project_state: dict[str, Any],
            committed_revision_id: str,
            *,
            _image_id: str = image_id,
            _packet: VisionPacket = packet,
            _annotation: Any = annotation,
        ) -> None:
            current_frame = (
                batch_context.frames_by_id.get(_image_id)
                if batch_context is not None
                else frame_by_id.get(_image_id)
            )
            if current_frame is None:
                current_frame = next(
                    (
                        item
                        for item in project_state.get("frames", [])
                        if item.get("frame_id") == _image_id
                    ),
                    None,
                )
            if current_frame is None:
                raise ValidationFailure(f"Vision packet focus frame is absent: {_image_id}")
            _apply_semantic_links(
                project_state,
                provider=provider,
                packet=_packet,
                annotation_event_type=_annotation.event_type,
                annotation_description=_annotation.factual_visible_description,
                annotation_roles=dict(_annotation.before_action_after_roles),
                annotation_confidence=_annotation.confidence,
                annotation_uncertainty=_annotation.uncertainty,
                image_id=_image_id,
                revision_id=committed_revision_id,
                supported_claim_ids=[
                    str(item) for item in current_frame.get("supported_claim_ids", [])
                ],
                sufficiency_status=str(
                    current_frame.get("metadata_sufficiency_state", "")
                ),
                event_by_id=(batch_context.events_by_id if batch_context is not None else None),
                blocks_by_frame_id=(
                    batch_context.blocks_by_frame_id if batch_context is not None else None
                ),
                reviews_by_frame_id=(
                    batch_context.reviews_by_frame_id if batch_context is not None else None
                ),
            )

        result = ingest_project_observation(
            project_dir,
            observation_path,
            base_revision=base_revision,
            finalize=False,
            project_mutator=apply_links,
            incremental_fields=(
                "frames",
                "evidence_image_metadata",
                "visual_observations",
                "image_claims",
                "metadata_revisions",
                "sufficiency_decisions",
                "script_blocks",
                "visual_events",
                "review_items",
            ),
            project_state=project,
            ledger_state=ledger,
            canonical_patch_state=canonical_patch_state,
            canonical_journal=append_batch_journal if batch_journal is not None else None,
            defer_ledger=batch_journal is not None,
            batch_context=batch_context,
        )
        if isinstance(result, dict):
            result = dict(result)
            result["semantic_cache_hit"] = cache_hit
            result["semantic_deterministic_no_change"] = annotation_from_deterministic_no_change
            result["semantic_deterministic_upgrade"] = deterministic_upgrade
        applied.append(result)
    if filtered_pass:
        # Recompute after commits because project_state now contains the newly
        # linked event observations.  This preserves unresolved IDs from other
        # packets and removes the marker only when the entire frontier is empty.
        semantic_deferred = _pending_semantic_event_ids(project_dir, project)
        budget_review_changed = _sync_semantic_budget_review_item(
            project,
            semantic_deferred,
            provider_id=provider.descriptor.provider_id,
        )
    final_validation: ValidationResult | None = None
    if applied or semantic_deferred or budget_review_changed:
        final_validation = _finalize_semantic_project(
            project_dir,
            project,
            canonical_patch_state=canonical_patch_state,
            force_full_write=batch_journal is not None,
        )
        if batch_journal is not None:
            batch_journal.complete(ledger)
    return {
        "provider": provider.descriptor.provider_id,
        "model": provider.descriptor.model,
        "model_version": provider.descriptor.model_version,
        "applied": applied,
        "skipped_event_ids": skipped,
        "semantic_cache_enabled": semantic_cache_enabled,
        "semantic_cache_hit_count": semantic_cache_hits,
        "semantic_cache_miss_count": semantic_cache_misses,
        "semantic_cache_write_count": semantic_cache_writes,
        "semantic_content_cache_hit_count": semantic_content_cache_hits,
        "semantic_content_cache_write_count": semantic_content_cache_writes,
        "semantic_visual_reuse_hit_count": semantic_visual_reuse_hits,
        "semantic_visual_content_reuse_hit_count": semantic_visual_content_reuse_hits,
        "semantic_deterministic_no_change_count": semantic_deterministic_no_change_count,
        "semantic_cache_path": str(semantic_cache_root) if semantic_cache_root else None,
        "semantic_provider_failures": semantic_provider_failures,
        "semantic_provider_attempt_failure_count": semantic_provider_attempt_failures,
        "semantic_fallback_annotation_count": len(semantic_provider_failures),
        "semantic_circuit_breaker_triggered": semantic_circuit_breaker_triggered,
        "semantic_deferred_event_ids": semantic_deferred,
        "semantic_max_packets": semantic_max_packets
        if semantic_max_packets is not None
        else _semantic_packet_budget(),
        "semantic_retry_fallbacks": retry_fallbacks,
        "semantic_retry_semantic_pending": retry_semantic_pending,
        "semantic_worker_count": resolved_workers,
        "semantic_validation": _semantic_validation_payload(final_validation),
        "semantic_batch_journal_recovered_candidate_ids": recovered_batch_candidate_ids,
        "semantic_batch_journal_recovered_count": len(
            set(str(value) for value in recovered_batch_candidate_ids)
        ),
    }


def refresh_semantic_state(project_dir: Path) -> None:
    """Reconcile supported claim links on a resume without starting a provider."""

    project_dir = project_dir.resolve(strict=True)
    project = _load_project(project_dir)
    # A project can reach zero pending packets after a bounded continuation
    # while retaining an older unresolved scheduling marker.  Clear only that
    # automatic frontier; provider observations and human review decisions are
    # preserved by the normal finalization path.
    _sync_semantic_budget_review_item(
        project,
        [],
        provider_id="semantic-scheduler",
    )
    _finalize_semantic_project(project_dir, project)


def run_semantic_pass(
    project_dir: Path,
    provider: VisionProvider,
    *,
    semantic_max_packets: int | None = None,
    retry_fallbacks: bool = False,
    retry_semantic_pending: bool = False,
    semantic_workers: int | None = None,
    candidate_ids: set[str] | None = None,
    allow_observed_candidate_ids: bool = False,
) -> dict[str, Any]:
    """Process pending semantic packets without rerunning transcript or video stages.

    This is the efficient continuation path for long projects: ASR, frame
    extraction, OCR, and metadata remain immutable while only unobserved
    packet events consume local VLM time.  A fresh validation receipt is written
    when the existing run key is available, so a normal resume can still use its
    stat-bound fast path.
    """

    project_dir = project_dir.resolve(strict=True)
    started_at = time.perf_counter()
    summary = apply_vision_provider(
        project_dir,
        provider,
        semantic_max_packets=semantic_max_packets,
        retry_fallbacks=retry_fallbacks,
        retry_semantic_pending=retry_semantic_pending,
        semantic_workers=semantic_workers,
        candidate_ids=candidate_ids,
        allow_observed_candidate_ids=allow_observed_candidate_ids,
    )
    elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    applied_count = len(summary.get("applied", []))
    summary["semantic_elapsed_seconds"] = round(elapsed_seconds, 6)
    summary["semantic_observations_per_second"] = round(
        applied_count / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        6,
    )
    project = _load_project(project_dir)
    if summary.get("applied") or summary.get("semantic_deferred_event_ids"):
        manifest = project.setdefault("manifest", {})
        usage = manifest.setdefault("provider_usage", [])
        fallback_count = int(summary.get("semantic_fallback_annotation_count", 0))
        deferred_count = len(summary.get("semantic_deferred_event_ids", []))
        usage_record = {
            "route": provider.descriptor.route,
            "provider": provider.descriptor.provider_id,
            "model": provider.descriptor.model,
            "model_version": provider.descriptor.model_version,
            "purpose": "visual",
            "applied_observations": len(summary.get("applied", [])),
            "skipped_events": len(summary.get("skipped_event_ids", [])),
            "semantic_cache_enabled": bool(summary.get("semantic_cache_enabled", False)),
            "semantic_cache_hit_count": int(summary.get("semantic_cache_hit_count", 0)),
            "semantic_cache_miss_count": int(summary.get("semantic_cache_miss_count", 0)),
            "semantic_cache_write_count": int(summary.get("semantic_cache_write_count", 0)),
            "semantic_content_cache_hit_count": int(
                summary.get("semantic_content_cache_hit_count", 0)
            ),
            "semantic_content_cache_write_count": int(
                summary.get("semantic_content_cache_write_count", 0)
            ),
            "semantic_visual_reuse_hit_count": int(
                summary.get("semantic_visual_reuse_hit_count", 0)
            ),
            "semantic_visual_content_reuse_hit_count": int(
                summary.get("semantic_visual_content_reuse_hit_count", 0)
            ),
            "semantic_deterministic_no_change_count": int(
                summary.get("semantic_deterministic_no_change_count", 0)
            ),
            "semantic_provider_attempt_failure_count": int(
                summary.get("semantic_provider_attempt_failure_count", 0)
            ),
            "semantic_fallback_annotation_count": fallback_count,
            "semantic_circuit_breaker_triggered": bool(
                summary.get("semantic_circuit_breaker_triggered", False)
            ),
            "semantic_provider_failures": list(summary.get("semantic_provider_failures", [])),
            "semantic_deferred_event_count": deferred_count,
            "semantic_deferred_event_ids": list(summary.get("semantic_deferred_event_ids", [])),
            "semantic_max_packets": summary.get("semantic_max_packets"),
            "semantic_retry_fallbacks": bool(summary.get("semantic_retry_fallbacks", False)),
            "semantic_retry_semantic_pending": bool(
                summary.get("semantic_retry_semantic_pending", False)
            ),
            "semantic_worker_count": int(summary.get("semantic_worker_count", 1)),
            "semantic_elapsed_seconds": float(summary["semantic_elapsed_seconds"]),
            "semantic_observations_per_second": float(
                summary["semantic_observations_per_second"]
            ),
        }
        if isinstance(usage, list):
            usage.append(usage_record)
        degradations = manifest.setdefault("degradations", [])
        if fallback_count:
            message = (
                "Semantic provider returned invalid responses for "
                f"{fallback_count} packet(s); conservative review-only fallbacks were persisted."
            )
            if message not in degradations:
                degradations.append(message)
        if deferred_count:
            message = (
                f"Semantic packet budget deferred {deferred_count} event(s); "
                "deterministic evidence and targeted review items were retained."
            )
            if message not in degradations:
                degradations.append(message)
        canonical_path = project_dir / ".state" / "canonical-project.json"
        # ``_finalize_semantic_project`` has already committed the large
        # evidence/audit projections.  Only the continuation telemetry in the
        # manifest changed after timing the provider call; patch that single
        # root field instead of re-serializing the entire canonical project.
        # This keeps resume passes crash-safe while avoiding a second full
        # JSON parse/encode of multi-hundred-megabyte projects.
        atomic_update_json_fields(
            canonical_path,
            {"manifest": manifest},
            fallback_payload=project,
        )
        atomic_write_json(project_dir / ".state" / "run-manifest.json", manifest)
        # ``apply_vision_provider`` already finalized the batch, including the
        # audit, review queue, Markdown render, and filesystem validation.  The
        # continuation-only bookkeeping above changes manifest telemetry only,
        # so rewriting those projections a second time would add disk churn
        # without changing their content.
    validation_payload = summary.get("semantic_validation")
    if isinstance(validation_payload, Mapping) and validation_payload.get("valid") is True:
        validation = ValidationResult(
            valid=True,
            errors=[str(item) for item in validation_payload.get("errors", [])],
            warnings=[str(item) for item in validation_payload.get("warnings", [])],
            checks=(
                dict(validation_payload.get("checks", {}))
                if isinstance(validation_payload.get("checks", {}), Mapping)
                else {}
            ),
        )
    else:
        # No semantic batch finalization occurred (for example, a no-op pass),
        # so there is no proof to reuse and the normal independent validator
        # remains the source of truth.
        validation = validate_project(project_dir, use_cached_file_hash=True)
    if not validation.valid:
        raise ValidationFailure("Semantic-only pass validation failed: " + "; ".join(validation.errors))
    run_key = project.get("manifest", {}).get("run_cache_key")
    if isinstance(run_key, str) and run_key:
        write_validation_receipt(
            project_dir,
            project,
            run_cache_key=run_key,
            validation=validation,
        )
    return {
        "project_dir": str(project_dir),
        "status": project.get("project_status"),
        "summary": summary,
        "validation": validation.checks,
        "validation_errors": validation.errors,
    }


def run_semantic_batch(
    output_root: Path,
    provider: VisionProvider,
    *,
    semantic_max_packets: int = 32,
    retry_fallbacks: bool = False,
    retry_semantic_pending: bool = False,
    min_free_bytes: int = 10 * 1024**3,
    continue_on_error: bool = False,
    semantic_workers: int | None = None,
) -> dict[str, Any]:
    """Process a collection of canonical projects with one reusable provider.

    This is deliberately a continuation-only scheduler: it never discovers or
    decodes source media.  Projects are visited in retention order, packets are
    bounded per project, and a free-space reserve is checked before each
    mutating pass.  Keeping one provider alive avoids repeatedly loading the
    same local multimodal model when a collection contains several videos.
    """

    if semantic_max_packets <= 0:
        raise ValueError("semantic_max_packets must be positive")
    if min_free_bytes < 0:
        raise ValueError("min_free_bytes must be zero or greater")
    root = output_root.expanduser().resolve(strict=True)
    projects = discover_projects(root)
    records: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for item in projects:
        project_dir = Path(item.path)
        pending_before = pending_packet_count(project_dir)
        retry_fallback_before = 0
        prompt_refresh_before = 0
        if retry_fallbacks or retry_semantic_pending:
            retry_fallback_before, prompt_refresh_before = _retryable_semantic_counts(
                project_dir,
                prompt_template_hash=getattr(provider, "prompt_template_hash", None),
                include_http400=retry_fallbacks,
                include_semantic_pending=retry_semantic_pending,
            )
        if pending_before == 0 and prompt_refresh_before == 0 and retry_fallback_before == 0:
            # Keep no-model skips genuinely cheap, but repair stale automatic
            # budget markers left by older bounded passes when present.
            project_snapshot = _load_project(project_dir)
            if any(
                isinstance(review, dict)
                and review.get("category") == "semantic_budget_deferred"
                and not review.get("decision")
                for review in project_snapshot.get("review_items", [])
            ):
                refresh_semantic_state(project_dir)
            records.append(
                {
                    "project_dir": str(project_dir),
                    "status": "skipped",
                    "pending_before": 0,
                    "pending_after": 0,
                    "retry_fallback_before": 0,
                    "prompt_refresh_before": 0,
                    "reason": "no_pending_semantic_packets",
                }
            )
            continue
        free_bytes = shutil.disk_usage(root).free
        if free_bytes < min_free_bytes:
            blocked_item = {
                "project_dir": str(project_dir),
                "status": "blocked",
                "pending_before": pending_before,
                "pending_after": pending_before,
                "retry_fallback_before": retry_fallback_before,
                "prompt_refresh_before": prompt_refresh_before,
                "reason": "free_space_reserve",
                "free_bytes": free_bytes,
                "min_free_bytes": min_free_bytes,
            }
            blocked.append(blocked_item)
            records.append(blocked_item)
            break
        try:
            result = run_semantic_pass(
                project_dir,
                provider,
                semantic_max_packets=semantic_max_packets,
                retry_fallbacks=retry_fallbacks,
                retry_semantic_pending=retry_semantic_pending,
                semantic_workers=semantic_workers,
            )
        except (OSError, ValidationFailure, ValueError) as exc:
            blocked_item = {
                "project_dir": str(project_dir),
                "status": "blocked",
                "pending_before": pending_before,
                "pending_after": pending_before,
                "prompt_refresh_before": prompt_refresh_before,
                "reason": "semantic_pass_error",
                "error": str(exc),
            }
            blocked.append(blocked_item)
            records.append(blocked_item)
            if not continue_on_error:
                break
            continue
        pending_after = pending_packet_count(project_dir)
        records.append(
            {
                "project_dir": str(project_dir),
                "status": result.get("status"),
                "pending_before": pending_before,
                "pending_after": pending_after,
                "retry_fallback_before": retry_fallback_before,
                "prompt_refresh_before": prompt_refresh_before,
                "summary": result.get("summary", {}),
                "validation_errors": result.get("validation_errors", []),
            }
        )
    statuses = {str(item.get("status")) for item in records}
    overall_status = "blocked" if blocked else (
        "review_required" if "review_required" in statuses else "automatically_checked"
    )
    return {
        "output_root": str(root),
        "provider": provider.descriptor.provider_id,
        "model": provider.descriptor.model,
        "semantic_max_packets": semantic_max_packets,
        "retry_fallbacks": retry_fallbacks,
        "retry_semantic_pending": retry_semantic_pending,
        "min_free_bytes": min_free_bytes,
        "projects": records,
        "processed_count": sum(1 for item in records if item.get("status") != "skipped"),
        "skipped_count": sum(1 for item in records if item.get("status") == "skipped"),
        "blocked": blocked,
        "status": overall_status,
    }


__all__ = [
    "apply_vision_provider",
    "pending_packet_count",
    "prepare_host_agent_handoffs",
    "refresh_semantic_state",
    "run_semantic_batch",
    "run_semantic_pass",
]
