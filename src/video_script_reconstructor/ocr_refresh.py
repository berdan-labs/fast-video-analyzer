"""Repair OCR in an existing canonical project without rerunning media stages."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import audit_project
from .errors import BlockedError, ValidationFailure
from .ocr import OCRAdapter, OCRObservation, TesseractOCRAdapter, to_schema_observation
from .render_markdown import render_to_path
from .security import atomic_write_json, canonical_compact_for_payload, safe_relative_path
from .validate_output import validate_project, write_validation_receipt
from .vision_packets import VisionPacket

_REFRESH_SCHEMA = "ocr-refresh-v2-quote-safe-tsv"


def _refresh_workers(value: int | None) -> int:
    raw = str(value) if value is not None else os.environ.get("VSR_OCR_REFRESH_WORKERS", "")
    if not raw.strip():
        return min(8, max(1, os.cpu_count() or 1))
    try:
        workers = int(raw)
    except ValueError as exc:
        raise ValueError("OCR refresh workers must be an integer") from exc
    if not 1 <= workers <= 16:
        raise ValueError("OCR refresh workers must be between 1 and 16")
    return workers


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _load_project(project_dir: Path) -> dict[str, Any]:
    path = project_dir / ".state" / "canonical-project.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"Unable to read canonical project: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationFailure("Canonical project root must be an object")
    return value


def _ocr_signature(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        value.get("raw_engine_text"),
        value.get("normalized_interpretation"),
        value.get("confidence"),
        value.get("bounding_region"),
        value.get("uncertain_characters"),
        value.get("engine"),
        value.get("engine_version"),
    )


def _frame_path(project_dir: Path, frame: Mapping[str, Any]) -> Path:
    raw_path = frame.get("full_frame_path") or frame.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValidationFailure(f"Frame {frame.get('frame_id')} has no image path")
    return safe_relative_path(project_dir, raw_path)


def _packet_files(project_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted((project_dir / ".state" / "vision" / "packets").glob("V*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("schema_name") == (
            "video-script-reconstructor.vision-packet"
        ):
            files.append(path)
    return files


def _replace_packet_ocr(
    project_dir: Path, observations: Mapping[str, Mapping[str, Any]]
) -> int:
    """Rewrite packet OCR projections while leaving packet identity/timing intact."""

    changed_packets = 0
    for path in _packet_files(project_dir):
        try:
            packet = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(packet, dict) or not isinstance(packet.get("raw_ocr"), list):
            continue
        changed = False
        projected: list[Any] = []
        for item in packet["raw_ocr"]:
            if not isinstance(item, dict):
                projected.append(item)
                continue
            observation_id = str(item.get("observation_id", ""))
            replacement = observations.get(observation_id)
            if replacement is None:
                projected.append(item)
                continue
            projected.append(dict(replacement))
            changed = changed or _ocr_signature(item) != _ocr_signature(replacement)
        if changed:
            packet["raw_ocr"] = projected
            atomic_write_json(path, packet)
            changed_packets += 1
    return changed_packets


def _packet_ocr_projection(observation: OCRObservation) -> dict[str, Any]:
    """Return exactly the narrow OCR shape accepted by ``VisionPacket``."""

    return {
        "observation_id": observation.observation_id,
        "frame_id": observation.frame_id,
        "crop_id": observation.crop_id,
        "raw_engine_text": observation.raw_engine_text,
        "normalized_interpretation": observation.normalized_interpretation,
        "confidence": observation.confidence,
        "bounding_region": observation.bounding_region,
        "uncertain_characters": [dict(item) for item in observation.uncertain_characters],
    }


def _packet_ocr_projection_from_canonical(item: Mapping[str, Any]) -> dict[str, Any]:
    uncertain: list[dict[str, Any]] = []
    for value in item.get("uncertain_characters", []):
        if isinstance(value, Mapping):
            uncertain.append(dict(value))
        elif isinstance(value, str):
            uncertain.append(
                {
                    "text": value,
                    "confidence": None,
                    "reason": "Canonical OCR uncertainty was preserved without source geometry.",
                }
            )
    return {
        key: uncertain if key == "uncertain_characters" else item.get(key)
        for key in (
            "observation_id",
            "frame_id",
            "crop_id",
            "raw_engine_text",
            "normalized_interpretation",
            "confidence",
            "bounding_region",
            "uncertain_characters",
        )
        if key in item or key == "uncertain_characters"
    }


def repair_project_ocr_packets(project_dir: Path) -> dict[str, Any]:
    """Repair packet-only schema drift without invoking an OCR engine."""

    project_root = project_dir.expanduser().resolve(strict=True)
    project = _load_project(project_root)
    observations = {
        str(item["observation_id"]): _packet_ocr_projection_from_canonical(item)
        for item in project.get("ocr_observations", [])
        if isinstance(item, Mapping) and item.get("observation_id")
    }
    changed = _replace_packet_ocr(project_root, observations)
    invalid: list[str] = []
    for path in _packet_files(project_root):
        try:
            VisionPacket.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            invalid.append(path.name)
    if invalid:
        raise ValidationFailure("Packet OCR repair left invalid packets: " + ", ".join(invalid[:8]))
    return {
        "project_dir": str(project_root),
        "status": "packets_repaired" if changed else "packets_already_valid",
        "changed_packet_count": changed,
        "invalid_packet_count": 0,
    }


def refresh_project_ocr(
    project_dir: Path,
    *,
    workers: int | None = None,
    language: str | None = None,
    adapter: OCRAdapter | None = None,
) -> dict[str, Any]:
    """Refresh persisted OCR observations using existing evidence PNGs only.

    This command is a targeted migration for projects generated before the
    quote-safe Tesseract TSV parser. It never decodes source media, reruns ASR,
    changes evidence pixels, or discards semantic observation history. Packet
    OCR context and canonical OCR observations are updated atomically after all
    recognition workers succeed; validation and Markdown rendering then run once.
    """

    project_root = project_dir.expanduser().resolve(strict=True)
    project = _load_project(project_root)
    frames = {
        str(frame.get("frame_id")): frame
        for frame in project.get("frames", [])
        if isinstance(frame, Mapping) and frame.get("frame_id")
    }
    existing = [
        dict(item)
        for item in project.get("ocr_observations", [])
        if isinstance(item, Mapping) and item.get("observation_id")
    ]
    if not existing:
        return {
            "project_dir": str(project_root),
            "status": "no_ocr_observations",
            "observation_count": 0,
            "changed_observation_count": 0,
            "changed_packet_count": 0,
        }
    selected_adapter = adapter or TesseractOCRAdapter()
    if not selected_adapter.available():
        raise BlockedError("Tesseract is unavailable; configure VSR_TESSERACT_PATH or install it")
    worker_count = _refresh_workers(workers)
    jobs: list[tuple[dict[str, Any], Path]] = []
    stale_observation_ids: list[str] = []
    for item in existing:
        frame_id = str(item.get("frame_id"))
        frame = frames.get(frame_id)
        if frame is None:
            # Older runs may retain an OCR checkpoint record for a candidate
            # frame that was later rejected from the final evidence ledger.
            # There is no safe image to re-read; preserve that historical
            # record and report it instead of inventing a replacement.
            stale_observation_ids.append(str(item["observation_id"]))
            continue
        jobs.append((item, _frame_path(project_root, frame)))

    def recognize(job: tuple[dict[str, Any], Path]) -> tuple[str, OCRObservation]:
        item, image_path = job
        observation_id = str(item["observation_id"])
        fresh = selected_adapter.recognize(
            image_path,
            frame_id=str(item["frame_id"]),
            observation_id=observation_id,
            crop_id=item.get("crop_id") if isinstance(item.get("crop_id"), str) else None,
            language=language,
        )
        human_decision = item.get("human_decision")
        if isinstance(human_decision, str) and human_decision:
            fresh = replace(fresh, human_decision=human_decision)
        return observation_id, fresh

    fresh_by_id: dict[str, OCRObservation] = {}
    if jobs:
        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="vsr-ocr-refresh"
        ) as pool:
            futures = {pool.submit(recognize, job): job[0] for job in jobs}
            try:
                for future in as_completed(futures):
                    observation_id, fresh = future.result()
                    fresh_by_id[observation_id] = fresh
            except BaseException as exc:
                for future in futures:
                    future.cancel()
                raise ValidationFailure(f"OCR refresh failed before commit: {exc}") from exc

    refreshed: dict[str, dict[str, Any]] = {
        observation_id: to_schema_observation(observation).model_dump(mode="json")
        for observation_id, observation in fresh_by_id.items()
    }
    packet_refreshed = {
        observation_id: _packet_ocr_projection(observation)
        for observation_id, observation in fresh_by_id.items()
    }
    changed_ids = {
        observation_id
        for observation_id, old in ((str(item["observation_id"]), item) for item in existing)
        if observation_id in refreshed
        and _ocr_signature(old) != _ocr_signature(refreshed[observation_id])
    }
    project["ocr_observations"] = [
        refreshed.get(str(item["observation_id"]), item) for item in existing
    ]
    packet_count = _replace_packet_ocr(project_root, packet_refreshed)

    manifest = project.setdefault("manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
        project["manifest"] = manifest
    usage = manifest.setdefault("provider_usage", [])
    if isinstance(usage, list):
        usage.append(
            {
                "provider": f"{selected_adapter.__class__.__module__}.{selected_adapter.__class__.__name__}",
                "purpose": "ocr-refresh",
                "route": "local",
                "parser_schema": _REFRESH_SCHEMA,
                "observations": len(existing),
                "refreshed_observations": len(jobs),
                "stale_observations": len(stale_observation_ids),
                "changed_observations": len(changed_ids),
                "workers": worker_count,
                "language": language,
            }
        )
    manifest["last_ocr_refresh"] = {
        "at_utc": _now(),
        "schema": _REFRESH_SCHEMA,
        "changed_observation_count": len(changed_ids),
        "changed_packet_count": packet_count,
        "stale_observation_count": len(stale_observation_ids),
    }

    project["audit"] = audit_project(project)
    project["project_status"] = project["audit"]["final_project_status"]
    project["status_reason"] = (
        "Consequential visual or wording uncertainty remains in the review queue."
        if project["project_status"] == "review_required"
        else "All mandatory automatic checks passed; no human verification is implied."
    )
    canonical_path = project_root / ".state" / "canonical-project.json"
    atomic_write_json(
        canonical_path,
        project,
        compact=canonical_compact_for_payload(canonical_path, project),
    )
    atomic_write_json(project_root / ".state" / "audit.json", project["audit"])
    atomic_write_json(project_root / ".state" / "review-queue.json", project.get("review_items", []))
    markdown = next(project_root.glob("*.md"), project_root / f"{project_root.name}.md")
    render_to_path(project, markdown)
    validation = validate_project(project_root, verify_metadata=True, use_cached_file_hash=True)
    if not validation.valid:
        raise ValidationFailure("OCR refresh validation failed: " + "; ".join(validation.errors))
    run_key = manifest.get("run_cache_key")
    if isinstance(run_key, str) and run_key:
        write_validation_receipt(
            project_root,
            project,
            run_cache_key=run_key,
            validation=validation,
        )
    return {
        "project_dir": str(project_root),
        "status": project["project_status"],
        "observation_count": len(existing),
        "refreshed_observation_count": len(jobs),
        "stale_observation_count": len(stale_observation_ids),
        "stale_observation_ids": stale_observation_ids,
        "changed_observation_count": len(changed_ids),
        "changed_packet_count": packet_count,
        "workers": worker_count,
        "parser_schema": _REFRESH_SCHEMA,
        "validation_errors": validation.errors,
        "semantic_reanalysis_recommended": bool(changed_ids),
    }


__all__ = ["refresh_project_ocr", "repair_project_ocr_packets"]
