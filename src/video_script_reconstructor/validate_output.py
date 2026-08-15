from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from stat import S_ISDIR, S_ISLNK, S_ISREG
from threading import Lock
from typing import Any

from .errors import SecurityError, ValidationFailure
from .render_markdown import TOP_SECTIONS
from .security import ContainmentSnapshot, atomic_write_json, safe_relative_path, sha256_file

_ANCHOR = re.compile(r'<a\s+id="([A-Za-z0-9-]+)"\s*></a>')
_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
# Markdown alt text may contain escaped brackets from OCR (for example
# ``\[Pickup City\]``).  Consume escaped characters as a unit so the closing
# bracket in the alt text is not mistaken for the end of the image label.
_IMAGE = re.compile(r"!\[(?:\\.|[^\]\\])*\]\(([^)]+)\)")
_FILE_HASH_CACHE_LIMIT = 4096
_FILE_HASH_CACHE: dict[Path, tuple[tuple[int, int, int, int], str]] = {}
_CANONICAL_CACHE_LIMIT = 2
_CANONICAL_CACHE: dict[Path, tuple[tuple[int, int, int, int], dict[str, Any]]] = {}
_CANONICAL_MODEL_CACHE: dict[Path, tuple[tuple[int, int, int, int], Any]] = {}
_AUDIT_CACHE: dict[Path, tuple[tuple[int, int, int, int], dict[str, Any]]] = {}
_VALIDATION_RESULT_CACHE_LIMIT = 8
_VALIDATION_RESULT_CACHE: dict[tuple[Path, bool, bool], tuple[tuple[Any, ...], Any]] = {}
_METADATA_VERIFY_CACHE_LIMIT = 4096
_METADATA_VERIFY_CACHE: dict[
    tuple[Path, tuple[int, int, int, int], str, str | None, bool], Any
] = {}
_VALIDATION_CACHE_LOCK = Lock()


def _metadata_verify_workers() -> int:
    """Choose a bounded image-verification pool without weakening checks."""

    override = os.environ.get("VSR_VALIDATOR_METADATA_WORKERS", "").strip()
    if override:
        try:
            return max(1, min(16, int(override)))
        except ValueError:
            pass
    logical_cpus = os.cpu_count() or 1
    if logical_cpus >= 16:
        return min(16, logical_cpus)
    return max(1, min(8, logical_cpus // 2))


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def _cached_file_hash(path: Path) -> str:
    """Return a stat-bound hash for repeated internal validation passes.

    The public validator never calls this helper.  Internal pipeline calls have
    already completed one full hash and only reuse it while the file's size,
    timestamps, and inode remain unchanged; a changed signature forces a new
    cryptographic read.  The bounded process-local cache cannot alter the
    independent public validation contract.
    """

    signature = _stat_signature(path)
    with _VALIDATION_CACHE_LOCK:
        cached = _FILE_HASH_CACHE.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    digest = sha256_file(path)
    with _VALIDATION_CACHE_LOCK:
        if len(_FILE_HASH_CACHE) >= _FILE_HASH_CACHE_LIMIT:
            _FILE_HASH_CACHE.pop(next(iter(_FILE_HASH_CACHE)))
        _FILE_HASH_CACHE[path] = (signature, digest)
    return digest


def _metadata_payload_cache_key(payload: Any) -> str:
    """Return a stable identity for an expected metadata mirror.

    Canonical image payloads carry their own cryptographic payload digest.  A
    deterministic fallback keeps this helper useful for legacy records that do
    not yet expose that field, without changing the public validation path.
    """

    integrity = (
        payload.get("integrity")
        if isinstance(payload, Mapping)
        else getattr(payload, "integrity", None)
    )
    digest = (
        integrity.get("payload_digest")
        if isinstance(integrity, Mapping)
        else getattr(integrity, "payload_digest", None)
    )
    if digest:
        return str(digest)
    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError):
        serialized = repr(payload).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _cached_verify_embedded_metadata(
    path: Path,
    canonical_payload: Any,
    *,
    expected_pixel_hash: str | None,
    verifier: Callable[..., Any],
    canonical_payload_prevalidated: bool = False,
) -> Any:
    """Reuse an unchanged internal metadata verification result.

    The cache is deliberately opt-in: callers use it only after the internal
    canonical-file-hash check has succeeded.  Its key includes the file stat
    signature, expected canonical payload identity, and expected pixel hash, so
    changing either the bytes or the canonical mirror forces a fresh verifier
    call.  The public validator never calls this helper.
    """

    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    key = (
        path,
        signature,
        _metadata_payload_cache_key(canonical_payload),
        expected_pixel_hash,
        canonical_payload_prevalidated,
    )
    with _VALIDATION_CACHE_LOCK:
        cached = _METADATA_VERIFY_CACHE.get(key)
    if cached is not None:
        return cached
    verifier_kwargs: dict[str, Any] = {"expected_pixel_hash": expected_pixel_hash}
    if canonical_payload_prevalidated:
        verifier_kwargs["canonical_payload_prevalidated"] = True
    verified = verifier(path, canonical_payload, **verifier_kwargs)
    with _VALIDATION_CACHE_LOCK:
        if len(_METADATA_VERIFY_CACHE) >= _METADATA_VERIFY_CACHE_LIMIT:
            _METADATA_VERIFY_CACHE.pop(next(iter(_METADATA_VERIFY_CACHE)))
        _METADATA_VERIFY_CACHE[key] = verified
    return verified


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    # Internal cache-hit metadata. Public validation leaves this unset; a
    # trusted receipt carries it so the pipeline can avoid parsing the large
    # canonical project merely to recover the final status.
    project_status: str | None = None

    def require_valid(self) -> None:
        if not self.valid:
            raise ValidationFailure("; ".join(self.errors))


def _validation_inventory_signature(
    inventory: list[dict[str, Any]],
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """Normalize a stat-bound inventory into a compact cache key."""

    normalized: list[tuple[str, str, tuple[int, ...]]] = []
    for record in inventory:
        raw_signature = record.get("signature", ())
        signature = (
            tuple(int(value) for value in raw_signature)
            if isinstance(raw_signature, list | tuple)
            else ()
        )
        normalized.append(
            (str(record.get("path", "")), str(record.get("kind", "")), signature)
        )
    return tuple(normalized)


def _validation_state_signature(
    project_dir: Path,
    *,
    inventory: list[dict[str, Any]] | None = None,
) -> tuple[tuple[int, int, int, int], tuple[tuple[str, str, tuple[int, ...]], ...]] | None:
    """Return the stat-bound internal validation cache identity.

    This is intentionally used only by the internal ``use_cached_file_hash``
    path.  The public validator remains independent and always performs the
    full checks.  Canonical state and every generated file are included, while
    volatile manifest/receipt files remain excluded by ``_validation_inventory``.
    """

    try:
        canonical_signature = _stat_signature(
            project_dir / ".state" / "canonical-project.json"
        )
        current_inventory = (
            inventory if inventory is not None else _validation_inventory(project_dir)
        )
    except OSError:
        return None
    return canonical_signature, _validation_inventory_signature(current_inventory)


def _clone_validation_result(result: ValidationResult) -> ValidationResult:
    """Return an isolated result so callers cannot mutate the cached proof."""

    return ValidationResult(
        result.valid,
        errors=list(result.errors),
        warnings=list(result.warnings),
        checks=dict(result.checks),
        project_status=result.project_status,
    )


def _cached_validation_result(
    project_dir: Path,
    *,
    verify_metadata: bool,
    use_cached_file_hash: bool,
) -> tuple[ValidationResult | None, tuple[tuple[int, int, int, int], tuple[tuple[str, str, tuple[int, ...]], ...]] | None]:
    """Look up a stat-bound internal validation result and its current identity."""

    if not use_cached_file_hash:
        return None, None
    state_signature = _validation_state_signature(project_dir)
    if state_signature is None:
        return None, None
    cache_key = (project_dir, verify_metadata, use_cached_file_hash)
    with _VALIDATION_CACHE_LOCK:
        cached = _VALIDATION_RESULT_CACHE.get(cache_key)
    if cached is None or cached[0] != state_signature:
        return None, state_signature
    return _clone_validation_result(cached[1]), state_signature


def _remember_validation_result(
    project_dir: Path,
    *,
    verify_metadata: bool,
    use_cached_file_hash: bool,
    result: ValidationResult,
) -> None:
    """Store one validated internal result against the final project state."""

    if not use_cached_file_hash:
        return
    state_signature = _validation_state_signature(project_dir)
    if state_signature is None:
        return
    cache_key = (project_dir, verify_metadata, use_cached_file_hash)
    with _VALIDATION_CACHE_LOCK:
        if len(_VALIDATION_RESULT_CACHE) >= _VALIDATION_RESULT_CACHE_LIMIT:
            _VALIDATION_RESULT_CACHE.pop(next(iter(_VALIDATION_RESULT_CACHE)))
        _VALIDATION_RESULT_CACHE[cache_key] = (state_signature, _clone_validation_result(result))


_VALIDATION_RECEIPT_RELATIVE = ".state/validation-receipt.json"
_VALIDATION_RECEIPT_DYNAMIC = {
    ".state/canonical-project.json",
    ".state/run-manifest.json",
    _VALIDATION_RECEIPT_RELATIVE,
}
_RECEIPT_PROJECT_STATUSES = {
    "processing",
    "blocked",
    "review_required",
    "automatically_checked",
    "human_reviewed",
    "fully_verified",
    "failed",
}


def _canonical_state_digest(canonical: Mapping[str, Any]) -> str:
    """Hash canonical evidence state while excluding volatile manifest telemetry."""

    stable = {key: value for key, value in canonical.items() if key != "manifest"}
    encoded = json.dumps(
        stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_inventory(root: Path) -> list[dict[str, Any]]:
    """Capture a stat-bound inventory for the internal validation fast path.

    Canonical/run manifests are deliberately excluded because final resource
    telemetry rewrites them after the first proof.  The canonical state digest
    still binds all non-volatile evidence fields; every other file and symlink
    remains part of the inventory so additions, removals, and path escapes
    invalidate the receipt before it can bypass public validation.
    """

    records: list[dict[str, Any]] = []

    def walk(directory: Path, prefix: str = "") -> None:
        with os.scandir(directory) as stream:
            entries = sorted(stream, key=lambda entry: (entry.name.casefold(), entry.name))
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            if relative in _VALIDATION_RECEIPT_DYNAMIC:
                continue
            entry_stat = entry.stat(follow_symlinks=False)
            signature = [
                int(entry_stat.st_size),
                int(entry_stat.st_mtime_ns),
                int(entry_stat.st_ctime_ns),
                int(entry_stat.st_ino),
            ]
            if S_ISLNK(entry_stat.st_mode):
                records.append({"path": relative, "kind": "symlink", "signature": signature})
            elif S_ISDIR(entry_stat.st_mode):
                records.append({"path": relative, "kind": "directory"})
                walk(Path(entry.path), relative)
            elif S_ISREG(entry_stat.st_mode):
                records.append({"path": relative, "kind": "file", "signature": signature})

    walk(root)
    return records


def _iter_output_files(root: Path, *, skip_state: bool = False) -> Iterator[Path]:
    """Yield output files with one directory-entry walk.

    Validation scans the generated tree twice for the document and evidence
    contracts.  ``Path.rglob`` allocates a ``Path`` for every entry and then
    performs another metadata lookup for ``is_file``.  A bounded scandir walk
    keeps the same symlink-file behavior while avoiding symlinked directories;
    the hidden ``.state`` subtree can be pruned before opening its children.
    """

    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            ordered_entries = sorted(entries, key=lambda entry: (entry.name.casefold(), entry.name))
        for entry in ordered_entries:
            if skip_state and entry.name == ".state":
                continue
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))
                continue
            # ``Path.is_file`` followed symlinks in the previous rglob path;
            # retain that behavior for symlinked output files.
            if entry.is_file():
                yield Path(entry.path)


def write_validation_receipt(
    project_dir: Path,
    canonical: Mapping[str, Any],
    *,
    run_cache_key: str,
    validation: ValidationResult,
) -> Path:
    """Persist a stat-bound proof for safe unchanged cache-hit resumes.

    This receipt is an internal acceleration layer, not a replacement for the
    public validator.  It is written only after a valid proof and is checked
    against canonical state plus the complete generated-file inventory before a
    cache-hit run can skip re-verification.
    """

    path = project_dir / _VALIDATION_RECEIPT_RELATIVE
    canonical_path = project_dir / ".state" / "canonical-project.json"
    atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "run_cache_key": run_cache_key,
            "canonical_state_digest": _canonical_state_digest(canonical),
            "canonical_file_signature": list(_stat_signature(canonical_path)),
            "project_status": canonical.get("project_status"),
            "cache_contract_complete": not canonical.get("frames")
            or bool(canonical.get("sufficiency_decisions")),
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
            "checks": dict(validation.checks),
            "inventory": _validation_inventory(project_dir),
        },
        compact=True,
    )
    return path


def refresh_validation_receipt_signature(project_dir: Path) -> bool:
    """Bind a receipt to the final canonical-file signature after telemetry writes.

    Resource telemetry intentionally rewrites the volatile manifest after a
    receipt is created.  Refreshing only this receipt field avoids forcing the
    next warm run to re-serialize the entire canonical project merely because
    that manifest write changed the canonical file's stat signature.
    """

    path = project_dir / _VALIDATION_RECEIPT_RELATIVE
    canonical_path = project_dir / ".state" / "canonical-project.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        signature = list(_stat_signature(canonical_path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("canonical_file_signature") == signature:
        return True
    payload["canonical_file_signature"] = signature
    try:
        atomic_write_json(path, payload, compact=True)
    except OSError:
        return False
    return True


def read_trusted_validation_receipt(
    project_dir: Path,
    canonical: Mapping[str, Any] | None,
    *,
    run_cache_key: str,
) -> ValidationResult | None:
    """Return a cached valid result only when every receipt invariant matches."""

    path = project_dir / _VALIDATION_RECEIPT_RELATIVE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema_version") != "1.0" or payload.get("run_cache_key") != run_cache_key:
        return None
    canonical_file_signature = payload.get("canonical_file_signature")
    try:
        current_canonical_signature = list(
            _stat_signature(project_dir / ".state" / "canonical-project.json")
        )
    except OSError:
        return None
    if canonical_file_signature != current_canonical_signature:
        # A manifest-only telemetry rewrite changes the canonical file stat but
        # not its stable evidence state.  Preserve that compatibility fallback
        # for receipts written before the final signature refresh; all other
        # canonical edits still fail the digest check and force validation.
        if canonical is None or payload.get("canonical_state_digest") != _canonical_state_digest(
            canonical
        ):
            return None
        canonical_file_signature = current_canonical_signature
        payload = dict(payload)
        payload["canonical_file_signature"] = canonical_file_signature
        try:
            atomic_write_json(path, payload, compact=True)
        except OSError:
            return None
    elif payload.get("canonical_state_digest") is None:
        return None
    expected_inventory = payload.get("inventory")
    if not isinstance(expected_inventory, list):
        return None
    try:
        current_inventory = _validation_inventory(project_dir)
    except OSError:
        return None
    if current_inventory != expected_inventory:
        return None
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or checks.get("metadata_verified") is not True:
        return None
    errors = payload.get("errors")
    warnings = payload.get("warnings")
    if not isinstance(errors, list) or not all(isinstance(item, str) for item in errors):
        return None
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        return None
    project_status = payload.get("project_status")
    if not isinstance(project_status, str):
        return None
    if project_status not in _RECEIPT_PROJECT_STATUSES:
        return None
    if payload.get("cache_contract_complete") is not True:
        return None
    return ValidationResult(
        True,
        errors=list(errors),
        warnings=list(warnings),
        checks=dict(checks),
        project_status=project_status,
    )


def _load_canonical(
    project_dir: Path,
    *,
    use_cache: bool = False,
) -> dict[str, Any] | None:
    path = project_dir / ".state" / "canonical-project.json"
    signature: tuple[int, int, int, int] | None = None
    if use_cache:
        try:
            signature = _stat_signature(path)
        except OSError:
            return None
        with _VALIDATION_CACHE_LOCK:
            cached = _CANONICAL_CACHE.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
    elif not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if use_cache and signature is not None:
        with _VALIDATION_CACHE_LOCK:
            if len(_CANONICAL_CACHE) >= _CANONICAL_CACHE_LIMIT:
                _CANONICAL_CACHE.pop(next(iter(_CANONICAL_CACHE)))
            _CANONICAL_CACHE[path] = (signature, value)
    return value


def _cached_canonical_model(
    path: Path,
    payload: dict[str, Any],
    validator: Callable[[Any], Any],
    *,
    projection: Callable[[Any], Any] | None = None,
) -> Any:
    """Reuse an unchanged schema model or lightweight projection internally."""

    signature = _stat_signature(path)
    with _VALIDATION_CACHE_LOCK:
        cached = _CANONICAL_MODEL_CACHE.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    model = validator(payload)
    cached_value = projection(model) if projection is not None else model
    with _VALIDATION_CACHE_LOCK:
        if len(_CANONICAL_MODEL_CACHE) >= _CANONICAL_CACHE_LIMIT:
            _CANONICAL_MODEL_CACHE.pop(next(iter(_CANONICAL_MODEL_CACHE)))
        _CANONICAL_MODEL_CACHE[path] = (signature, cached_value)
    return cached_value


def _cached_audit(
    path: Path,
    payload: dict[str, Any],
    auditor: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """Reuse pure audit output for unchanged internal canonical state."""

    signature = _stat_signature(path)
    with _VALIDATION_CACHE_LOCK:
        cached = _AUDIT_CACHE.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    report = auditor(payload)
    with _VALIDATION_CACHE_LOCK:
        if len(_AUDIT_CACHE) >= _CANONICAL_CACHE_LIMIT:
            _AUDIT_CACHE.pop(next(iter(_AUDIT_CACHE)))
        _AUDIT_CACHE[path] = (signature, report)
    return report


def validate_project(
    project_dir: Path,
    *,
    verify_metadata: bool = True,
    use_cached_file_hash: bool = False,
) -> ValidationResult:
    """Validate one generated project and its portable evidence.

    ``use_cached_file_hash`` is an internal pipeline optimization.  When a
    frame already carries a canonical whole-file SHA-256, the validator checks
    that exact byte digest and uses the embedded pixel hash for the metadata
    consistency check, avoiding a second full PNG pixel decode.  The default
    remains the independent decoded-pixel path for public validation and for
    artifacts without a canonical file hash.
    """
    root = project_dir.resolve(strict=True)
    cached_result, _cache_state = _cached_validation_result(
        root,
        verify_metadata=verify_metadata,
        use_cached_file_hash=use_cached_file_hash,
    )
    if cached_result is not None:
        return cached_result
    containment_snapshot = ContainmentSnapshot()
    errors: list[str] = []
    warnings: list[str] = []
    metadata_workers = 0
    # One tree walk is enough for both output-contract inventories. Keeping the
    # traversal shared matters on large evidence trees and does not alter the
    # independent public validation semantics.
    markdown_files: list[Path] = []
    html_files: list[Path] = []
    # ``.state`` is intentionally hidden canonical/runtime state. Internal
    # review handoffs may contain human-readable notes, but they are not
    # deliverable documents and must not violate the one-document contract.
    for path in _iter_output_files(root, skip_state=True):
        suffix = path.suffix.casefold()
        if suffix == ".md":
            markdown_files.append(path)
        elif suffix in {".html", ".htm"}:
            html_files.append(path)
    if len(markdown_files) != 1:
        errors.append(
            f"output_contract: expected exactly one Markdown file, found {len(markdown_files)}"
        )
    if html_files:
        errors.append(f"output_contract: HTML artifacts are forbidden ({len(html_files)} found)")
    canonical = _load_canonical(root, use_cache=use_cached_file_hash)
    if canonical is None:
        errors.append("canonical_state: missing or invalid .state/canonical-project.json")
    if len(markdown_files) != 1:
        return ValidationResult(
            False,
            errors,
            warnings,
            {"markdown_count": len(markdown_files), "html_count": len(html_files)},
        )

    markdown_path = markdown_files[0]
    text = markdown_path.read_text(encoding="utf-8")
    positions: list[int] = []
    for section in TOP_SECTIONS:
        heading = f"## {section}"
        count = text.count(heading)
        if count != 1:
            errors.append(f"markdown_contract: section {section!r} occurs {count} times")
        positions.append(text.find(heading))
    if any(position < 0 for position in positions) or positions != sorted(positions):
        errors.append("markdown_contract: required sections are missing or out of order")

    anchors = _ANCHOR.findall(text)
    duplicate_anchors = sorted({anchor for anchor in anchors if anchors.count(anchor) > 1})
    if duplicate_anchors:
        errors.append(f"navigation: duplicate anchors: {', '.join(duplicate_anchors)}")
    anchor_set = set(anchors)
    for target in _LINK.findall(text):
        target = target.strip().split(" ", 1)[0]
        if target.startswith("#"):
            if target[1:] not in anchor_set:
                heading_target = target[1:]
                known_heading_targets = {
                    re.sub(r"[^a-z0-9-]", "", re.sub(r"\s+", "-", section.casefold()))
                    for section in TOP_SECTIONS
                }
                if heading_target not in known_heading_targets:
                    errors.append(f"navigation: missing anchor target {target}")
            continue
        if "://" in target or target.startswith("mailto:"):
            continue
        try:
            resolved = safe_relative_path(
                root,
                target,
                root_resolved=root,
                containment_snapshot=containment_snapshot,
            )
        except SecurityError as exc:
            errors.append(f"path_containment: {exc}")
            continue
        if not resolved.is_file():
            errors.append(f"linked_artifact: missing {target}")

    image_links = set(_IMAGE.findall(text))
    evidence_root = root / "evidence"
    evidence_images = (
        {
            path.relative_to(root).as_posix()
            for path in _iter_output_files(evidence_root)
            if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}
        }
        if evidence_root.exists()
        else set()
    )
    orphaned = sorted(evidence_images - image_links)
    if orphaned:
        errors.append(f"visual_evidence: orphan final images: {', '.join(orphaned)}")

    canonical_evidence_models: list[Any] | None = None
    if canonical is not None:
        try:
            from .schemas import CanonicalProject

            canonical_path = root / ".state" / "canonical-project.json"
            if use_cached_file_hash:
                canonical_evidence_models = _cached_canonical_model(
                    canonical_path,
                    canonical,
                    CanonicalProject.model_validate,
                    projection=lambda model: list(model.evidence_image_metadata),
                )
            else:
                validated_canonical_project = CanonicalProject.model_validate(canonical)
                canonical_evidence_models = list(validated_canonical_project.evidence_image_metadata)
        except Exception as exc:
            errors.append(f"canonical_schema: {exc}")
        timeline_records = canonical.get("timeline", [])
        if timeline_records:
            try:
                from .timeline import TimelineItem, validate_timeline

                timeline = [TimelineItem(**item) for item in timeline_records]
                media_duration = (canonical.get("media") or {}).get("duration_ms")
                timeline_check = validate_timeline(
                    timeline, media_duration_ms=media_duration, require_sorted=True
                )
                if not timeline_check.valid:
                    errors.extend(f"timeline: {error}" for error in timeline_check.errors)
            except Exception as exc:
                errors.append(f"timeline: invalid canonical timeline: {exc}")
        timeline_path = root / ".state" / "timeline" / "timeline.json"
        if timeline_records and not timeline_path.is_file():
            errors.append("timeline: canonical timeline artifact is missing")
        canonical_status = canonical.get("audit", {}).get(
            "final_project_status", canonical.get("project_status")
        )
        metadata_match = re.search(r'^project_status:\s*"([^"\n]+)"', text, re.MULTILINE)
        displayed = metadata_match.group(1) if metadata_match else None
        if canonical_status and displayed != canonical_status:
            errors.append(f"status: Markdown={displayed!r}, canonical={canonical_status!r}")
        frames = canonical.get("frames", canonical.get("snapshots", []))
        known_frames = {
            str(frame.get("frame_id") or frame.get("image_id")): frame for frame in frames
        }
        canonical_payloads = {
            str(item.get("image", {}).get("image_id")): item
            for item in canonical.get("evidence_image_metadata", [])
        }
        known_segments = {
            str(item.get("segment_id")): item for item in canonical.get("transcript_segments", [])
        }
        known_claim_records = {
            str(item.get("claim_id")): item for item in canonical.get("image_claims", [])
        }
        required_ids = [
            *[str(item.get("chapter_id")) for item in canonical.get("chapters", [])],
            *[str(item.get("block_id")) for item in canonical.get("script_blocks", [])],
            *[
                str(item.get("review_id"))
                for item in canonical.get("review_items", [])
                if not item.get("decision")
            ],
            *list(known_frames),
        ]
        for required_id in required_ids:
            if required_id and required_id not in anchor_set:
                errors.append(
                    f"navigation: missing anchor target #{required_id} (canonical anchor required)"
                )
        for block in canonical.get("script_blocks", canonical.get("blocks", [])):
            for segment_id in block.get("transcript_segment_ids", block.get("segment_ids", [])):
                if str(segment_id) not in known_segments:
                    errors.append(
                        f"trace: block {block.get('block_id')} cites unknown segment {segment_id}"
                    )
            for frame_id in block.get("frame_ids", []):
                if str(frame_id) not in known_frames:
                    errors.append(
                        f"trace: block {block.get('block_id')} cites unknown frame {frame_id}"
                    )
            for claim_id in block.get("image_claim_ids", []):
                if str(claim_id) not in known_claim_records:
                    errors.append(
                        f"trace: block {block.get('block_id')} cites unknown claim {claim_id}"
                    )
                elif known_claim_records[str(claim_id)].get("status") != "supported":
                    errors.append(
                        f"trace: block {block.get('block_id')} consumes non-supported claim {claim_id}"
                    )
            visual = str(block.get("visual_description") or "")
            if (
                visual
                and visual
                not in {
                    "[no visual source available]",
                    "[visual evidence retained; semantic description pending review]",
                }
                and block.get("frame_ids")
                and not block.get("image_claim_ids")
            ):
                errors.append(
                    f"support: visual statement in {block.get('block_id')} lacks an image-claim citation"
                )
        for frame_id, frame in known_frames.items():
            parent = frame.get("parent_full_frame_id")
            if parent and str(parent) not in known_frames:
                errors.append(f"visual_evidence: crop {frame_id} lacks parent {parent}")
            pts = frame.get("pts")
            time_base = frame.get("time_base")
            actual_ms = frame.get("actual_ms")
            if pts is not None and time_base and actual_ms is not None:
                try:
                    measured_ms = round(float(Fraction(str(time_base)) * int(pts)) * 1000)
                except (ValueError, ZeroDivisionError):
                    errors.append(f"timeline: frame {frame_id} has an invalid PTS time base")
                else:
                    if abs(measured_ms - int(actual_ms)) > 1:
                        errors.append(
                            f"timeline: frame {frame_id} actual time disagrees with raw PTS/time base"
                        )

        observations = {
            str(item.get("observation_id")): item
            for item in canonical.get("visual_observations", [])
        }
        revisions_by_image: dict[str, list[dict[str, Any]]] = {}
        for revision in canonical.get("metadata_revisions", []):
            revisions_by_image.setdefault(str(revision.get("image_id")), []).append(revision)
            for observation_id in revision.get("observation_ids", []):
                if str(observation_id) not in observations:
                    errors.append(
                        f"metadata_revision: {revision.get('revision_id')} cites missing observation {observation_id}"
                    )
        for image_id, revisions in revisions_by_image.items():
            ordered = sorted(revisions, key=lambda item: int(item.get("revision_number", 0)))
            for index, revision in enumerate(ordered):
                expected_number = index + 1
                if revision.get("revision_number") != expected_number:
                    errors.append(
                        f"metadata_revision: {image_id} expected revision number {expected_number}, "
                        f"found {revision.get('revision_number')}"
                    )
                expected_base = None if index == 0 else ordered[index - 1].get("revision_id")
                if revision.get("base_revision_id") != expected_base:
                    errors.append(
                        f"metadata_revision: {revision.get('revision_id')} base is stale or broken"
                    )
            frame = known_frames.get(image_id)
            if (
                frame
                and ordered
                and frame.get("latest_revision_id") != ordered[-1].get("revision_id")
            ):
                errors.append(
                    f"metadata_revision: frame {image_id} does not cite the latest revision"
                )
        for claim_id, claim in known_claim_records.items():
            if claim.get("status") == "supported" and not claim.get("supporting_observation_ids"):
                errors.append(f"image_claim: supported claim {claim_id} lacks an observation")
            for observation_id in claim.get("supporting_observation_ids", []) + claim.get(
                "contradicting_observation_ids", []
            ):
                if str(observation_id) not in observations:
                    errors.append(
                        f"image_claim: {claim_id} cites missing observation {observation_id}"
                    )

        from .audit import audit_project

        canonical_path = root / ".state" / "canonical-project.json"
        recalculated = (
            _cached_audit(canonical_path, canonical, audit_project)
            if use_cached_file_hash
            else audit_project(canonical)
        )
        stored_audit = canonical.get("audit", {})
        if recalculated.get("blocking_failures") != stored_audit.get("blocking_failures"):
            errors.append(
                "audit_staleness: stored blocking failures differ from deterministic recalculation"
            )
        stored_coverage = stored_audit.get("source_segment_coverage", {})
        if (
            isinstance(stored_coverage, dict)
            and recalculated.get("source_segment_coverage") != stored_coverage
        ):
            errors.append("audit_staleness: stored transcript coverage differs from recalculation")

        if verify_metadata:
            try:
                from .image_metadata import read_embedded_metadata, verify_embedded_metadata
                from .schemas import EvidenceImageMetadata
            except ImportError:
                errors.append("image_metadata: production verifier is unavailable")
            else:
                canonical_payload_models: dict[str, EvidenceImageMetadata] = {}
                if canonical_evidence_models is not None:
                    canonical_payload_models = {
                        str(item.image.image_id): item
                        for item in canonical_evidence_models
                    }

                def verify_frame_metadata(
                    item: tuple[str, Mapping[str, Any]],
                ) -> list[str]:
                    frame_id, frame = item
                    frame_errors: list[str] = []
                    relative = frame.get("full_frame_path") or frame.get("path")
                    if not relative:
                        return [f"image_metadata: frame {frame_id} lacks a path"]
                    try:
                        image_path = safe_relative_path(
                            root,
                            str(relative),
                            root_resolved=root,
                            containment_snapshot=containment_snapshot,
                        )
                        expected = (
                            canonical_payloads.get(frame_id)
                            or frame.get("metadata")
                            or frame.get("evidence_image_metadata")
                        )
                        if expected:
                            expected_model = canonical_payload_models.get(frame_id)
                            expected_for_verify: Any = expected_model or expected
                            expected_prevalidated = expected_model is not None
                            expected_pixel_hash: str | None = None
                            if use_cached_file_hash and frame.get("file_hash"):
                                actual_file_hash = _cached_file_hash(image_path)
                                if str(frame["file_hash"]) != actual_file_hash:
                                    frame_errors.append(
                                        f"image_metadata: frame {frame_id} whole-file hash disagrees with canonical state"
                                    )
                                else:
                                    expected_image = expected.get("image", {})
                                    expected_hash = (
                                        expected_image.get("pixel_hash", {})
                                        if isinstance(expected_image, dict)
                                        else {}
                                    )
                                    if isinstance(expected_hash, dict) and expected_hash.get(
                                        "value"
                                    ):
                                        expected_pixel_hash = str(expected_hash["value"])
                            if use_cached_file_hash and expected_pixel_hash is not None:
                                embedded = _cached_verify_embedded_metadata(
                                    image_path,
                                    expected_for_verify,
                                    expected_pixel_hash=expected_pixel_hash,
                                    verifier=verify_embedded_metadata,
                                    canonical_payload_prevalidated=expected_prevalidated,
                                )
                            else:
                                embedded = verify_embedded_metadata(
                                    image_path,
                                    expected_for_verify,
                                    expected_pixel_hash=expected_pixel_hash,
                                    canonical_payload_prevalidated=expected_prevalidated,
                                )
                            image_identity = embedded.image
                            if image_identity.image_id != str(frame_id):
                                frame_errors.append(
                                    f"image_metadata: frame {frame_id} image ID disagrees with embedded metadata"
                                )
                            if frame.get("requested_ms") is not None and (
                                image_identity.requested_ms != int(frame["requested_ms"])
                            ):
                                frame_errors.append(
                                    f"image_metadata: frame {frame_id} requested time disagrees with embedded metadata"
                                )
                            if frame.get("actual_ms") is not None and (
                                image_identity.actual_ms != int(frame["actual_ms"])
                            ):
                                frame_errors.append(
                                    f"image_metadata: frame {frame_id} actual time disagrees with embedded metadata"
                                )
                            if frame.get("pts") != image_identity.pts.value:
                                frame_errors.append(
                                    f"image_metadata: frame {frame_id} raw PTS disagrees with embedded metadata"
                                )
                            if frame.get("time_base") != image_identity.pts.time_base:
                                frame_errors.append(
                                    f"image_metadata: frame {frame_id} time base disagrees with embedded metadata"
                                )
                            if (
                                frame.get("parent_full_frame_id")
                                != image_identity.parent_full_frame_id
                            ):
                                frame_errors.append(
                                    f"image_metadata: frame {frame_id} parent disagrees with embedded metadata"
                                )
                            if frame.get("crop_xywh") is not None and list(
                                frame["crop_xywh"]
                            ) != list(image_identity.crop_xywh or ()):
                                frame_errors.append(
                                    f"image_metadata: frame {frame_id} crop geometry disagrees with embedded metadata"
                                )
                            if frame.get("pixel_hash") and frame.get(
                                "pixel_hash"
                            ) != image_identity.pixel_hash.model_dump(mode="json"):
                                frame_errors.append(
                                    f"image_metadata: frame {frame_id} pixel hash disagrees with embedded metadata"
                                )
                            if (
                                frame.get("latest_revision_id")
                                != embedded.analysis.latest_revision_id
                            ):
                                frame_errors.append(
                                    f"image_metadata: frame {frame_id} latest revision disagrees with embedded metadata"
                                )
                            if (
                                frame.get("metadata_payload_digest")
                                != embedded.integrity.payload_digest
                            ):
                                frame_errors.append(
                                    f"image_metadata: frame {frame_id} payload digest disagrees with embedded metadata"
                                )
                            if frame.get("file_hash") and not (
                                use_cached_file_hash and expected_pixel_hash is not None
                            ):
                                actual_file_hash = (
                                    _cached_file_hash(image_path)
                                    if use_cached_file_hash
                                    else sha256_file(image_path)
                                )
                                if str(frame["file_hash"]) != actual_file_hash:
                                    frame_errors.append(
                                        f"image_metadata: frame {frame_id} whole-file hash disagrees with canonical state"
                                    )
                        else:
                            verify_embedded_metadata(image_path)
                    except Exception as exc:  # validation must report each corrupt artifact
                        frame_errors.append(f"image_metadata: {frame_id}: {exc}")
                    return frame_errors

                frame_items = list(known_frames.items())
                metadata_workers = min(_metadata_verify_workers(), max(1, len(frame_items)))
                with ThreadPoolExecutor(
                    max_workers=metadata_workers,
                    thread_name_prefix="vsr-validator-metadata",
                ) as pool:
                    for frame_errors in pool.map(verify_frame_metadata, frame_items):
                        errors.extend(frame_errors)
                # Candidates and diagnostics are hidden from Markdown, but they
                # are still generated evidence and must carry the same portable
                # envelope.  Their canonical mirrors live in the image ledger.
                candidate_payloads: dict[str, dict[str, Any]] = {}
                candidate_file_hashes: dict[str, str] = {}
                ledger_path = root / ".state" / "vision" / "image-observations.json"
                if ledger_path.is_file():
                    try:
                        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                        if isinstance(ledger, dict):
                            candidate_payloads = {
                                str(item.get("image", {}).get("image_id")): item
                                for item in ledger.get("candidate_payloads", [])
                                if isinstance(item, dict)
                            }
                            candidate_file_hashes = {
                                str(item.get("frame_id")): str(item.get("file_hash"))
                                for item in ledger.get("candidate_frames", [])
                                if isinstance(item, dict)
                                and item.get("frame_id")
                                and item.get("file_hash")
                            }
                    except (OSError, json.JSONDecodeError) as exc:
                        errors.append(f"image_metadata: candidate ledger unreadable: {exc}")
                generated_hidden_images = (
                    [
                        path
                        for path in (root / ".state" / "candidates").rglob("*")
                        if path.is_file()
                        and path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}
                    ]
                    if (root / ".state" / "candidates").exists()
                    else []
                )
                known_final_paths = {
                    (root / str(frame.get("full_frame_path") or frame.get("path"))).resolve()
                    for frame in known_frames.values()
                    if frame.get("full_frame_path") or frame.get("path")
                }
                for image_path in generated_hidden_images:
                    if image_path.resolve() in known_final_paths:
                        continue
                    try:
                        # Read only the canonical iTXt envelope first.  A
                        # matching ledger whole-file hash can then prove the
                        # hidden candidate bytes without decoding its pixels;
                        # artifacts without that hash retain the strict path.
                        embedded = read_embedded_metadata(image_path)
                        expected = candidate_payloads.get(embedded.image.image_id)
                        if expected is None:
                            errors.append(
                                f"image_metadata: hidden candidate {image_path.name} lacks a canonical ledger mirror"
                            )
                        else:
                            candidate_expected_pixel_hash: str | None = None
                            candidate_hash = candidate_file_hashes.get(embedded.image.image_id)
                            if use_cached_file_hash and candidate_hash:
                                actual_candidate_hash = _cached_file_hash(image_path)
                                if candidate_hash == actual_candidate_hash:
                                    expected_image = expected.get("image", {})
                                    expected_hash = (
                                        expected_image.get("pixel_hash", {})
                                        if isinstance(expected_image, dict)
                                        else {}
                                    )
                                    if isinstance(expected_hash, dict) and expected_hash.get(
                                        "value"
                                    ):
                                        candidate_expected_pixel_hash = str(expected_hash["value"])
                                else:
                                    errors.append(
                                        f"image_metadata: hidden candidate {image_path.name} whole-file hash disagrees with canonical state"
                                    )
                            if use_cached_file_hash and candidate_expected_pixel_hash is not None:
                                _cached_verify_embedded_metadata(
                                    image_path,
                                    expected,
                                    expected_pixel_hash=candidate_expected_pixel_hash,
                                    verifier=verify_embedded_metadata,
                                )
                            else:
                                verify_embedded_metadata(
                                    image_path,
                                    expected,
                                    expected_pixel_hash=candidate_expected_pixel_hash,
                                )
                    except Exception as exc:
                        errors.append(f"image_metadata: hidden candidate {image_path.name}: {exc}")

    try:
        containment_snapshot.verify_unchanged()
    except SecurityError as exc:
        errors.append(f"path_containment: {exc}")

    checks = {
        "markdown_count": len(markdown_files),
        "html_count": len(html_files),
        "anchors": len(anchor_set),
        "image_links": len(image_links),
        "evidence_images": len(evidence_images),
        "metadata_verified": verify_metadata,
        "metadata_workers": metadata_workers if verify_metadata else 0,
        "metadata_integrity_mode": (
            "canonical-file-hash" if verify_metadata and use_cached_file_hash else "decoded-pixels"
        ),
    }
    result = ValidationResult(not errors, errors, warnings, checks)
    _remember_validation_result(
        root,
        verify_metadata=verify_metadata,
        use_cached_file_hash=use_cached_file_hash,
        result=result,
    )
    return result
