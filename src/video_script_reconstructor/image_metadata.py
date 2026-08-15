"""Secure PNG evidence-metadata carriage and integrity verification.

The canonical UTF-8 JSON envelope is stored in an uncompressed PNG iTXt chunk
named ``video-script-reconstructor``.  A concise Description is generated from
the same validated envelope.  Metadata-only writes are accepted only when the
normalized decoded-pixel hash remains unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import struct
import tempfile
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from pydantic import ValidationError

from .schemas import (
    EmbeddedSufficiency,
    EvidenceImageMetadata,
    ImageAnalysis,
    ImageDerivation,
    ImageIdentity,
    ImageIntegrity,
    ImageKnowledge,
    ImageLinks,
    PixelHash,
    PTSReference,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ITXT_KEYWORD = "video-script-reconstructor"
DESCRIPTION_KEY = "Description"
DEFAULT_MAX_PAYLOAD_BYTES = 1_048_576
DEFAULT_MAX_DEPTH = 24
DEFAULT_MAX_NODES = 50_000
_DIGEST_ALGORITHM = "sha256-canonical-json-with-digest-omitted-v1"
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?:^|[\s\"'])(?:[a-z]:[\\/]|\\\\[^\\\s]+[\\][^\\\s]+)")
_UNIX_ABSOLUTE_PATH = re.compile(
    r"(?:^|[\s\"'])/(?:Users|home|root|etc|var|tmp|private|mnt|media)/[^\s\"']+"
)
_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|rk|pk)-(?:live|test|proj)?[_-]?[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|authorization)\s*[:=]\s*[^\s,;]{8,}"),
    re.compile(r"(?i)https?://[^\s]+[?&](?:x-amz-signature|signature|sig|token|access_token)="),
)
_PRIVATE_REASONING_PATTERNS = (
    re.compile(r"(?i)\bchain[- ]of[- ]thought\b"),
    re.compile(r"(?i)\bhidden reasoning\b"),
    re.compile(r"(?i)\bprivate prompt\b"),
    re.compile(r"(?i)\bsystem prompt\b"),
)


class ImageMetadataError(ValueError):
    """Base error for malformed or inconsistent image metadata."""


class MissingImageMetadataError(ImageMetadataError):
    """The image has no supported embedded evidence envelope."""


class UnsupportedMetadataVersionError(ImageMetadataError):
    """The payload version has no explicit registered migration."""


class MetadataDigestError(ImageMetadataError):
    """The canonical payload digest does not match the envelope."""


class PixelInvariantError(ImageMetadataError):
    """A metadata-only operation changed decoded image pixels."""


class MetadataSecurityError(ImageMetadataError):
    """Portable metadata violates a size, nesting, secret, or trust boundary."""


MetadataMigration = Callable[[dict[str, Any]], dict[str, Any]]
_MIGRATIONS: dict[str, MetadataMigration] = {}


def register_metadata_migration(source_version: str, migration: MetadataMigration) -> None:
    """Register an explicit, testable migration into the current 1.0 schema."""

    if not source_version or source_version == "1.0":
        raise ValueError("source_version must be a non-current version")
    if source_version in _MIGRATIONS:
        raise ValueError(f"migration already registered for schema {source_version}")
    _MIGRATIONS[source_version] = migration


def canonical_json_bytes(value: Any) -> bytes:
    """Return the project's byte-identical canonical JSON representation.

    This is the documented JCS-equivalent subset used by the project: models are
    JSON-mode dumped, object keys are Unicode-codepoint sorted, insignificant
    whitespace is omitted, UTF-8 is emitted directly, and NaN/Infinity are denied.
    Persisted schemas avoid floating-point values requiring JCS exponent rewriting.
    """

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ImageMetadataError(f"payload is not canonical-JSON serializable: {exc}") from exc


def _without_payload_digest(payload: EvidenceImageMetadata | Mapping[str, Any]) -> dict[str, Any]:
    data = (
        payload.model_dump(mode="json")
        if isinstance(payload, EvidenceImageMetadata)
        else copy.deepcopy(dict(payload))
    )
    integrity = data.get("integrity")
    if not isinstance(integrity, dict):
        raise ImageMetadataError("payload.integrity must be an object")
    integrity.pop("payload_digest", None)
    return data


def canonical_payload_digest(payload: EvidenceImageMetadata | Mapping[str, Any]) -> str:
    """Hash the canonical envelope with integrity.payload_digest omitted."""

    return hashlib.sha256(canonical_json_bytes(_without_payload_digest(payload))).hexdigest()


def _walk_strings(value: Any, *, max_depth: int, max_nodes: int) -> list[str]:
    strings: list[str] = []
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise MetadataSecurityError(f"metadata exceeds the {max_nodes}-node limit")
        if depth > max_depth:
            raise MetadataSecurityError(f"metadata nesting exceeds the {max_depth}-level limit")
        if isinstance(current, str):
            strings.append(current)
        elif isinstance(current, Mapping):
            stack.extend((key, depth + 1) for key in current.keys())
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend((item, depth + 1) for item in current)
    return strings


def validate_metadata_security(
    payload: Mapping[str, Any],
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> None:
    """Reject oversized, secret-bearing, path-bearing, or private-reasoning payloads.

    Strings remain untrusted evidence after this check.  No string is interpreted
    as a command, URL to fetch, prompt, or host-agent instruction by this module.
    """

    encoded = canonical_json_bytes(payload)
    if len(encoded) > max_payload_bytes:
        raise MetadataSecurityError(
            f"metadata payload is {len(encoded)} bytes; limit is {max_payload_bytes}"
        )
    for text in _walk_strings(payload, max_depth=max_depth, max_nodes=max_nodes):
        if "\x00" in text:
            raise MetadataSecurityError("metadata strings may not contain NUL bytes")
        if _WINDOWS_ABSOLUTE_PATH.search(text) or _UNIX_ABSOLUTE_PATH.search(text):
            raise MetadataSecurityError(
                "portable image metadata may not contain absolute local paths"
            )
        if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            raise MetadataSecurityError(
                "portable image metadata appears to contain a credential or signed URL"
            )
        if any(pattern.search(text) for pattern in _PRIVATE_REASONING_PATTERNS):
            raise MetadataSecurityError(
                "portable image metadata may not contain private prompts or hidden reasoning"
            )


def _parse_payload(
    value: bytes | str | Mapping[str, Any], *, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
) -> EvidenceImageMetadata:
    if isinstance(value, bytes):
        if len(value) > max_payload_bytes:
            raise MetadataSecurityError("embedded metadata exceeds configured size limit")
        try:
            raw: Any = json.loads(value.decode("utf-8"))
        except RecursionError as exc:
            raise MetadataSecurityError(
                "embedded JSON nesting exceeds the safe parser limit"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImageMetadataError(f"embedded metadata is not valid UTF-8 JSON: {exc}") from exc
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        if len(encoded) > max_payload_bytes:
            raise MetadataSecurityError("embedded metadata exceeds configured size limit")
        try:
            raw = json.loads(value)
        except RecursionError as exc:
            raise MetadataSecurityError(
                "embedded JSON nesting exceeds the safe parser limit"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ImageMetadataError(f"embedded metadata is not valid JSON: {exc}") from exc
    else:
        try:
            raw = copy.deepcopy(dict(value))
        except RecursionError as exc:
            raise MetadataSecurityError(
                "metadata object nesting exceeds the safe parser limit"
            ) from exc
    if not isinstance(raw, dict):
        raise ImageMetadataError("embedded metadata root must be an object")
    version = raw.get("schema_version")
    if version != "1.0":
        migration = _MIGRATIONS.get(str(version))
        if migration is None:
            raise UnsupportedMetadataVersionError(
                f"unsupported image metadata schema version: {version!r}"
            )
        raw = migration(raw)
        if raw.get("schema_version") != "1.0":
            raise ImageMetadataError("metadata migration did not produce schema version 1.0")
    validate_metadata_security(raw, max_payload_bytes=max_payload_bytes)
    try:
        model = EvidenceImageMetadata.model_validate(raw)
    except ValidationError as exc:
        raise ImageMetadataError(f"image metadata schema validation failed: {exc}") from exc
    expected = canonical_payload_digest(model)
    if model.integrity.payload_digest != expected:
        detail = f"embedded={model.integrity.payload_digest} calculated={expected}"
        raise MetadataDigestError(f"payload digest mismatch: {detail}")
    return model


def prepare_metadata_payload(
    payload: EvidenceImageMetadata | Mapping[str, Any],
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> EvidenceImageMetadata:
    """Validate an intended payload and deterministically fill its own digest."""

    raw = (
        payload.model_dump(mode="json")
        if isinstance(payload, EvidenceImageMetadata)
        else copy.deepcopy(dict(payload))
    )
    integrity = raw.get("integrity")
    if not isinstance(integrity, dict):
        raise ImageMetadataError("payload.integrity must be an object")
    integrity["payload_digest_algorithm"] = _DIGEST_ALGORITHM
    integrity["payload_digest"] = "0" * 64
    # Apply bounds and string policy before asking Pydantic to traverse attacker-
    # supplied lists or nested objects.
    validate_metadata_security(raw, max_payload_bytes=max_payload_bytes)
    # Validate all other model constraints before calculating the final digest.
    try:
        provisional = EvidenceImageMetadata.model_validate(raw)
    except ValidationError as exc:
        raise ImageMetadataError(f"image metadata schema validation failed: {exc}") from exc
    raw["integrity"]["payload_digest"] = canonical_payload_digest(provisional)
    validate_metadata_security(raw, max_payload_bytes=max_payload_bytes)
    return _parse_payload(raw, max_payload_bytes=max_payload_bytes)


def _normalized_rgba(image: Image.Image) -> Image.Image:
    oriented = ImageOps.exif_transpose(image)
    icc = image.info.get("icc_profile")
    if icc:
        try:
            alpha = oriented.getchannel("A") if "A" in oriented.getbands() else None
            rgb = oriented.convert("RGB")
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            srgb = ImageCms.createProfile("sRGB")
            converted = ImageCms.profileToProfile(rgb, source_profile, srgb, outputMode="RGB")
            if converted is None:
                raise ImageMetadataError("ICC conversion returned no decoded image")
            rgba = converted.convert("RGBA")
            if alpha is not None:
                rgba.putalpha(alpha)
            return rgba
        except (OSError, ValueError) as exc:
            raise ImageMetadataError(f"invalid or unsupported image color profile: {exc}") from exc
    return oriented.convert("RGBA")


def normalized_pixel_hash(path: str | Path) -> str:
    """Hash oriented RGBA8 pixels normalized to sRGB, including dimensions."""

    try:
        with Image.open(path) as image:
            image.load()
            rgba = _normalized_rgba(image)
            digest = hashlib.sha256()
            digest.update(b"sha256-rgba8-srgb-v1\x00")
            digest.update(struct.pack(">II", rgba.width, rgba.height))
            digest.update(rgba.tobytes())
            return digest.hexdigest()
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageMetadataError(f"cannot decode image pixels from {path}: {exc}") from exc


def create_creation_metadata(
    path: str | Path,
    *,
    image_id: str,
    media_id: str,
    origin: Literal["extracted_full_frame", "derived_crop", "candidate", "diagnostic_overlay"],
    derivation_method: str,
    requested_ms: int,
    actual_ms: int,
    pts_value: int | None,
    time_base: str | None,
    pts_source: str,
    role: str,
    selection_reason: str,
    revision_id: str,
    canonical_revision_locator: str,
    canonical_revision_digest: str,
    parent_full_frame_id: str | None = None,
    crop_xywh: tuple[int, int, int, int] | None = None,
    transformation_ids: list[str] | None = None,
    links: ImageLinks | Mapping[str, Any] | None = None,
    why_it_matters: str | None = None,
    unanswered_questions: list[str] | None = None,
) -> EvidenceImageMetadata:
    """Create the mandatory first envelope for an already emitted PNG.

    ``actual_ms`` is always explicit; this API intentionally has no fallback to the
    requested time.  The caller must pass measured/estimated timing provenance.
    """

    target = Path(path)
    try:
        with Image.open(target) as image:
            image.load()
            width, height = ImageOps.exif_transpose(image).size
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageMetadataError(f"cannot decode creation image {target}: {exc}") from exc
    questions = unanswered_questions or []
    raw = EvidenceImageMetadata(
        image=ImageIdentity(
            image_id=image_id,
            media_id=media_id,
            parent_full_frame_id=parent_full_frame_id,
            origin=origin,
            derivation=ImageDerivation(
                method=derivation_method,
                transformation_ids=transformation_ids or [],
            ),
            requested_ms=requested_ms,
            actual_ms=actual_ms,
            pts=PTSReference(value=pts_value, time_base=time_base, source=pts_source),
            role=role,
            width=width,
            height=height,
            crop_xywh=crop_xywh,
            pixel_hash=PixelHash(value=normalized_pixel_hash(target)),
        ),
        links=links if isinstance(links, ImageLinks) else ImageLinks.model_validate(links or {}),
        knowledge=ImageKnowledge(
            selection_reason=selection_reason,
            why_it_matters=why_it_matters,
            current_factual_description=None,
            explicit_unknowns=questions,
        ),
        analysis=ImageAnalysis(
            enrichment_level="creation",
            semantic_status="unobserved",
            sufficiency=EmbeddedSufficiency(
                status="insufficient" if questions else "semantic_observer_unavailable",
                unanswered_questions=questions,
                recommended_next_action=(
                    "Inspect original-resolution full/crop/adjacent evidence "
                    "for the listed questions."
                    if questions
                    else "Determine whether semantic visual analysis is required by a linked block."
                ),
            ),
            latest_revision_id=revision_id,
            revision_number=1,
        ),
        integrity=ImageIntegrity(
            payload_digest="0" * 64,
            canonical_revision_locator=canonical_revision_locator,
            canonical_revision_digest=canonical_revision_digest,
        ),
    )
    return prepare_metadata_payload(raw)


def _png_itxt_payload(path: Path, *, max_payload_bytes: int) -> bytes:
    """Extract the canonical uncompressed iTXt safely before Pillow decoding."""

    matches: list[bytes] = []
    try:
        with path.open("rb") as handle:
            if handle.read(8) != PNG_SIGNATURE:
                raise ImageMetadataError("evidence metadata is supported only in PNG files")
            while True:
                header = handle.read(8)
                if len(header) == 0:
                    break
                if len(header) != 8:
                    raise ImageMetadataError("truncated PNG chunk header")
                length, chunk_type = struct.unpack(">I4s", header)
                if length > max(max_payload_bytes + 4096, 16 * 1024 * 1024):
                    raise MetadataSecurityError(f"PNG chunk {chunk_type!r} exceeds safe size limit")
                data = handle.read(length)
                crc = handle.read(4)
                if len(data) != length or len(crc) != 4:
                    raise ImageMetadataError("truncated PNG chunk")
                if chunk_type == b"iTXt":
                    keyword, separator, rest = data.partition(b"\x00")
                    if separator and keyword == ITXT_KEYWORD.encode("latin-1"):
                        if len(rest) < 4:
                            raise ImageMetadataError("malformed canonical iTXt chunk")
                        compression_flag, compression_method = rest[0], rest[1]
                        if compression_flag != 0 or compression_method != 0:
                            raise MetadataSecurityError("canonical iTXt must be uncompressed")
                        rest = rest[2:]
                        _language, separator, rest = rest.partition(b"\x00")
                        if not separator:
                            raise ImageMetadataError("malformed canonical iTXt language field")
                        _translated, separator, text = rest.partition(b"\x00")
                        if not separator:
                            raise ImageMetadataError("malformed canonical iTXt translated keyword")
                        if len(text) > max_payload_bytes:
                            raise MetadataSecurityError(
                                "embedded metadata exceeds configured size limit"
                            )
                        matches.append(text)
                if chunk_type == b"IEND":
                    break
    except OSError as exc:
        raise ImageMetadataError(f"unable to read image {path}: {exc}") from exc
    if not matches:
        raise MissingImageMetadataError(f"PNG has no {ITXT_KEYWORD!r} iTXt payload")
    if len(matches) != 1:
        raise ImageMetadataError("PNG contains multiple canonical metadata payloads")
    return matches[0]


def read_embedded_metadata(
    path: str | Path, *, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
) -> EvidenceImageMetadata:
    return _parse_payload(
        _png_itxt_payload(Path(path), max_payload_bytes=max_payload_bytes),
        max_payload_bytes=max_payload_bytes,
    )


read_image_metadata = read_embedded_metadata


def _description(payload: EvidenceImageMetadata, limit: int = 2000) -> str:
    text = payload.knowledge.current_factual_description
    if not text:
        if payload.analysis.semantic_status in {"unobserved", "deterministic_only"}:
            text = "Visual evidence retained; semantic description pending review."
        else:
            text = payload.knowledge.selection_reason
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _color_signature(image: Image.Image) -> tuple[str, str | None, str | None, str]:
    icc = image.info.get("icc_profile")
    exif = image.info.get("exif")
    interpretation = {
        key: image.info.get(key)
        for key in ("gamma", "srgb", "chromaticity", "transparency")
        if key in image.info
    }
    return (
        image.mode,
        hashlib.sha256(icc).hexdigest() if isinstance(icc, bytes) else None,
        hashlib.sha256(exif).hexdigest() if isinstance(exif, bytes) else None,
        hashlib.sha256(repr(interpretation).encode("utf-8")).hexdigest(),
    )


def _normalized_header_size(image: Image.Image) -> tuple[int, int]:
    """Read the orientation-normalized dimensions without inflating IDAT pixels."""

    orientation: int | None = None
    raw_exif = image.info.get("exif")
    if isinstance(raw_exif, bytes):
        exif = Image.Exif()
        try:
            exif.load(raw_exif)
        except (OSError, ValueError) as exc:
            raise ImageMetadataError(f"invalid EXIF orientation data: {exc}") from exc
        orientation = exif.get(274)
    return (image.height, image.width) if orientation in {5, 6, 7, 8} else image.size


def _chunk_bytes(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _png_idat_digest(path: Path) -> str:
    """Hash the encoded pixel stream without decoding the image.

    Metadata-only rewrites in :func:`_rewrite_png_text_chunks` copy every PNG
    chunk except the generated text chunks.  Comparing the IDAT stream, image
    dimensions, and color signature therefore proves that a fast metadata
    update cannot alter the decoded pixels while avoiding a second full RGBA
    decode for every retained frame.
    """

    digest = hashlib.sha256()
    saw_idat = False
    try:
        with path.open("rb") as handle:
            if handle.read(8) != PNG_SIGNATURE:
                raise ImageMetadataError("embedded evidence metadata requires a PNG target")
            while True:
                header = handle.read(8)
                if len(header) != 8:
                    raise ImageMetadataError("truncated PNG while hashing IDAT")
                length, chunk_type = struct.unpack(">I4s", header)
                data = handle.read(length)
                crc = handle.read(4)
                if len(data) != length or len(crc) != 4:
                    raise ImageMetadataError("truncated PNG while hashing IDAT")
                if chunk_type == b"IDAT":
                    saw_idat = True
                    # Include chunk boundaries as well as bytes.  The writer
                    # promises a byte-preserving pixel stream, not merely
                    # equivalent decompressed pixels.
                    digest.update(struct.pack(">I", length))
                    digest.update(data)
                if chunk_type == b"IEND":
                    break
    except OSError as exc:
        raise ImageMetadataError(f"unable to read PNG IDAT stream from {path}: {exc}") from exc
    if not saw_idat:
        raise ImageMetadataError("PNG has no IDAT pixel stream")
    return digest.hexdigest()


def _text_chunk_keyword(chunk_type: bytes, data: bytes) -> bytes | None:
    if chunk_type not in {b"tEXt", b"zTXt", b"iTXt"}:
        return None
    keyword, separator, _rest = data.partition(b"\x00")
    return keyword if separator else None


def _rewrite_png_text_chunks(
    source: Path, destination: Path, *, canonical_payload: bytes, description: str
) -> tuple[str, str]:
    """Copy a PNG verbatim except for generated text and return file/IDAT hashes.

    The IDAT digest is accumulated from the same byte stream that is written to
    the temporary file.  This keeps the existing byte-preserving proof while
    avoiding a second full read of the temporary PNG after the rewrite.
    """

    canonical_data = ITXT_KEYWORD.encode("latin-1") + b"\x00\x00\x00\x00\x00" + canonical_payload
    description_data = (
        DESCRIPTION_KEY.encode("latin-1") + b"\x00\x00\x00\x00\x00" + description.encode("utf-8")
    )
    inserted = False
    saw_idat = False
    file_digest = hashlib.sha256()
    idat_digest = hashlib.sha256()

    def write_bytes(writer: Any, value: bytes) -> None:
        writer.write(value)
        file_digest.update(value)

    with source.open("rb") as reader, destination.open("wb") as writer:
        signature = reader.read(8)
        if signature != PNG_SIGNATURE:
            raise ImageMetadataError("embedded evidence metadata requires a PNG target")
        write_bytes(writer, signature)
        while True:
            header = reader.read(8)
            if len(header) != 8:
                raise ImageMetadataError("truncated PNG while writing metadata")
            length, chunk_type = struct.unpack(">I4s", header)
            data = reader.read(length)
            crc = reader.read(4)
            if len(data) != length or len(crc) != 4:
                raise ImageMetadataError("truncated PNG while writing metadata")
            keyword = _text_chunk_keyword(chunk_type, data)
            if keyword in {ITXT_KEYWORD.encode("latin-1"), DESCRIPTION_KEY.encode("latin-1")}:
                continue
            if chunk_type in {b"IDAT", b"IEND"} and not inserted:
                write_bytes(writer, _chunk_bytes(b"iTXt", canonical_data))
                write_bytes(writer, _chunk_bytes(b"iTXt", description_data))
                inserted = True
            if chunk_type == b"IDAT":
                saw_idat = True
                idat_digest.update(struct.pack(">I", length))
                idat_digest.update(data)
            write_bytes(writer, header)
            write_bytes(writer, data)
            write_bytes(writer, crc)
            if chunk_type == b"IEND":
                break
    if not inserted:
        raise ImageMetadataError("PNG has no IEND chunk")
    if not saw_idat:
        raise ImageMetadataError("PNG has no IDAT pixel stream")
    return file_digest.hexdigest(), idat_digest.hexdigest()


def _embed_metadata_internal(
    path: str | Path,
    payload: EvidenceImageMetadata | Mapping[str, Any],
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    verify_source_pixels: bool = True,
    verify_decoded_pixels: bool | None = None,
) -> tuple[EvidenceImageMetadata, str]:
    """Atomically embed metadata while proving decoded-pixel invariance.

    ``verify_source_pixels`` defaults to ``True`` for the public safety
    contract.  Deterministic pipeline enrichment may set it to ``False`` only
    after an earlier creation write has established the canonical normalized
    pixel hash.  ``verify_decoded_pixels`` controls whether Pillow decodes the
    source and rewritten PNGs in addition to the hash/IDAT checks.  It defaults
    to the value of ``verify_source_pixels`` so existing callers retain their
    safety behavior.  An internal caller may set both flags to ``False`` only
    after it has independently verified the stored pixel hash; the fast path
    still checks the PNG IDAT stream, dimensions, color signature, and canonical
    metadata read-back, so a metadata-only rewrite cannot change the encoded
    image data.
    """

    target = Path(path)
    if verify_decoded_pixels is None:
        verify_decoded_pixels = verify_source_pixels
    prepared = prepare_metadata_payload(payload, max_payload_bytes=max_payload_bytes)
    before_hash = (
        normalized_pixel_hash(target)
        if verify_source_pixels
        else prepared.image.pixel_hash.value
    )
    if prepared.image.pixel_hash.value != before_hash:
        detail = f"payload={prepared.image.pixel_hash.value} decoded={before_hash}"
        raise PixelInvariantError(f"payload pixel hash does not match decoded pixels: {detail}")
    before_idat_digest = _png_idat_digest(target)
    try:
        with Image.open(target) as image:
            if image.format != "PNG":
                raise ImageMetadataError("embedded evidence metadata requires a PNG target")
            if verify_decoded_pixels:
                image.load()
            before_size = image.size
            before_color = _color_signature(image)
            if (prepared.image.width, prepared.image.height) != _normalized_header_size(image):
                raise PixelInvariantError(
                    "payload dimensions disagree with normalized decoded image dimensions"
                )
            serialized = canonical_json_bytes(prepared)
            if len(serialized) > max_payload_bytes:
                raise MetadataSecurityError("canonical metadata exceeds configured size limit")
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            file_hash, after_idat_digest = _rewrite_png_text_chunks(
                target,
                temporary_path,
                canonical_payload=serialized,
                description=_description(prepared),
            )
    except ImageMetadataError:
        if "temporary_path" in locals() and temporary_path.exists():
            temporary_path.unlink()
        raise
    except (OSError, UnidentifiedImageError) as exc:
        if "temporary_path" in locals() and temporary_path.exists():
            temporary_path.unlink()
        raise ImageMetadataError(f"unable to write PNG metadata for {target}: {exc}") from exc

    try:
        after_hash = (
            normalized_pixel_hash(temporary_path) if verify_source_pixels else before_hash
        )
        with Image.open(temporary_path) as candidate:
            if verify_decoded_pixels:
                candidate.load()
            after_size = candidate.size
            after_color = _color_signature(candidate)
        if (
            after_hash != before_hash
            or after_idat_digest != before_idat_digest
            or after_size != before_size
            or after_color != before_color
        ):
            raise PixelInvariantError(
                "metadata-only rewrite changed pixels, dimensions, mode, ICC profile, or EXIF"
            )
        read_back = read_embedded_metadata(temporary_path, max_payload_bytes=max_payload_bytes)
        if canonical_json_bytes(read_back) != canonical_json_bytes(prepared):
            raise MetadataDigestError(
                "metadata read-back differs from the intended canonical payload"
            )
        mode = target.stat().st_mode
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return prepared, file_hash


def embed_metadata(
    path: str | Path,
    payload: EvidenceImageMetadata | Mapping[str, Any],
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    verify_source_pixels: bool = True,
    verify_decoded_pixels: bool | None = None,
) -> EvidenceImageMetadata:
    """Atomically embed metadata while proving decoded-pixel invariance."""

    prepared, _file_hash = _embed_metadata_internal(
        path,
        payload,
        max_payload_bytes=max_payload_bytes,
        verify_source_pixels=verify_source_pixels,
        verify_decoded_pixels=verify_decoded_pixels,
    )
    return prepared


def embed_metadata_with_file_hash(
    path: str | Path,
    payload: EvidenceImageMetadata | Mapping[str, Any],
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    verify_source_pixels: bool = True,
    verify_decoded_pixels: bool | None = None,
) -> tuple[EvidenceImageMetadata, str]:
    """Embed metadata and return the exact post-write file SHA-256.

    The digest is accumulated while the atomic PNG envelope is copied, so
    deterministic pipeline callers do not need a second full read of every
    generated image merely to populate the canonical ``file_hash`` field.
    """

    return _embed_metadata_internal(
        path,
        payload,
        max_payload_bytes=max_payload_bytes,
        verify_source_pixels=verify_source_pixels,
        verify_decoded_pixels=verify_decoded_pixels,
    )


write_image_metadata = embed_metadata


def verify_embedded_metadata(
    path: str | Path,
    canonical_payload: EvidenceImageMetadata | Mapping[str, Any] | None = None,
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    expected_pixel_hash: str | None = None,
    canonical_payload_prevalidated: bool = False,
) -> EvidenceImageMetadata:
    """Validate schema, digest, pixel identity, and optional mirror.

    ``expected_pixel_hash`` is an internal fast-path escape hatch for callers
    that already decoded and verified the image immediately before a
    byte-preserving metadata rewrite.  The default still decodes the PNG so
    public verification remains fully independent.  The
    ``canonical_payload_prevalidated`` flag is reserved for callers that have
    already validated an immutable canonical project model; it still checks
    that model's payload digest before comparing the embedded bytes.
    """

    embedded = read_embedded_metadata(path, max_payload_bytes=max_payload_bytes)
    calculated_pixel_hash = (
        normalized_pixel_hash(path) if expected_pixel_hash is None else expected_pixel_hash
    )
    if embedded.image.pixel_hash.value != calculated_pixel_hash:
        detail = f"embedded={embedded.image.pixel_hash.value} calculated={calculated_pixel_hash}"
        raise PixelInvariantError(f"image {embedded.image.image_id} pixel hash mismatch: {detail}")
    if expected_pixel_hash is None:
        with Image.open(path) as image:
            normalized_size = ImageOps.exif_transpose(image).size
            description = image.info.get(DESCRIPTION_KEY)
    else:
        # The internal validator reaches this branch only after an exact
        # whole-file digest and canonical pixel hash have already matched.
        # Pillow can read dimensions, EXIF orientation, and text chunks from
        # headers without inflating IDAT; avoid decoding the same pixels again.
        with Image.open(path) as image:
            normalized_size = _normalized_header_size(image)
            description = image.info.get(DESCRIPTION_KEY)
    if normalized_size != (embedded.image.width, embedded.image.height):
        raise PixelInvariantError("embedded image dimensions disagree with decoded dimensions")
    if description != _description(embedded):
        raise ImageMetadataError("human-readable Description does not match the canonical payload")
    if canonical_payload is not None:
        if canonical_payload_prevalidated:
            if not isinstance(canonical_payload, EvidenceImageMetadata):
                raise ImageMetadataError(
                    "canonical_payload_prevalidated requires an EvidenceImageMetadata model"
                )
            canonical = canonical_payload
            validate_metadata_security(
                canonical.model_dump(mode="json"),
                max_payload_bytes=max_payload_bytes,
            )
            expected_digest = canonical_payload_digest(canonical)
            if canonical.integrity.payload_digest != expected_digest:
                raise MetadataDigestError(
                    "prevalidated canonical payload digest does not match its content; "
                    "canonical-state mirror is stale"
                )
        else:
            canonical = prepare_metadata_payload(
                canonical_payload,
                max_payload_bytes=max_payload_bytes,
            )
        if canonical_json_bytes(canonical) != canonical_json_bytes(embedded):
            raise MetadataDigestError("embedded payload disagrees with the canonical-state mirror")
    return embedded


validate_image_metadata = verify_embedded_metadata


__all__ = [
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "DESCRIPTION_KEY",
    "ITXT_KEYWORD",
    "ImageMetadataError",
    "MetadataDigestError",
    "MetadataSecurityError",
    "MissingImageMetadataError",
    "PixelInvariantError",
    "UnsupportedMetadataVersionError",
    "canonical_json_bytes",
    "canonical_payload_digest",
    "create_creation_metadata",
    "embed_metadata",
    "embed_metadata_with_file_hash",
    "normalized_pixel_hash",
    "prepare_metadata_payload",
    "read_embedded_metadata",
    "read_image_metadata",
    "register_metadata_migration",
    "validate_image_metadata",
    "validate_metadata_security",
    "verify_embedded_metadata",
    "write_image_metadata",
]
