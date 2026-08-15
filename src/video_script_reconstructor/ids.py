"""Deterministic, user-visible identifiers and evidence filenames."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

ID_WIDTHS: dict[str, int] = {
    "media": 0,
    "transcript": 6,
    "word": 6,
    "visual_event": 6,
    "frame": 6,
    "ocr": 6,
    "visual_analysis": 6,
    "image_claim": 6,
    "metadata_revision": 6,
    "question": 6,
    "metadata_sufficiency": 6,
    "block": 6,
    "chapter": 3,
    "review": 6,
    "transformation": 6,
    "candidate": 6,
    "repair": 6,
    "run": 6,
}
ID_PREFIXES: dict[str, str] = {
    "media": "M",
    "transcript": "T",
    "word": "W",
    "visual_event": "V",
    "frame": "F",
    "ocr": "O",
    "visual_analysis": "VA",
    "image_claim": "IC",
    "metadata_revision": "MR",
    "question": "Q",
    "metadata_sufficiency": "MS",
    "block": "B",
    "chapter": "C",
    "review": "R",
    "transformation": "X",
    "candidate": "TC",
    "repair": "RP",
    "run": "RUN",
}
PREFIX_TO_NAMESPACE = {prefix: namespace for namespace, prefix in ID_PREFIXES.items()}
_VISIBLE_ID = re.compile(
    r"^(?P<prefix>RUN|VA|IC|MR|MS|TC|RP|[MTWVFOQBCRX])(?P<number>[0-9]+|[0-9A-F]{16})$"
)


class IdentifierError(ValueError):
    """Raised for an invalid namespace, sequence number, or evidence ID."""


def sequential_id(namespace: str, number: int) -> str:
    """Return the contract ID for a one-based sequence number."""

    if namespace not in ID_PREFIXES:
        raise IdentifierError(f"unknown ID namespace: {namespace}")
    if namespace == "media":
        raise IdentifierError("media IDs are content-derived; use media_id()")
    if number < 1:
        raise IdentifierError("sequence numbers are one-based")
    width = ID_WIDTHS[namespace]
    return f"{ID_PREFIXES[namespace]}{number:0{width}d}"


def media_id(content: bytes | str) -> str:
    """Build a portable media ID from a content digest or byte sequence."""

    if isinstance(content, bytes):
        digest = hashlib.sha256(content).hexdigest()
    else:
        candidate = content.lower()
        digest = (
            candidate
            if re.fullmatch(r"[0-9a-f]{64}", candidate)
            else hashlib.sha256(content.encode()).hexdigest()
        )
    return "M" + digest[:16].upper()


def deterministic_id(namespace: str, *parts: object, length: int | None = None) -> str:
    """Return a content-derived ID for records whose order is not authoritative.

    The input framing is unambiguous (length followed by UTF-8 bytes), so tuples
    such as ``("ab", "c")`` and ``("a", "bc")`` cannot collide by concatenation.
    """

    if namespace not in ID_PREFIXES:
        raise IdentifierError(f"unknown ID namespace: {namespace}")
    if namespace == "media":
        return media_id("\x1f".join(str(part) for part in parts))
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    width = length or ID_WIDTHS[namespace]
    # Digit-only IDs remain compatible with the visible contract.  The modulo is
    # deterministic; callers needing collision-free ordered records use allocator.
    value = int.from_bytes(digest.digest()[:8], "big") % (10**width - 1) + 1
    return sequential_id(namespace, value)


stable_id = deterministic_id


def validate_id(identifier: str, namespace: str | None = None) -> bool:
    match = _VISIBLE_ID.fullmatch(identifier)
    if match is None:
        return False
    actual_namespace = PREFIX_TO_NAMESPACE.get(match.group("prefix"))
    if actual_namespace is None or (namespace is not None and actual_namespace != namespace):
        return False
    number = match.group("number")
    if actual_namespace == "media":
        return len(number) == 16
    return len(number) == ID_WIDTHS[actual_namespace] and int(number) > 0


def parse_id(identifier: str) -> tuple[str, int | str]:
    if not validate_id(identifier):
        raise IdentifierError(f"invalid evidence ID: {identifier!r}")
    match = _VISIBLE_ID.fullmatch(identifier)
    assert match is not None
    namespace = PREFIX_TO_NAMESPACE[match.group("prefix")]
    number = match.group("number")
    return namespace, number if namespace == "media" else int(number)


@dataclass
class StableIdAllocator:
    """Append-only key-to-ID allocator suitable for persistence in canonical state."""

    namespace: str
    assignments: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.namespace not in ID_PREFIXES or self.namespace == "media":
            raise IdentifierError(f"namespace cannot be allocated sequentially: {self.namespace}")
        if len(set(self.assignments.values())) != len(self.assignments):
            raise IdentifierError("allocator assignments contain duplicate IDs")
        if not all(validate_id(value, self.namespace) for value in self.assignments.values()):
            raise IdentifierError("allocator assignments contain an invalid ID")

    def get(self, stable_key: str) -> str:
        if stable_key in self.assignments:
            return self.assignments[stable_key]
        used = [int(parse_id(identifier)[1]) for identifier in self.assignments.values()]
        identifier = sequential_id(self.namespace, max(used, default=0) + 1)
        self.assignments[stable_key] = identifier
        return identifier

    def assign_all(self, stable_keys: Iterable[str]) -> list[str]:
        return [self.get(key) for key in stable_keys]


def timestamp_slug(actual_ms: int) -> str:
    if actual_ms < 0:
        raise ValueError("actual_ms must not be negative")
    hours, remainder = divmod(actual_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s{milliseconds:03d}"


def evidence_filename(image_id: str, actual_ms: int, *, detail: bool | None = None) -> str:
    namespace, _ = parse_id(image_id.split("-C", 1)[0])
    if namespace != "frame":
        raise IdentifierError("evidence filenames require a frame or crop ID")
    is_crop = "-C" in image_id
    if is_crop and not re.fullmatch(r"F[0-9]{6}-C[0-9]{2}", image_id):
        raise IdentifierError(f"invalid crop ID: {image_id}")
    suffix = "detail" if (is_crop if detail is None else detail) else "full"
    return f"{image_id}__{timestamp_slug(actual_ms)}__{suffix}.png"


def crop_id(frame_id: str, number: int) -> str:
    if not validate_id(frame_id, "frame") or number < 1 or number > 99:
        raise IdentifierError("crop IDs require a valid frame ID and number from 1 to 99")
    return f"{frame_id}-C{number:02d}"


__all__ = [
    "ID_PREFIXES",
    "ID_WIDTHS",
    "IdentifierError",
    "StableIdAllocator",
    "crop_id",
    "deterministic_id",
    "evidence_filename",
    "media_id",
    "parse_id",
    "sequential_id",
    "stable_id",
    "timestamp_slug",
    "validate_id",
]
