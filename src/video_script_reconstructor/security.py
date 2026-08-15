from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse, urlunparse

from .errors import SecurityError

_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_SECRET_KEY = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|auth(?:entication|orization)?[_-]?token|"
    r"bearer[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|"
    r"password|authorization|credential)"
)
_SIGNED_QUERY = re.compile(r"(?i)(signature|sig|token|key|credential|expires|x-amz-[^=]+)")
_SIGNED_URL = re.compile(
    r"(?i)https?://[^\s\"']+[?&](?:signature|sig|token|key|credential|expires|x-amz-[^=]+)="
)
_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "access_token",
    "accesstoken",
    "auth_token",
    "authtoken",
    "authentication_token",
    "authenticationtoken",
    "authorization_token",
    "authorizationtoken",
    "bearer_token",
    "bearertoken",
    "refresh_token",
    "refreshtoken",
    "id_token",
    "idtoken",
    "client_secret",
    "clientsecret",
    "password",
    "authorization",
    "credential",
)
_REDACTED_FILE_CACHE_LIMIT = 4096
_REDACTED_FILE_CACHE: dict[Path, tuple[int, int, int, int]] = {}
_WINDOWS_REPARSE_POINT = 0x0400
_ComponentSignature = tuple[int, int, int, int, int, int, int] | None


@dataclass
class JsonPatchState:
    """Process-local offsets for repeated canonical JSON patches.

    The state is only an optimization for one sequential worker.  Every patch
    still checks the stat-bound redaction receipt; a restart, external edit, or
    malformed state therefore takes the existing complete-write fallback.
    """

    text: str = ""
    signature: tuple[int, int, int, int] | None = None
    compact: bool = False
    root_spans: dict[str, tuple[int, int]] = field(default_factory=dict)
    array_items: dict[str, list[tuple[int, int, Any]]] = field(default_factory=dict)

    def clear(self) -> None:
        self.text = ""
        self.signature = None
        self.compact = False
        self.root_spans.clear()
        self.array_items.clear()

    def refresh(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        self.text = text
        self.compact = "\n" not in text.rstrip("\n")
        self.root_spans = _top_level_json_value_spans(text)
        self.array_items.clear()
        self.signature = _file_signature(path)


def _component_state(path: Path) -> tuple[_ComponentSignature, bool]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None, False
    except OSError as exc:
        raise SecurityError(f"Unable to inspect project path safely: {path}") from exc
    signature = (
        info.st_mode,
        info.st_ino,
        info.st_dev,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        int(getattr(info, "st_file_attributes", 0)),
    )
    is_reparse = stat.S_ISLNK(info.st_mode) or bool(
        signature[-1] & _WINDOWS_REPARSE_POINT
    )
    return signature, is_reparse


@dataclass
class ContainmentSnapshot:
    """Memoize component checks and detect mutations at validation end."""

    states: dict[Path, tuple[_ComponentSignature, bool]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def check(self, path: Path) -> bool:
        with self._lock:
            if path in self.states:
                return self.states[path][1]
        state, is_reparse = _component_state(path)
        with self._lock:
            if path not in self.states:
                self.states[path] = (state, is_reparse)
                return is_reparse
            previous_state, previous_reparse = self.states[path]
        if previous_state != state:
            raise SecurityError(f"Project path changed during validation: {path}")
        return previous_reparse

    def verify_unchanged(self) -> None:
        with self._lock:
            items = tuple(self.states.items())
        for path, (expected_state, _is_reparse) in items:
            current_state, _ = _component_state(path)
            if current_state != expected_state:
                raise SecurityError(f"Project path changed during validation: {path}")


def _contains_secret_key(text: str) -> bool:
    """Check key-like secret markers with linear C-level substring scans."""

    normalized = text.casefold().replace("-", "_")
    return any(marker in normalized for marker in _SECRET_KEY_MARKERS)


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def _remember_redacted_file(path: Path) -> None:
    resolved = path.resolve()
    if len(_REDACTED_FILE_CACHE) >= _REDACTED_FILE_CACHE_LIMIT:
        _REDACTED_FILE_CACHE.pop(next(iter(_REDACTED_FILE_CACHE)))
    _REDACTED_FILE_CACHE[resolved] = _file_signature(resolved)


def _is_known_redacted_file(path: Path) -> bool:
    resolved = path.resolve()
    try:
        return _REDACTED_FILE_CACHE.get(resolved) == _file_signature(resolved)
    except OSError:
        return False


def safe_slug(value: str, fallback: str = "media") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(" .-_")
    normalized = re.sub(r"-+", "-", normalized)[:100] or fallback
    if normalized.upper().split(".")[0] in _WINDOWS_RESERVED:
        normalized = f"_{normalized}"
    return normalized


def _is_reparse_component(path: Path) -> bool:
    """Detect symlinks and Windows reparse points without following them."""

    _state, is_reparse = _component_state(path)
    return is_reparse


def _lexically_contained(
    root_resolved: Path,
    candidate: Path,
    *,
    allow_missing: bool,
    containment_snapshot: ContainmentSnapshot | None = None,
) -> Path:
    """Contain a path using lstat-only component checks on a fixed root."""

    resolved = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise SecurityError(f"Path escapes project root: {candidate}") from exc
    current = root_resolved
    component_checker = (
        containment_snapshot.check
        if containment_snapshot is not None
        else _is_reparse_component
    )
    for part in relative.parts:
        current = current / part
        if component_checker(current):
            raise SecurityError(f"Symlink or reparse point is not allowed inside project output: {current}")
    if not allow_missing:
        resolved.stat()
    return resolved


def ensure_contained(
    root: Path,
    candidate: Path,
    *,
    allow_missing: bool = True,
    root_resolved: Path | None = None,
    containment_snapshot: ContainmentSnapshot | None = None,
) -> Path:
    """Resolve a candidate while preserving project-root and symlink guards.

    Callers validating many artifacts from one immutable project snapshot may
    pass the already strict-resolved root to avoid resolving the same root for
    every artifact. Candidate resolution and per-component symlink checks still
    run for every call.
    """

    if root_resolved is not None:
        return _lexically_contained(
            root_resolved,
            candidate,
            allow_missing=allow_missing,
            containment_snapshot=containment_snapshot,
        )
    root_resolved = root.resolve(strict=True)
    resolved = candidate.resolve(strict=not allow_missing)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise SecurityError(f"Path escapes project root: {candidate}") from exc
    current = root_resolved
    for part in resolved.relative_to(root_resolved).parts:
        current = current / part
        # is_symlink() uses lstat and safely returns False for a missing path;
        # the preceding resolve already established the candidate target.
        if current.is_symlink():
            raise SecurityError(f"Symlink is not allowed inside project output: {current}")
    return resolved


def safe_relative_path(
    project_root: Path,
    relative: str,
    *,
    root_resolved: Path | None = None,
    containment_snapshot: ContainmentSnapshot | None = None,
) -> Path:
    if "\\" in relative:
        raise SecurityError("Portable artifact paths must use forward slashes")
    path = Path(relative)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise SecurityError(f"Unsafe relative path: {relative}")
    return ensure_contained(
        project_root,
        project_root.joinpath(*path.parts),
        root_resolved=root_resolved,
        containment_snapshot=containment_snapshot,
    )


def validate_remote_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SecurityError("Only explicit HTTP(S) media URLs are supported")
    host = parsed.hostname.rstrip(".")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if host.lower() in {"localhost", "localhost.localdomain"}:
            raise SecurityError("Loopback targets are forbidden") from None
    else:
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise SecurityError(
                "Private, loopback, link-local, reserved, or multicast targets are forbidden"
            )
    return value


def redact(value: Any) -> Any:
    """Return a secret-redacted, read-only-by-convention structure.

    Unchanged containers may be shared with ``value`` to avoid rebuilding large
    canonical state trees during serialization. Any branch containing a secret
    key, signed URL, or changed descendant is copied before replacement.
    """

    if isinstance(value, dict):
        sanitized: dict[str, Any] | None = None
        pending_pairs: list[tuple[str, Any]] = []
        for raw_key, item in value.items():
            key = str(raw_key)
            replacement = "[REDACTED]" if _SECRET_KEY.search(key) else redact(item)
            if sanitized is None:
                if key != raw_key or replacement is not item:
                    sanitized = dict(pending_pairs)
                else:
                    pending_pairs.append((key, replacement))
                    continue
            sanitized[key] = replacement
        # Most canonical-project subtrees contain no secret-bearing keys. Keep
        # those immutable-by-convention subtrees shared with the input instead
        # of allocating a second full tree solely for serialization.
        return value if sanitized is None else sanitized
    if isinstance(value, list):
        sanitized_list: list[Any] | None = None
        pending_values: list[Any] = []
        for item in value:
            replacement = redact(item)
            if sanitized_list is None and replacement is not item:
                sanitized_list = list(pending_values)
            elif sanitized_list is None:
                pending_values.append(replacement)
            if sanitized_list is not None:
                sanitized_list.append(replacement)
        return value if sanitized_list is None else sanitized_list
    if isinstance(value, tuple):
        sanitized_tuple = tuple(redact(item) for item in value)
        return value if all(new is old for new, old in zip(sanitized_tuple, value, strict=True)) else sanitized_tuple
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        if parsed.query and _SIGNED_QUERY.search(parsed.query):
            return urlunparse(parsed._replace(query="[REDACTED]"))
    return value


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    # Python 3.11+ exposes a C-level file_digest path that avoids a Python
    # read/update loop for the common default chunk size. Keep the explicit
    # loop for Python 3.10 and callers that intentionally choose a custom
    # chunk size; both paths return the same cryptographic digest.
    file_digest = getattr(hashlib, "file_digest", None)
    if file_digest is not None and chunk_size == 1024 * 1024:
        with path.open("rb") as stream:
            return str(file_digest(stream, "sha256").hexdigest())
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    encoded = json.dumps(
        redact(payload),
        ensure_ascii=False,
        indent=None if compact else 2,
        separators=(",", ":") if compact else None,
        sort_keys=True,
    ) + "\n"
    atomic_write_text(path, encoded)
    _remember_redacted_file(path)


def canonical_compact_for_payload(path: Path, payload: Any) -> bool:
    """Choose compact canonical JSON only when it helps long-form state.

    Existing canonical files keep their established layout, so a small
    human-inspectable project is not made slower or harder to read merely by a
    later review/metadata write. New projects switch to compact JSON once their
    evidence/timeline cardinality makes write amplification material.
    """

    if path.is_file():
        try:
            with path.open("rb") as stream:
                return b"\n" not in stream.read(8192).rstrip(b"\n")
        except OSError:
            pass
    if isinstance(payload, Mapping):
        frames = payload.get("frames")
        timeline = payload.get("timeline")
        frame_count = len(frames) if isinstance(frames, list) else 0
        timeline_count = len(timeline) if isinstance(timeline, list) else 0
        return frame_count >= 32 or timeline_count >= 256
    return False


def _top_level_json_value_spans(text: str) -> dict[str, tuple[int, int]]:
    """Locate existing root-object value spans without rebuilding the payload.

    The scanner delegates nested syntax and string escaping to the standard
    library decoder, so commas/braces inside strings or nested arrays cannot
    confuse the boundaries.  It is intentionally strict: callers fall back to
    a complete atomic write when the existing file is not a canonical object.
    """

    decoder = json.JSONDecoder()
    length = len(text)

    def skip_whitespace(position: int) -> int:
        while position < length and text[position] in " \t\r\n":
            position += 1
        return position

    position = skip_whitespace(0)
    if position >= length or text[position] != "{":
        raise ValueError("JSON patch target must be an object")
    position += 1
    spans: dict[str, tuple[int, int]] = {}
    while True:
        position = skip_whitespace(position)
        if position >= length:
            raise ValueError("JSON patch target is truncated")
        if text[position] == "}":
            if text[skip_whitespace(position + 1) :] != "":
                raise ValueError("JSON patch target has trailing data")
            return spans
        key, key_end = decoder.raw_decode(text, position)
        if not isinstance(key, str) or key in spans:
            raise ValueError("JSON patch target has invalid or duplicate keys")
        position = skip_whitespace(key_end)
        if position >= length or text[position] != ":":
            raise ValueError("JSON patch target is missing a key separator")
        value_start = skip_whitespace(position + 1)
        _value, value_end = decoder.raw_decode(text, value_start)
        spans[key] = (value_start, value_end)
        position = skip_whitespace(value_end)
        if position >= length:
            raise ValueError("JSON patch target is truncated")
        if text[position] == ",":
            position += 1
            continue
        if text[position] == "}":
            if text[skip_whitespace(position + 1) :] != "":
                raise ValueError("JSON patch target has trailing data")
            return spans
        raise ValueError("JSON patch target is missing an item separator")


def _array_item_spans(text: str, start: int, end: int) -> list[tuple[int, int, Any]]:
    """Return raw spans and decoded values for one top-level JSON array."""

    decoder = json.JSONDecoder()

    def skip_whitespace(position: int) -> int:
        while position < end and text[position] in " \t\r\n":
            position += 1
        return position

    position = skip_whitespace(start)
    if position >= end or text[position] != "[":
        raise ValueError("JSON array patch target is not an array")
    position += 1
    items: list[tuple[int, int, Any]] = []
    while True:
        position = skip_whitespace(position)
        if position >= end:
            raise ValueError("JSON array patch target is truncated")
        if text[position] == "]":
            return items
        item_start = position
        item, item_end = decoder.raw_decode(text, position)
        items.append((item_start, item_end, item))
        position = skip_whitespace(item_end)
        if position >= end:
            raise ValueError("JSON array patch target is truncated")
        if text[position] == ",":
            position += 1
            continue
        if text[position] == "]":
            return items
        raise ValueError("JSON array patch target is missing an item separator")


def _dotted_value(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _indent_json_fragment(original: str, start: int, encoded: str) -> str:
    line_start = original.rfind("\n", 0, start) + 1
    first_nonspace = line_start
    while first_nonspace < start and original[first_nonspace] in " \t":
        first_nonspace += 1
    continuation_indent = original[line_start:first_nonspace] + "  "
    encoded_lines = encoded.splitlines()
    if len(encoded_lines) <= 1:
        return encoded
    return "\n".join(
        [encoded_lines[0], *(continuation_indent + line for line in encoded_lines[1:])]
    )


def _shift_position(position: int, replacements: list[tuple[int, int, str]]) -> int:
    return position + sum(
        len(replacement) - (end - start)
        for start, end, replacement in replacements
        if end <= position
    )


def _shift_root_spans(
    spans: Mapping[str, tuple[int, int]], replacements: list[tuple[int, int, str]]
) -> dict[str, tuple[int, int]]:
    return {
        key: (_shift_position(start, replacements), _shift_position(end, replacements))
        for key, (start, end) in spans.items()
    }


def _shift_array_items(
    items: list[tuple[int, int, Any]],
    replacements: list[tuple[int, int, str]],
    replacement_values: Mapping[tuple[int, int], tuple[str, Any]],
) -> list[tuple[int, int, Any]]:
    shifted: list[tuple[int, int, Any]] = []
    for start, end, value in items:
        replacement = replacement_values.get((start, end))
        if replacement is not None:
            replacement_text, replacement_value = replacement
            new_start = _shift_position(start, replacements)
            shifted.append((new_start, new_start + len(replacement_text), replacement_value))
            continue
        shifted.append(
            (
                _shift_position(start, replacements),
                _shift_position(end, replacements),
                value,
            )
        )
    return shifted


def atomic_update_json_fields(
    path: Path,
    updates: Mapping[str, Any],
    *,
    fallback_payload: Any | None = None,
    array_item_updates: Mapping[str, tuple[str, Mapping[str, Any]]] | None = None,
    patch_state: JsonPatchState | None = None,
) -> None:
    """Atomically replace existing root fields while preserving large siblings.

    The existing canonical file is copied byte-for-byte except for the named
    fields, whose replacement values are redacted and deterministically encoded.
    ``array_item_updates`` can replace selected objects inside root arrays while
    preserving their unchanged siblings. This avoids serializing large evidence
    arrays when only one frame/event/revision changed. A malformed/missing target
    falls back to ``atomic_write_json`` when ``fallback_payload`` is supplied.
    ``patch_state`` reuses validated offsets across one sequential batch; it is
    invalidated automatically when the file's stat-bound receipt changes.
    """

    try:
        state_is_current = False
        compact_target = False
        if patch_state is not None and patch_state.text:
            if patch_state.signature != _file_signature(path):
                raise ValueError("JSON patch state is stale")
            original = patch_state.text
            state_is_current = True
            compact_target = patch_state.compact
        else:
            original = path.read_text(encoding="utf-8")
            compact_target = "\n" not in original.rstrip("\n")
        if not _is_known_redacted_file(path):
            if fallback_payload is not None:
                # A state file created by another process may have been edited;
                # one complete redacted write re-establishes the trusted receipt
                # before later incremental patches use it.
                raise ValueError("JSON patch target is not a trusted redacted receipt")
            lowered = original.casefold()
            if _contains_secret_key(original) or (
                ("http://" in lowered or "https://" in lowered)
                and _SIGNED_URL.search(original)
            ):
                raise ValueError("JSON patch target contains secret-bearing fields")
        spans = (
            patch_state.root_spans
            if state_is_current and patch_state is not None
            else _top_level_json_value_spans(original)
        )
        missing = [key for key in updates if key not in spans]
        if missing:
            raise ValueError(f"JSON patch fields are absent: {', '.join(sorted(missing))}")
        array_updates = array_item_updates or {}
        overlap = set(updates).intersection(array_updates)
        if overlap:
            raise ValueError(
                "JSON patch fields cannot be both root and array updates: "
                + ", ".join(sorted(overlap))
            )
        replacements: list[tuple[int, int, str]] = []
        for key, value in updates.items():
            start, end = spans[key]
            encoded = json.dumps(
                redact(value),
                ensure_ascii=False,
                indent=None if compact_target else 2,
                separators=(",", ":") if compact_target else None,
                sort_keys=True,
            )
            replacements.append((start, end, _indent_json_fragment(original, start, encoded)))
        cached_array_items: dict[str, list[tuple[int, int, Any]]] = {}
        replacement_values: dict[tuple[int, int], tuple[str, Any]] = {}
        for key, (identity_path, item_updates) in array_updates.items():
            if key not in spans:
                raise ValueError(f"JSON array patch field is absent: {key}")
            item_spans = (
                patch_state.array_items.get(key)
                if state_is_current and patch_state is not None
                else None
            )
            if item_spans is None:
                item_spans = _array_item_spans(original, *spans[key])
            cached_array_items[key] = item_spans
            by_identity = {
                str(_dotted_value(item, identity_path)): (item_start, item_end)
                for item_start, item_end, item in item_spans
                if _dotted_value(item, identity_path) is not None
            }
            for identity, value in item_updates.items():
                item_span = by_identity.get(str(identity))
                if item_span is None:
                    raise ValueError(f"JSON array item is absent: {key}#{identity}")
                item_start, item_end = item_span
                encoded = json.dumps(
                    redact(value),
                    ensure_ascii=False,
                    indent=None if compact_target else 2,
                    separators=(",", ":") if compact_target else None,
                    sort_keys=True,
                )
                replacement = _indent_json_fragment(original, item_start, encoded)
                replacements.append((item_start, item_end, replacement))
                replacement_values[(item_start, item_end)] = (replacement, value)
        patched_parts: list[str] = []
        cursor = 0
        for start, end, replacement in sorted(replacements):
            patched_parts.append(original[cursor:start])
            patched_parts.append(replacement)
            cursor = end
        patched_parts.append(original[cursor:])
        patched = "".join(patched_parts)
        if patched != original:
            atomic_write_text(path, patched)
        _remember_redacted_file(path)
        if patch_state is not None:
            cached_arrays = dict(patch_state.array_items) if state_is_current else {}
            cached_arrays.update(cached_array_items)
            patch_state.text = patched
            patch_state.compact = compact_target
            patch_state.root_spans = _shift_root_spans(spans, replacements)
            patch_state.array_items = {
                key: _shift_array_items(items, replacements, replacement_values)
                for key, items in cached_arrays.items()
                if key not in updates
            }
            patch_state.signature = _file_signature(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        if fallback_payload is None:
            raise
        atomic_write_json(path, fallback_payload, compact=compact_target)
        if patch_state is not None:
            try:
                patch_state.refresh(path)
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                patch_state.clear()
