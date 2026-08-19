"""Create bounded, sanitized support bundles without copying user media."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .errors import InputError

_PATH_PATTERN = re.compile(r"(?i)(?<![\w])(?:[a-z]:[\\/][^\r\n,;]+|/(?:[^/\s]+/)+[^\s,;]*)")
_SECRET_PATTERN = re.compile(
    r"(?i)\b(?:authorization|api[_-]?key|cookie|password|secret|token)\b\s*[:=]\s*[^\s,;]+"
)
_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "command",
    "configured_command",
    "configured_executable",
    "cookie",
    "credential",
    "directory",
    "executable",
    "home",
    "input_reference",
    "library",
    "path",
    "password",
    "python",
    "secret",
    "source_reference",
    "token",
    "username",
}


def _sanitize_string(value: str) -> str:
    value = _SECRET_PATTERN.sub("[REDACTED]", value)
    value = _PATH_PATTERN.sub("[REDACTED_PATH]", value)
    return value[:500]


def sanitize_diagnostic_value(value: Any, *, key: str | None = None) -> Any:
    """Recursively remove paths, credentials, and unbounded text from diagnostics."""

    normalized_key = key.casefold().replace("-", "_") if key else ""
    if normalized_key in _SENSITIVE_KEYS:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(item_key): sanitize_diagnostic_value(item_value, key=str(item_key))
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_diagnostic_value(item) for item in value[:100]]
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return _sanitize_string(str(value))


def _safe_runtime_report() -> dict[str, Any]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "machine": platform.machine(),
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def create_diagnostic_bundle(output: str | Path, *, force: bool = False) -> dict[str, Any]:
    """Write a ZIP support bundle containing only sanitized capability metadata.

    The bundle intentionally excludes source media, transcripts, screenshots,
    generated projects, environment variables, command lines, and host paths.
    Existing files are never overwritten unless ``force`` is explicit.
    """

    output_path = Path(output).expanduser().resolve()
    if output_path.suffix.casefold() != ".zip":
        raise InputError("diagnostic bundle output must use a .zip extension")
    if output_path.exists() and not force:
        raise InputError(f"Refusing to overwrite existing diagnostic bundle: {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from .pipeline import doctor_report

    doctor = sanitize_diagnostic_value(doctor_report(output_path=output_path.parent, offline=True))
    runtime = _safe_runtime_report()
    files: dict[str, bytes] = {
        "README.txt": (
            b"This support bundle contains sanitized capability metadata only.\n"
            b"It deliberately excludes source media, transcripts, screenshots, generated projects,\n"
            b"environment variables, credentials, command lines, and host filesystem paths.\n"
        ),
        "runtime.json": _json_bytes(runtime),
        "doctor.json": _json_bytes(doctor),
    }
    manifest = {
        "format": "fast-video-analyzer-diagnostic-bundle",
        "format_version": 1,
        "application": "fast-video-analyzer",
        "application_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "media_included": False,
        "paths_included": False,
        "credentials_included": False,
        "files": {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in files.items()
        },
    }
    files["manifest.json"] = _json_bytes(manifest)
    try:
        with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(files):
                archive.writestr(name, files[name])
    except (OSError, zipfile.BadZipFile) as exc:
        output_path.unlink(missing_ok=True)
        raise InputError(f"Unable to write diagnostic bundle: {exc}") from exc
    return {
        "bundle": str(output_path),
        "format": manifest["format"],
        "format_version": manifest["format_version"],
        "media_included": False,
        "paths_included": False,
        "credentials_included": False,
        "files": sorted(files),
    }


__all__ = ["create_diagnostic_bundle", "sanitize_diagnostic_value"]
