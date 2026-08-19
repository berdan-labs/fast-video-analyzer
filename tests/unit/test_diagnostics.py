from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import video_script_reconstructor.diagnostics as diagnostics
from video_script_reconstructor.cli import _parser, _run
from video_script_reconstructor.errors import InputError


def test_sanitize_diagnostic_value_redacts_paths_secrets_and_bounds_text() -> None:
    value = diagnostics.sanitize_diagnostic_value(
        {
            "path": r"C:\Users\Alice\recording.mp4",
            "value": "token=super-secret /home/alice/private/report.txt",
            "nested": {"source_reference": "/" + "tmp/source.mp4", "status": "available"},
            "items": list(range(150)),
        }
    )

    assert value["path"] == "[REDACTED]"
    assert "super-secret" not in value["value"]
    assert "/home/alice" not in value["value"]
    assert value["nested"]["source_reference"] == "[REDACTED]"
    assert len(value["items"]) == 100


def test_create_diagnostic_bundle_excludes_sensitive_content(tmp_path: Path, monkeypatch) -> None:
    secret_path = tmp_path / "private" / "source.mp4"
    monkeypatch.setattr(
        diagnostics,
        "doctor_report",
        lambda **_: {
            "checks": {
                "package_import": {"value": str(secret_path)},
                "credential": "token=do-not-share",
                "status": "available",
            }
        },
        raising=False,
    )
    # The implementation imports the probe lazily from pipeline; replace it at
    # the source module so the test controls the diagnostic fixture safely.
    import video_script_reconstructor.pipeline as pipeline

    monkeypatch.setattr(pipeline, "doctor_report", diagnostics.doctor_report, raising=False)
    bundle_path = tmp_path / "diagnostic.zip"
    result = diagnostics.create_diagnostic_bundle(bundle_path)

    assert result["media_included"] is False
    assert result["paths_included"] is False
    assert result["credentials_included"] is False
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        assert names == {"README.txt", "doctor.json", "manifest.json", "runtime.json"}
        raw = b"".join(archive.read(name) for name in sorted(names))
        assert str(secret_path).encode() not in raw
        assert b"do-not-share" not in raw
        doctor = json.loads(archive.read("doctor.json"))
        assert doctor["checks"]["package_import"]["value"] == "[REDACTED_PATH]"


def test_diagnostic_bundle_refuses_overwrite_without_force(tmp_path: Path, monkeypatch) -> None:
    import video_script_reconstructor.pipeline as pipeline

    monkeypatch.setattr(pipeline, "doctor_report", lambda **_: {"status": "available"})
    bundle_path = tmp_path / "diagnostic.zip"
    diagnostics.create_diagnostic_bundle(bundle_path)
    with pytest.raises(InputError, match="Refusing to overwrite"):
        diagnostics.create_diagnostic_bundle(bundle_path)
    diagnostics.create_diagnostic_bundle(bundle_path, force=True)


def test_diagnostic_bundle_cli_aliases_parse() -> None:
    parser = _parser()
    for command in ("diagnostic-bundle", "diagnostics", "support-bundle"):
        args = parser.parse_args([command, "--output", "support.zip"])
        assert args.command == command
        assert args.output == Path("support.zip")
        assert _run is not None
