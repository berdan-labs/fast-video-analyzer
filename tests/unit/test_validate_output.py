from __future__ import annotations

from pathlib import Path

import video_script_reconstructor.validate_output as validate_output_module


def test_image_link_parser_accepts_escaped_brackets_in_alt_text() -> None:
    markdown = (
        r"![Slide text \[Pickup City\] remains visible]"
        "(evidence/full/F000108__00h30m30s000__full.png)"
    )
    assert validate_output_module._IMAGE.findall(markdown) == [
        "evidence/full/F000108__00h30m30s000__full.png"
    ]


def test_output_file_inventory_uses_scandir_and_prunes_state(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "lesson.md").write_text("# lesson", encoding="utf-8")
    state = tmp_path / ".state"
    state.mkdir()
    (state / "hidden.md").write_text("hidden", encoding="utf-8")

    def fail_rglob(*_args, **_kwargs):
        raise AssertionError("output inventory must recurse with scandir")

    monkeypatch.setattr(validate_output_module.Path, "rglob", fail_rglob)

    assert [
        path.relative_to(tmp_path).as_posix()
        for path in validate_output_module._iter_output_files(tmp_path, skip_state=True)
    ] == ["nested/lesson.md"]


def test_metadata_validation_workers_are_bounded_and_overridable(monkeypatch) -> None:
    monkeypatch.setenv("VSR_VALIDATOR_METADATA_WORKERS", "12")
    assert validate_output_module._metadata_verify_workers() == 12
    monkeypatch.setenv("VSR_VALIDATOR_METADATA_WORKERS", "999")
    assert validate_output_module._metadata_verify_workers() == 16
    monkeypatch.setenv("VSR_VALIDATOR_METADATA_WORKERS", "invalid")
    assert validate_output_module._metadata_verify_workers() >= 1


def test_internal_file_hash_cache_reuses_unchanged_bytes_and_invalidates_on_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "evidence.png"
    path.write_bytes(b"first bytes")
    validate_output_module._FILE_HASH_CACHE.clear()
    original = validate_output_module.sha256_file
    calls = 0

    def counted(target: Path) -> str:
        nonlocal calls
        calls += 1
        return original(target)

    monkeypatch.setattr(validate_output_module, "sha256_file", counted)
    first = validate_output_module._cached_file_hash(path)
    assert validate_output_module._cached_file_hash(path) == first
    assert calls == 1

    path.write_bytes(b"changed bytes")
    changed = validate_output_module._cached_file_hash(path)
    assert changed != first
    assert calls == 2


def test_internal_metadata_cache_reuses_same_file_and_payload(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "evidence.png"
    path.write_bytes(b"first bytes")
    validate_output_module._METADATA_VERIFY_CACHE.clear()
    calls = 0

    def counted(target: Path, payload, *, expected_pixel_hash: str | None):
        nonlocal calls
        calls += 1
        return (target.name, payload, expected_pixel_hash)

    payload = {"integrity": {"payload_digest": "digest-1"}}
    first = validate_output_module._cached_verify_embedded_metadata(
        path,
        payload,
        expected_pixel_hash="pixel-1",
        verifier=counted,
    )
    assert (
        validate_output_module._cached_verify_embedded_metadata(
            path,
            payload,
            expected_pixel_hash="pixel-1",
            verifier=counted,
        )
        == first
    )
    assert calls == 1

    changed_payload = {"integrity": {"payload_digest": "digest-2"}}
    validate_output_module._cached_verify_embedded_metadata(
        path,
        changed_payload,
        expected_pixel_hash="pixel-1",
        verifier=counted,
    )
    assert calls == 2


def test_internal_canonical_cache_reuses_and_invalidates_stat_bound_json(tmp_path: Path) -> None:
    project = tmp_path
    state = project / ".state"
    state.mkdir()
    path = state / "canonical-project.json"
    path.write_text('{"version": 1}', encoding="utf-8")
    validate_output_module._CANONICAL_CACHE.clear()
    validate_output_module._CANONICAL_MODEL_CACHE.clear()
    validate_output_module._AUDIT_CACHE.clear()

    first = validate_output_module._load_canonical(project, use_cache=True)
    second = validate_output_module._load_canonical(project, use_cache=True)
    assert first is second
    assert first == {"version": 1}

    path.write_text('{"version": 2, "changed": true}', encoding="utf-8")
    changed = validate_output_module._load_canonical(project, use_cache=True)
    assert changed == {"version": 2, "changed": True}
    assert changed is not first


def test_internal_canonical_model_cache_reuses_and_invalidates_signature(tmp_path: Path) -> None:
    project = tmp_path
    state = project / ".state"
    state.mkdir()
    path = state / "canonical-project.json"
    path.write_text('{"version": 1}', encoding="utf-8")
    validate_output_module._CANONICAL_MODEL_CACHE.clear()
    calls = 0

    def validator(payload):
        nonlocal calls
        calls += 1
        return {"model": payload["version"]}

    payload = {"version": 1}
    first = validate_output_module._cached_canonical_model(path, payload, validator)
    second = validate_output_module._cached_canonical_model(path, payload, validator)
    assert first is second
    assert calls == 1

    path.write_text('{"version": 2, "changed": true}', encoding="utf-8")
    changed = validate_output_module._cached_canonical_model(path, {"version": 2}, validator)
    assert changed == {"model": 2}
    assert calls == 2


def test_internal_audit_cache_reuses_and_invalidates_signature(tmp_path: Path) -> None:
    state = tmp_path / ".state"
    state.mkdir()
    path = state / "canonical-project.json"
    path.write_text('{"version": 1}', encoding="utf-8")
    validate_output_module._AUDIT_CACHE.clear()
    calls = 0

    def auditor(payload):
        nonlocal calls
        calls += 1
        return {"version": payload["version"]}

    payload = {"version": 1}
    first = validate_output_module._cached_audit(path, payload, auditor)
    second = validate_output_module._cached_audit(path, payload, auditor)
    assert first is second
    assert calls == 1

    path.write_text('{"version": 2, "changed": true}', encoding="utf-8")
    changed = validate_output_module._cached_audit(path, {"version": 2}, auditor)
    assert changed == {"version": 2}
    assert calls == 2


def test_internal_validation_result_cache_reuses_and_invalidates_inventory(
    tmp_path: Path,
) -> None:
    project = tmp_path
    state = project / ".state"
    state.mkdir()
    (state / "canonical-project.json").write_text('{"project_status":"blocked"}', encoding="utf-8")
    artifact = project / "artifact.txt"
    artifact.write_text("first", encoding="utf-8")
    validate_output_module._VALIDATION_RESULT_CACHE.clear()

    original = validate_output_module.ValidationResult(
        True,
        errors=[],
        warnings=["cached"],
        checks={"metadata_verified": True},
        project_status="blocked",
    )
    validate_output_module._remember_validation_result(
        project,
        verify_metadata=True,
        use_cached_file_hash=True,
        result=original,
    )
    cached, _signature = validate_output_module._cached_validation_result(
        project,
        verify_metadata=True,
        use_cached_file_hash=True,
    )
    assert cached is not None and cached is not original
    cached.warnings.append("caller mutation")
    cached_again, _signature = validate_output_module._cached_validation_result(
        project,
        verify_metadata=True,
        use_cached_file_hash=True,
    )
    assert cached_again is not None
    assert cached_again.warnings == ["cached"]

    artifact.write_text("changed", encoding="utf-8")
    invalidated, _signature = validate_output_module._cached_validation_result(
        project,
        verify_metadata=True,
        use_cached_file_hash=True,
    )
    assert invalidated is None
