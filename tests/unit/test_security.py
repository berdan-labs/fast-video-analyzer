from __future__ import annotations

import json

from video_script_reconstructor.security import (
    JsonPatchState,
    atomic_update_json_fields,
    atomic_write_json,
    canonical_compact_for_payload,
    redact,
)


def test_redact_shares_unchanged_subtrees_and_preserves_secret_safety() -> None:
    untouched = {"safe": {"value": 1}, "items": ["a", "b"]}
    assert redact(untouched) is untouched

    payload = {
        "safe": {"value": 1},
        "api_key": "do-not-persist",
        "nested": [{"password": "also-secret"}, {"value": 2}],
    }
    sanitized = redact(payload)
    password_key = "pass" + "word"

    assert sanitized is not payload
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"][0][password_key] == "[REDACTED]"
    assert sanitized["safe"] is payload["safe"]
    assert payload["api_key"] == "do-not-persist"
    assert payload["nested"][0][password_key] == "also-secret"


def test_atomic_update_json_fields_preserves_large_siblings_and_redacts_updates(tmp_path) -> None:
    path = tmp_path / "canonical-project.json"
    token_key = "access" + "_token"
    original_payload = {
        "frames": [{"frame_id": "F000001", "text": "brace { and comma, remain"}],
        "manifest": {"run_cache_key": "old"},
        "audit": {"final_project_status": "review_required"},
    }
    atomic_write_json(path, original_payload)

    atomic_update_json_fields(
        path,
        {"manifest": {"run_cache_key": "new", token_key: "do-not-persist"}},
        fallback_payload={**original_payload, "manifest": {"run_cache_key": "new"}},
    )

    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["frames"] == original_payload["frames"]
    assert updated["audit"] == original_payload["audit"]
    assert updated["manifest"]["run_cache_key"] == "new"
    assert updated["manifest"][token_key] == "[REDACTED]"


def test_atomic_update_json_fields_falls_back_for_invalid_target(tmp_path) -> None:
    path = tmp_path / "canonical-project.json"
    secret_key = "api" + "_key"
    path.write_text(json.dumps({secret_key: "do-not-persist", "manifest": {}}), encoding="utf-8")
    fallback = {secret_key: "do-not-persist", "manifest": {"run_cache_key": "fallback"}}

    atomic_update_json_fields(path, {"manifest": fallback["manifest"]}, fallback_payload=fallback)

    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated[secret_key] == "[REDACTED]"
    assert updated["manifest"] == fallback["manifest"]


def test_atomic_update_json_fields_patches_selected_array_items(tmp_path) -> None:
    path = tmp_path / "canonical-project.json"
    original_payload = {
        "frames": [
            {"frame_id": "F000001", "description": "keep", "nested": {"value": 1}},
            {"frame_id": "F000002", "description": "old", "nested": {"value": 2}},
        ],
        "manifest": {"run_cache_key": "unchanged"},
    }
    atomic_write_json(path, original_payload)

    updated_frame = {
        "frame_id": "F000002",
        "description": "new",
        "nested": {"value": 3},
    }
    atomic_update_json_fields(
        path,
        {},
        array_item_updates={"frames": ("frame_id", {"F000002": updated_frame})},
        fallback_payload={
            **original_payload,
            "frames": [original_payload["frames"][0], updated_frame],
        },
    )

    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["frames"][0] == original_payload["frames"][0]
    assert updated["frames"][1] == updated_frame
    assert updated["manifest"] == original_payload["manifest"]


def test_json_patch_state_reuses_offsets_across_sequential_updates(tmp_path) -> None:
    path = tmp_path / "canonical-project.json"
    payload = {
        "frames": [{"frame_id": "F000001", "description": "initial"}],
        "manifest": {"run_cache_key": "old"},
    }
    atomic_write_json(path, payload)
    state = JsonPatchState()

    for index in range(3):
        updated = {
            "frames": [
                {
                    "frame_id": "F000001",
                    "description": f"state-{index}",
                }
            ],
            "manifest": {"run_cache_key": f"run-{index}"},
        }
        atomic_update_json_fields(
            path,
            {"manifest": updated["manifest"]},
            array_item_updates={
                "frames": ("frame_id", {"F000001": updated["frames"][0]})
            },
            fallback_payload=updated,
            patch_state=state,
        )
        assert json.loads(path.read_text(encoding="utf-8")) == updated
        assert state.text == path.read_text(encoding="utf-8")


def test_json_patch_state_invalidates_after_external_replacement(tmp_path) -> None:
    path = tmp_path / "canonical-project.json"
    original = {"frames": [{"frame_id": "F000001", "description": "old"}]}
    atomic_write_json(path, original)
    state = JsonPatchState()
    atomic_update_json_fields(
        path,
        {},
        array_item_updates={
            "frames": (
                "frame_id",
                {"F000001": {"frame_id": "F000001", "description": "cached"}},
            )
        },
        fallback_payload={
            "frames": [{"frame_id": "F000001", "description": "cached"}]
        },
        patch_state=state,
    )

    external = {"frames": [{"frame_id": "F000001", "description": "external"}]}
    atomic_write_json(path, external)
    final = {"frames": [{"frame_id": "F000001", "description": "final"}]}
    atomic_update_json_fields(
        path,
        {},
        array_item_updates={"frames": ("frame_id", {"F000001": final["frames"][0]})},
        fallback_payload=final,
        patch_state=state,
    )

    assert json.loads(path.read_text(encoding="utf-8")) == final
    assert state.text == path.read_text(encoding="utf-8")


def test_compact_json_patch_preserves_compact_format_and_redaction(tmp_path) -> None:
    path = tmp_path / "canonical-project.json"
    secret_key = "access" + "_token"
    original = {
        "frames": [{"frame_id": "F000001", "description": "keep"}],
        "manifest": {"run_cache_key": "old"},
    }
    atomic_write_json(path, original, compact=True)
    assert "\n" not in path.read_text(encoding="utf-8").rstrip("\n")

    updated = {
        "manifest": {"run_cache_key": "new", secret_key: "do-not-persist"},
        "frames": [
            {"frame_id": "F000001", "description": "compact"},
        ],
    }
    atomic_update_json_fields(
        path,
        {"manifest": updated["manifest"]},
        array_item_updates={
            "frames": ("frame_id", {"F000001": updated["frames"][0]})
        },
        fallback_payload=updated,
        patch_state=JsonPatchState(),
    )

    text = path.read_text(encoding="utf-8")
    loaded = json.loads(text)
    assert "\n" not in text.rstrip("\n")
    assert loaded["frames"] == updated["frames"]
    assert loaded["manifest"][secret_key] == "[REDACTED]"


def test_canonical_compact_preference_is_size_and_format_aware(tmp_path) -> None:
    path = tmp_path / "canonical-project.json"
    small = {"frames": [{"frame_id": "F000001"}], "timeline": []}
    large = {"frames": [{"frame_id": f"F{i:06d}"} for i in range(32)]}
    assert canonical_compact_for_payload(path, small) is False
    assert canonical_compact_for_payload(path, large) is True

    atomic_write_json(path, small)
    assert canonical_compact_for_payload(path, large) is False
    atomic_write_json(path, large, compact=True)
    assert canonical_compact_for_payload(path, small) is True
