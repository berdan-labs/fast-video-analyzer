from __future__ import annotations

import json
import shutil
from pathlib import Path

import jsonschema
import pytest
import yaml

import video_script_reconstructor.config as config_module
from video_script_reconstructor.config import (
    ConfigError,
    config_digest,
    generated_json_schema,
    load_config,
)

ROOT = Path(__file__).resolve().parents[2]


def test_presets_validate_against_shipped_json_schema() -> None:
    schema = json.loads((ROOT / "configs" / "schema.json").read_text(encoding="utf-8"))
    assert schema == generated_json_schema()
    for name in ("strict", "balanced"):
        raw = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
        jsonschema.validate(raw, schema)
        parsed = load_config(name)
        assert parsed.preset == name
        assert parsed.script.fidelity_mode == "verbatim"
        assert parsed.image_metadata.require_pixel_invariance


def test_modified_schema_falls_back_to_full_schema_self_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copy2(ROOT / "configs" / "strict.yaml", tmp_path / "strict.yaml")
    modified_schema = json.loads((ROOT / "configs" / "schema.json").read_text(encoding="utf-8"))
    modified_schema["type"] = "not-a-json-schema-type"
    (tmp_path / "schema.json").write_text(
        json.dumps(modified_schema), encoding="utf-8"
    )
    monkeypatch.setattr(config_module, "_config_root", lambda: tmp_path)
    config_module._SCHEMA_VALIDATOR_CACHE = None

    with pytest.raises(ConfigError, match="JSON Schema validation failed|shipped configuration"):
        config_module.load_config("strict")


def test_recursive_override_and_digest_are_deterministic() -> None:
    changed = load_config("strict", {"asr.model": "fixture-model"}, override_is_dotted=True)
    again = load_config("strict", {"asr.model": "fixture-model"}, override_is_dotted=True)
    assert changed.asr.model == "fixture-model"
    assert config_digest(changed) == config_digest(again)


def test_unknown_nested_override_and_contract_disable_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown configuration key"):
        load_config("strict", {"visual": {"invented": True}})
    with pytest.raises(ConfigError, match="exactly-one-Markdown"):
        load_config("strict", {"output": {"allow_html": True}})
    with pytest.raises(ConfigError, match="strict preset invariant"):
        load_config("strict", {"image_metadata": {"read_before_reanalysis": False}})


def test_vision_none_retains_creation_metadata_contract() -> None:
    config = load_config("strict", {"visual": {"semantic_annotation": "none"}})
    assert config.visual.semantic_annotation == "none"
    assert config.image_metadata.embed_in_every_generated_image
    assert config.image_metadata.canonical_mirror
