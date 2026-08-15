"""Strict recursive configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import jsonschema
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(ValueError):
    """Configuration is malformed, unsafe, or violates a workflow invariant."""


_SCHEMA_VALIDATOR_CACHE: tuple[tuple[str, int, int, int, int], Any] | None = None
# The shipped schema is generated from ``AppConfig`` and packaged as a
# read-only resource.  Pinning its digest lets normal startup skip the costly
# jsonschema self-check while a modified/malformed resource still takes the
# complete defensive path below.
_SHIPPED_SCHEMA_SHA256 = "e425281565e65c5b1ec0d67b40a2255c613659cc3eade5b9d0309e8bddd6c68d"


class ConfigSection(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TranscriptConfig(ConfigSection):
    source_mode: Literal["auto", "provided", "embedded", "asr"] = "auto"
    compare_candidates: bool = True
    selective_repair: bool = True
    preserve_all_states: bool = True
    high_impact_tokens_block: bool = True


class ASRConfig(ConfigSection):
    # Whisper large-v3 is the default multilingual/Filipino transcription
    # authority.  Other local model stacks remain compatibility adapters only.
    backend: str = "faster-whisper"
    model: str = "faster-whisper-large-v3"
    language: str | None = None
    device: str = "auto"
    compute_type: str = "auto"
    word_timestamps: bool = True
    vad: Literal["off", "conservative", "default", "aggressive"] = "conservative"
    chunk_seconds: int = Field(default=900, gt=0)
    overlap_seconds: int = Field(default=15, ge=0)
    allow_model_download: bool = False

    @model_validator(mode="after")
    def overlap_is_bounded(self) -> ASRConfig:
        if self.overlap_seconds >= self.chunk_seconds:
            raise ValueError("asr.overlap_seconds must be less than chunk_seconds")
        return self


class SpeakerConfig(ConfigSection):
    diarization: Literal["off", "optional", "required"] = "optional"
    infer_identity: bool = False


class VisualConfig(ConfigSection):
    survey_interval_seconds: float = Field(default=30, gt=0)
    profile: str = "adaptive"
    scene_detection: bool = True
    frame_difference: bool = True
    ocr: bool = True
    semantic_annotation: Literal["required", "required_or_review", "optional", "none"] = (
        "required_or_review"
    )
    before_action_after: bool = True
    deduplicate: bool = True
    protect_small_changes: bool = True
    full_frame_required_for_crop: bool = True


class ImageMetadataConfig(ConfigSection):
    enabled: bool = True
    embed_in_every_generated_image: bool = True
    canonical_mirror: bool = True
    preferred_evidence_format: Literal["png"] = "png"
    png_itxt_keyword: str = "video-script-reconstructor"
    write_human_description: bool = True
    validate_after_every_write: bool = True
    require_pixel_invariance: bool = True
    preserve_observation_history: bool = True
    read_before_reanalysis: bool = True
    auto_enrich: bool = True
    blind_check_high_impact_or_disputed: bool = True
    mirror_consumed_facts_in_markdown: bool = True
    stop_after_no_new_supported_information_passes: int = Field(default=2, ge=1)
    on_limit_with_unresolved: Literal["review_required", "blocked"] = "review_required"


class ScriptConfig(ConfigSection):
    fidelity_mode: Literal["verbatim", "clean-verbatim", "production-script"] = "verbatim"
    max_block_seconds: int = Field(default=60, gt=0)
    preserve_non_speech: bool = True
    preserve_sponsor_segments: bool = True
    derive_navigation_chapters: bool = True
    max_navigation_chapter_seconds: int = Field(default=600, gt=0)


class VerificationConfig(ConfigSection):
    substantive_segment_coverage_target: float = Field(default=1.0, ge=0.0, le=1.0)
    ordered_meaning_coverage_target: float = Field(default=1.0, ge=0.0, le=1.0)
    block_on_missing_segments: bool = True
    block_on_unsupported_statements: bool = True
    require_review_for_high_impact_uncertainty: bool = True
    automatic_fully_verified: bool = False


class PrivacyConfig(ConfigSection):
    offline: bool = True
    allow_remote_download: bool = False
    allow_external_ai: bool = False


class OutputConfig(ConfigSection):
    exactly_one_markdown: bool = True
    include_images_inline: bool = True
    include_hidden_canonical_state: bool = True
    allow_html: bool = False
    allow_additional_markdown: bool = False


class AppConfig(ConfigSection):
    schema_version: Literal["1.0"] = "1.0"
    preset: Literal["strict", "balanced"] = "strict"
    transcript: TranscriptConfig = Field(default_factory=TranscriptConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    speaker: SpeakerConfig = Field(default_factory=SpeakerConfig)
    visual: VisualConfig = Field(default_factory=VisualConfig)
    image_metadata: ImageMetadataConfig = Field(default_factory=ImageMetadataConfig)
    script: ScriptConfig = Field(default_factory=ScriptConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def mandatory_invariants(self) -> AppConfig:
        metadata = self.image_metadata
        mandatory_metadata = {
            "enabled": metadata.enabled,
            "embed_in_every_generated_image": metadata.embed_in_every_generated_image,
            "canonical_mirror": metadata.canonical_mirror,
            "validate_after_every_write": metadata.validate_after_every_write,
            "require_pixel_invariance": metadata.require_pixel_invariance,
            "preserve_observation_history": metadata.preserve_observation_history,
        }
        disabled = [name for name, enabled in mandatory_metadata.items() if not enabled]
        if disabled:
            raise ValueError(
                "normal presets may not disable image metadata invariants: " + ", ".join(disabled)
            )
        if metadata.png_itxt_keyword != "video-script-reconstructor":
            raise ValueError("image_metadata.png_itxt_keyword is fixed by the artifact contract")
        output = self.output
        if not output.exactly_one_markdown or output.allow_html or output.allow_additional_markdown:
            raise ValueError("the exactly-one-Markdown/no-HTML output contract cannot be disabled")
        if self.speaker.infer_identity:
            raise ValueError("speaker identity inference is prohibited")
        if self.verification.automatic_fully_verified:
            raise ValueError("automatic checks cannot mark a project fully verified")
        if self.privacy.offline and (
            self.privacy.allow_remote_download or self.privacy.allow_external_ai
        ):
            raise ValueError(
                "offline mode conflicts with remote download or external AI permission"
            )
        if self.preset == "strict":
            strict_requirements = {
                "compare_candidates": self.transcript.compare_candidates,
                "selective_repair": self.transcript.selective_repair,
                "preserve_all_states": self.transcript.preserve_all_states,
                "high_impact_tokens_block": self.transcript.high_impact_tokens_block,
                "read_before_reanalysis": metadata.read_before_reanalysis,
                "auto_enrich": metadata.auto_enrich,
                "blind_check_high_impact_or_disputed": metadata.blind_check_high_impact_or_disputed,
                "mirror_consumed_facts_in_markdown": metadata.mirror_consumed_facts_in_markdown,
                "protect_small_changes": self.visual.protect_small_changes,
                "full_frame_required_for_crop": self.visual.full_frame_required_for_crop,
            }
            missing = [name for name, enabled in strict_requirements.items() if not enabled]
            if missing:
                raise ValueError("strict preset invariant disabled: " + ", ".join(missing))
            if self.script.fidelity_mode != "verbatim":
                raise ValueError("strict preset requires script.fidelity_mode=verbatim")
        return self


def _config_root() -> Path:
    source_root = Path(__file__).resolve().parents[2] / "configs"
    if source_root.is_dir():
        return source_root
    packaged = Path(__file__).resolve().parent / "resources" / "configs"
    if packaged.is_dir():
        return packaged
    raise ConfigError("packaged configuration resources are missing")


def _merge_known(
    base: dict[str, Any], overrides: Mapping[str, Any], path: tuple[str, ...] = ()
) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        current_path = path + (str(key),)
        if key not in base:
            raise ConfigError(f"unknown configuration key: {'.'.join(current_path)}")
        base_value = base[key]
        if isinstance(base_value, dict):
            if not isinstance(value, Mapping):
                raise ConfigError(
                    f"configuration section {'.'.join(current_path)} must be an object"
                )
            result[key] = _merge_known(base_value, value, current_path)
        elif isinstance(value, Mapping):
            raise ConfigError(f"configuration value {'.'.join(current_path)} is not an object")
        else:
            result[key] = value
    return result


def merge_config(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides, rejecting unknown keys at every depth."""

    return _merge_known(dict(base), overrides)


def dotted_overrides(values: Mapping[str, Any]) -> dict[str, Any]:
    """Turn ``{"asr.model": "..."}`` into a nested override mapping."""

    result: dict[str, Any] = {}
    for dotted_key, value in values.items():
        parts = dotted_key.split(".")
        if any(not part for part in parts):
            raise ConfigError(f"invalid dotted configuration key: {dotted_key!r}")
        target = result
        for part in parts[:-1]:
            existing = target.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ConfigError(f"conflicting dotted configuration key: {dotted_key!r}")
            target = existing
        if parts[-1] in target:
            raise ConfigError(f"duplicate configuration override: {dotted_key!r}")
        target[parts[-1]] = value
    return result


def _schema_validator() -> Any:
    """Load and validate the shipped JSON Schema once per unchanged file."""

    global _SCHEMA_VALIDATOR_CACHE
    schema_path = _config_root() / "schema.json"
    try:
        stat = schema_path.stat()
        signature = (
            str(schema_path),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(getattr(stat, "st_ctime_ns", 0)),
            int(getattr(stat, "st_ino", 0)),
        )
    except OSError as exc:
        raise ConfigError(f"unable to load configuration JSON Schema: {exc}") from exc
    if _SCHEMA_VALIDATOR_CACHE is not None and _SCHEMA_VALIDATOR_CACHE[0] == signature:
        return _SCHEMA_VALIDATOR_CACHE[1]
    try:
        schema_bytes = schema_path.read_bytes()
        schema = json.loads(schema_bytes.decode("utf-8"))
        validator_class = jsonschema.validators.validator_for(schema)
        schema_digest = hashlib.sha256(schema_bytes).hexdigest()
        if schema_digest != _SHIPPED_SCHEMA_SHA256:
            validator_class.check_schema(schema)
        validator = validator_class(schema)
    except OSError as exc:
        raise ConfigError(f"unable to load configuration JSON Schema: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"shipped configuration JSON Schema is invalid: {exc.msg}") from exc
    except jsonschema.exceptions.SchemaError as exc:
        raise ConfigError(f"shipped configuration JSON Schema is invalid: {exc.message}") from exc
    _SCHEMA_VALIDATOR_CACHE = (signature, validator)
    return validator


def validate_config(data: Mapping[str, Any]) -> AppConfig:
    try:
        _schema_validator().validate(dict(data))
    except jsonschema.exceptions.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ConfigError(f"JSON Schema validation failed at {location}: {exc.message}") from exc
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def load_config(
    preset: str | Path = "strict",
    overrides: Mapping[str, Any] | None = None,
    *,
    override_is_dotted: bool = False,
) -> AppConfig:
    """Load a preset or YAML path, merge explicit overrides, and validate it."""

    candidate = Path(preset)
    if candidate.suffix.lower() in {".yaml", ".yml"} or candidate.exists():
        path = candidate
    else:
        if str(preset) not in {"strict", "balanced"}:
            raise ConfigError(f"unknown preset: {preset}")
        path = _config_root() / f"{preset}.yaml"
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"unable to load configuration {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError("configuration root must be an object")
    if overrides:
        nested = dotted_overrides(overrides) if override_is_dotted else dict(overrides)
        loaded = merge_config(loaded, nested)
    return validate_config(loaded)


def config_digest(config: AppConfig | Mapping[str, Any]) -> str:
    data = (
        config.model_dump(mode="json")
        if isinstance(config, AppConfig)
        else validate_config(config).model_dump(mode="json")
    )
    encoded = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generated_json_schema() -> dict[str, Any]:
    return AppConfig.model_json_schema()


__all__ = [
    "AppConfig",
    "ConfigError",
    "config_digest",
    "dotted_overrides",
    "generated_json_schema",
    "load_config",
    "merge_config",
    "validate_config",
]
