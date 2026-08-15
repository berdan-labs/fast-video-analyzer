"""Deterministic, lossless transcript-to-script reconstruction helpers."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from hashlib import sha256
from typing import Any

LOGGER = logging.getLogger(__name__)


class ReconstructionError(ValueError):
    """Raised when source mappings would become lossy or untraceable."""


@dataclass(slots=True)
class ScriptBlockData:
    block_id: str
    start_ms: int | None
    end_ms: int | None
    speaker_label: str | None
    text: str
    source_segment_ids: list[str]
    source_texts: list[str] = field(default_factory=list)
    visual_event_ids: list[str] = field(default_factory=list)
    block_kind: str = "speech"
    residual: bool = False
    uncertainty_items: list[str] = field(default_factory=list)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @property
    def spoken_text(self) -> str:
        return self.text

    @property
    def transcript_segment_ids(self) -> list[str]:
        return self.source_segment_ids

    @property
    def speaker(self) -> str | None:
        return self.speaker_label

    @property
    def residual_source_text(self) -> str | None:
        return self.text if self.residual else None


@dataclass(frozen=True, slots=True)
class OrderedTokenAudit:
    faithful: bool
    source_tokens: tuple[str, ...]
    output_tokens: tuple[str, ...]
    missing_tokens: tuple[str, ...]
    added_tokens: tuple[str, ...]
    substitutions: tuple[tuple[str, str], ...]
    reordered: bool
    opcodes: tuple[tuple[str, int, int, int, int], ...]
    matched_ratio: float

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HighImpactAudit:
    valid: bool
    source_tokens: tuple[str, ...]
    output_tokens: tuple[str, ...]
    missing: tuple[str, ...]
    added: tuple[str, ...]
    reordered: bool


@dataclass(slots=True)
class CoverageAudit:
    valid: bool
    total_substantive_segments: int
    covered_segments: int
    missing_segment_ids: list[str]
    duplicated_segment_ids: list[str]
    partial_segment_ids: list[str]
    residual_text: list[str]
    ordered_token_audit: OrderedTokenAudit


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def segment_text(segment: Any) -> str:
    """Select text by verification provenance without conflating the states."""

    for key in ("human_verified_text", "repaired_text", "normalized_text", "raw_text", "text"):
        value = _get(segment, key)
        if value is not None:
            return str(value)
    return ""


def _segment_id(segment: Any, index: int) -> str:
    value = _get(segment, "segment_id", _get(segment, "id"))
    if value:
        return str(value)
    payload = json.dumps(
        [index, _get(segment, "start_ms"), _get(segment, "end_ms"), segment_text(segment)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"segment_{sha256(payload).hexdigest()[:20]}"


def _block_id(source_ids: Sequence[str], kind: str = "speech") -> str:
    payload = json.dumps([kind, list(source_ids)], separators=(",", ":")).encode("utf-8")
    return f"block_{sha256(payload).hexdigest()[:20]}"


def provisional_blocks(segments: Sequence[Any]) -> list[ScriptBlockData]:
    """Create the required deterministic one-to-one provisional mapping."""

    blocks: list[ScriptBlockData] = []
    for index, segment in enumerate(segments):
        if not bool(_get(segment, "substantive", True)):
            continue
        source_id = _segment_id(segment, index)
        text = segment_text(segment)
        blocks.append(
            ScriptBlockData(
                block_id=_block_id([source_id]),
                start_ms=_get(segment, "start_ms"),
                end_ms=_get(segment, "end_ms"),
                speaker_label=_get(segment, "speaker_label", _get(segment, "speaker")),
                text=text,
                source_segment_ids=[source_id],
                source_texts=[text],
                uncertainty_items=list(_get(segment, "uncertainty_items", []) or []),
            )
        )
    return blocks


def _can_group(left: ScriptBlockData, right: ScriptBlockData, max_gap_ms: int) -> bool:
    if left.block_kind != "speech" or right.block_kind != "speech":
        return False
    if left.speaker_label != right.speaker_label:
        return False
    if left.uncertainty_items or right.uncertainty_items:
        return False
    if left.end_ms is None or right.start_ms is None:
        return left.end_ms is None and right.start_ms is None
    gap = right.start_ms - left.end_ms
    return 0 <= gap <= max_gap_ms


def group_lossless_blocks(
    blocks: Sequence[ScriptBlockData],
    *,
    max_gap_ms: int = 1_500,
    separator: str = "\n\n",
) -> list[ScriptBlockData]:
    """Group adjacent blocks while retaining exact source IDs and source texts."""

    grouped: list[ScriptBlockData] = []
    for block in blocks:
        if grouped and _can_group(grouped[-1], block, max_gap_ms):
            left = grouped[-1]
            source_ids = left.source_segment_ids + block.source_segment_ids
            source_texts = left.source_texts + block.source_texts
            grouped[-1] = ScriptBlockData(
                block_id=_block_id(source_ids),
                start_ms=left.start_ms,
                end_ms=block.end_ms,
                speaker_label=left.speaker_label,
                text=separator.join(source_texts),
                source_segment_ids=source_ids,
                source_texts=source_texts,
                visual_event_ids=left.visual_event_ids + block.visual_event_ids,
                block_kind="speech",
                residual=False,
                uncertainty_items=[],
            )
        else:
            grouped.append(block)
    return grouped


def _visual_time(event: Any) -> tuple[int | None, int | None]:
    start = _get(event, "start_ms", _get(event, "timestamp_ms", _get(event, "actual_time_ms")))
    end = _get(event, "end_ms", start)
    return (int(start) if start is not None else None, int(end) if end is not None else None)


def _visual_id(event: Any, index: int) -> str:
    value = _get(event, "event_id", _get(event, "visual_event_id", _get(event, "id")))
    if value:
        return str(value)
    start, end = _visual_time(event)
    payload = json.dumps([index, start, end], separators=(",", ":")).encode("utf-8")
    return f"visual_{sha256(payload).hexdigest()[:16]}"


def _event_outside_speech(event: Any, blocks: Sequence[ScriptBlockData]) -> bool:
    start, end = _visual_time(event)
    if start is None:
        return True
    comparison_end = end if end is not None and end > start else start + 1
    return not any(
        block.start_ms is not None
        and block.end_ms is not None
        and block.start_ms < comparison_end
        and block.end_ms > start
        for block in blocks
    )


def visual_only_blocks(
    events: Sequence[Any], speech_blocks: Sequence[ScriptBlockData]
) -> list[ScriptBlockData]:
    """Emit supported important visual events that occur outside speech."""

    output: list[ScriptBlockData] = []
    for index, event in enumerate(events):
        if not bool(_get(event, "important", _get(event, "substantive", True))):
            continue
        if not _event_outside_speech(event, speech_blocks):
            continue
        visual_id = _visual_id(event, index)
        description = _get(
            event, "description", _get(event, "factual_grounded_description", _get(event, "text"))
        )
        if not description:
            description = "[visual evidence retained; semantic description pending review]"
        start, end = _visual_time(event)
        output.append(
            ScriptBlockData(
                block_id=_block_id([visual_id], "visual_only"),
                start_ms=start,
                end_ms=end,
                speaker_label=None,
                text=str(description),
                source_segment_ids=[],
                source_texts=[],
                visual_event_ids=[visual_id],
                block_kind="visual_only",
                uncertainty_items=list(_get(event, "uncertainty_items", []) or []),
            )
        )
    return output


def _block_sort_key(block: ScriptBlockData, index: int) -> tuple[int, int, int, int]:
    return (
        1 if block.start_ms is None else 0,
        block.start_ms or 0,
        0 if block.block_kind == "speech" else 1,
        index,
    )


def build_lossless_blocks(
    segments: Sequence[Any],
    *,
    visual_events: Sequence[Any] = (),
    group_adjacent: bool = True,
    max_group_gap_ms: int = 1_500,
    separator: str = "\n\n",
) -> list[ScriptBlockData]:
    """Build script blocks and immediately verify exact segment mappings."""

    provisional = provisional_blocks(segments)
    speech = (
        group_lossless_blocks(provisional, max_gap_ms=max_group_gap_ms, separator=separator)
        if group_adjacent
        else provisional
    )
    combined = speech + visual_only_blocks(visual_events, speech)
    indexed = list(enumerate(combined))
    indexed.sort(key=lambda pair: _block_sort_key(pair[1], pair[0]))
    output = [block for _, block in indexed]
    coverage = audit_block_coverage(segments, output)
    if (
        coverage.missing_segment_ids
        or coverage.duplicated_segment_ids
        or coverage.partial_segment_ids
    ):
        # Preserve anything not provably mapped in an explicit residual block.
        mapped = {source_id for block in output for source_id in block.source_segment_ids}
        for index, segment in enumerate(segments):
            source_id = _segment_id(segment, index)
            if bool(_get(segment, "substantive", True)) and source_id not in mapped:
                text = segment_text(segment)
                output.append(
                    ScriptBlockData(
                        block_id=_block_id([source_id], "residual"),
                        start_ms=_get(segment, "start_ms"),
                        end_ms=_get(segment, "end_ms"),
                        speaker_label=_get(segment, "speaker_label"),
                        text=text,
                        source_segment_ids=[source_id],
                        source_texts=[text],
                        block_kind="residual",
                        residual=True,
                        uncertainty_items=["automatic_grouping_mapping_failed"],
                    )
                )
        output = [
            block
            for _, block in sorted(
                enumerate(output), key=lambda pair: _block_sort_key(pair[1], pair[0])
            )
        ]
    LOGGER.info(
        "built_lossless_blocks",
        extra={"source_segment_count": len(segments), "block_count": len(output)},
    )
    return output


_TOKEN_RE = re.compile(
    r"https?://[^\s]+|[A-Za-z]:[\\/][^\s]+|(?:^|(?<=\s))/(?:[^\s]+)|"
    r"--?[A-Za-z][\w-]*|[^\W_]+(?:['’][^\W_]+)*(?:[.,:]\d+)*(?:%|mg|kg|cm|mm|ml|gb|mb|kb)?|[^\w\s]",
    re.UNICODE | re.I,
)


def ordered_tokens(
    text: str, *, case_sensitive: bool = False, include_punctuation: bool = False
) -> list[str]:
    """Tokenize in source order, preserving commands, URLs, paths, and numbers."""

    tokens = _TOKEN_RE.findall(text)
    if not include_punctuation:
        tokens = [
            token
            for token in tokens
            if any(character.isalnum() for character in token) or token.startswith(("/", "-"))
        ]
    return tokens if case_sensitive else [token.casefold() for token in tokens]


def audit_ordered_tokens(
    source_text: str,
    output_text: str,
    *,
    case_sensitive: bool = False,
    include_punctuation: bool = False,
) -> OrderedTokenAudit:
    """Sequence-aware audit that detects omissions, additions, and reordering."""

    source = ordered_tokens(
        source_text, case_sensitive=case_sensitive, include_punctuation=include_punctuation
    )
    output = ordered_tokens(
        output_text, case_sensitive=case_sensitive, include_punctuation=include_punctuation
    )
    matcher = SequenceMatcher(a=source, b=output, autojunk=False)
    missing: list[str] = []
    added: list[str] = []
    substitutions: list[tuple[str, str]] = []
    opcodes = tuple(matcher.get_opcodes())
    for tag, left_start, left_end, right_start, right_end in opcodes:
        if tag in {"delete", "replace"}:
            missing.extend(source[left_start:left_end])
        if tag in {"insert", "replace"}:
            added.extend(output[right_start:right_end])
        if tag == "replace":
            substitutions.extend(
                zip(source[left_start:left_end], output[right_start:right_end], strict=False)
            )
    reordered = source != output and Counter(source) == Counter(output)
    faithful = source == output
    return OrderedTokenAudit(
        faithful=faithful,
        source_tokens=tuple(source),
        output_tokens=tuple(output),
        missing_tokens=tuple(missing),
        added_tokens=tuple(added),
        substitutions=tuple(substitutions),
        reordered=reordered,
        opcodes=opcodes,
        matched_ratio=round(matcher.ratio(), 6),
    )


_HIGH_IMPACT_RE = re.compile(
    r"https?://[^\s]+|\b[^\s@]+@[^\s@]+\.[^\s@]+\b|"
    r"(?:[A-Za-z]:[\\/]|/)[^\s]+|--?[A-Za-z][\w-]*|"
    r"\b\d+(?:[.,:]\d+)*(?:%|mg|kg|cm|mm|ml|gb|mb|kb)?\b",
    re.I,
)


def high_impact_tokens(text: str) -> list[str]:
    return [match.group(0).rstrip(".,;!?") for match in _HIGH_IMPACT_RE.finditer(text)]


def audit_high_impact_tokens(source_text: str, output_text: str) -> HighImpactAudit:
    source = high_impact_tokens(source_text)
    output = high_impact_tokens(output_text)
    source_counts, output_counts = Counter(source), Counter(output)
    missing = list((source_counts - output_counts).elements())
    added = list((output_counts - source_counts).elements())
    reordered = not missing and not added and source != output
    return HighImpactAudit(
        not missing and not added and not reordered,
        tuple(source),
        tuple(output),
        tuple(missing),
        tuple(added),
        reordered,
    )


def audit_block_coverage(segments: Sequence[Any], blocks: Sequence[Any]) -> CoverageAudit:
    """Prove substantive segment mapping and overall ordered-token coverage."""

    source_by_id = {
        _segment_id(segment, index): segment_text(segment)
        for index, segment in enumerate(segments)
        if bool(_get(segment, "substantive", True))
    }
    occurrences: Counter[str] = Counter()
    mapped_texts: dict[str, list[str]] = {}
    residual: list[str] = []
    speech_block_texts: list[str] = []
    for block in blocks:
        source_ids = list(
            _get(block, "source_segment_ids", _get(block, "transcript_segment_ids", [])) or []
        )
        source_texts = list(_get(block, "source_texts", []) or [])
        for position, source_id in enumerate(source_ids):
            occurrences[str(source_id)] += 1
            if position < len(source_texts):
                mapped_texts.setdefault(str(source_id), []).append(str(source_texts[position]))
        if source_ids:
            speech_block_texts.append(str(_get(block, "text", _get(block, "spoken_text", ""))))
        if bool(_get(block, "residual", False)):
            residual.append(str(_get(block, "text", _get(block, "residual_source_text", ""))))
    missing = [source_id for source_id in source_by_id if not occurrences[source_id]]
    duplicated = [source_id for source_id in source_by_id if occurrences[source_id] > 1]
    partial = [
        source_id
        for source_id, text in source_by_id.items()
        if occurrences[source_id] and text not in mapped_texts.get(source_id, [])
    ]
    source_text = "\n\n".join(source_by_id.values())
    output_text = "\n\n".join(speech_block_texts)
    ordered_audit = audit_ordered_tokens(source_text, output_text)
    return CoverageAudit(
        valid=not missing and not duplicated and not partial and ordered_audit.faithful,
        total_substantive_segments=len(source_by_id),
        covered_segments=len(source_by_id) - len(missing),
        missing_segment_ids=missing,
        duplicated_segment_ids=duplicated,
        partial_segment_ids=partial,
        residual_text=residual,
        ordered_token_audit=ordered_audit,
    )


def reconstruct_verbatim(segments: Sequence[Any], **kwargs: Any) -> list[ScriptBlockData]:
    """Public strict-default reconstruction entry point."""

    return build_lossless_blocks(segments, **kwargs)


# Convenient public alias without claiming this is the canonical Pydantic type.
ScriptBlock = ScriptBlockData


__all__ = [
    "CoverageAudit",
    "HighImpactAudit",
    "OrderedTokenAudit",
    "ReconstructionError",
    "ScriptBlock",
    "ScriptBlockData",
    "audit_block_coverage",
    "audit_high_impact_tokens",
    "audit_ordered_tokens",
    "build_lossless_blocks",
    "group_lossless_blocks",
    "high_impact_tokens",
    "ordered_tokens",
    "provisional_blocks",
    "reconstruct_verbatim",
    "segment_text",
    "visual_only_blocks",
]
