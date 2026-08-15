"""Lightweight, identity-safe speaker-turn handling."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from hashlib import sha256
from typing import Any

LOGGER = logging.getLogger(__name__)


class DiarizationError(ValueError):
    """Raised for invalid diarization ranges or unsafe mappings."""


@dataclass(slots=True)
class SpeakerTurn:
    turn_id: str
    start_ms: int
    end_ms: int
    speaker_label: str
    backend_label: str | None = None
    confidence: float | None = None
    boundary_uncertain: bool = False
    overlaps_turn_ids: list[str] = field(default_factory=list)
    mapping_evidence: str | None = None

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpeakerIdentityEvidence:
    """A transcript-local, explicitly spoken speaker-name claim."""

    name: str
    segment_id: str | None
    start_ms: int | None
    end_ms: int | None
    quote: str
    pattern: str
    evidence_type: str = "explicit_self_identification"

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return asdict(self)


_IDENTITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("my_name_is", re.compile(r"\bmy\s+name\s+is\s+", re.I)),
    (
        "ang_pangalan_ko_ay",
        # ``si`` is optional in natural Filipino introductions (both
        # ``Ang pangalan ko ay Maria`` and ``... ay si Maria`` are common).
        re.compile(r"\bang\s+pangalan\s+ko\s+(?:ay|y)\s+(?:si\s+)?", re.I),
    ),
    ("ako_si", re.compile(r"\bako(?:\s+po)?\s+(?:ay\s+)?si\s+", re.I)),
    ("ako_ay", re.compile(r"\bako(?:\s+po)?\s+ay\s+", re.I)),
    ("i_am", re.compile(r"\bi(?:\s+am|'m)\s+", re.I)),
    ("this_is", re.compile(r"\bthis\s+is\s+", re.I)),
)
_REPORTED_PREFIX = re.compile(
    r"(?:said|says|told|asked|claimed|introduced|called)\s*[,:'\-]*$", re.I
)
_ROLE_ONLY = {
    "coach", "teacher", "host", "dispatcher", "driver", "agent", "trainer", "student"
}
_NAME_STOP = {
    "a", "an", "and", "at", "ay", "because", "but", "by", "from", "for", "here", "if", "in", "my", "na", "ng", "now", "of", "on", "po", "right", "si", "speaking", "that", "the", "to", "today", "who", "with", "your"
}
_NON_NAME_START = {
    "afraid", "already", "also", "available", "calling", "checking", "closer", "doing", "expecting", "fighting", "giving", "going", "genuinely", "having", "here", "just", "looking", "making", "meeting", "nasa", "not", "one", "proudly", "really", "receiving", "requesting", "seeing", "sending", "sharing", "showing", "so", "sorry", "telling", "targeting", "there", "using", "very", "willing"
}
_HYPOTHETICAL_IDENTITY_CONTEXT = re.compile(
    r"(?:for\s+example|let(?:'|’)s\s+say|use\s+(?:an?\s+)?(?:american|sample|fake|example)\s+name|hypothetical|role[- ]play|kunwari|sample\s+name)",
    re.I,
)


def _segment_text(record: Any) -> str:
    for key in ("raw_text", "human_verified_text", "normalized_text", "text"):
        value = _get(record, key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _identity_name(text: str, start: int) -> str | None:
    tail = text[start:]
    tail = re.split(r"[\n\r.!?,;:()\[\]{}\-–—]", tail, maxsplit=1)[0]
    tokens = tail.strip().split()
    chosen: list[str] = []
    for token in tokens[:4]:
        clean = re.sub(r"[^\wÀ-ÿ'’]", "", token).strip("'’")
        if not clean or clean.casefold() in _NAME_STOP:
            break
        chosen.append(clean)
    if not chosen:
        return None
    name = " ".join(chosen).strip()
    if chosen[0].casefold() in _NON_NAME_START:
        return None
    if len(chosen) == 1 and name.casefold() in _ROLE_ONLY:
        return None
    if name.casefold() in {"freight course", "freight course 101", "freight 101"}:
        return None
    return name


def extract_explicit_self_identifications(
    segments: Sequence[Any], *, include_introduction_pattern: bool = False
) -> list[SpeakerIdentityEvidence]:
    """Extract only direct, transcript-contained self-identification claims."""

    results: list[SpeakerIdentityEvidence] = []
    seen: set[tuple[str, str | None]] = set()
    for segment in segments:
        text = _segment_text(segment)
        if not text:
            continue
        segment_id_value = _get(segment, "segment_id", _get(segment, "id"))
        segment_id = str(segment_id_value) if segment_id_value else None
        for pattern, matcher in _IDENTITY_PATTERNS:
            if pattern == "this_is" and not include_introduction_pattern:
                continue
            for match in matcher.finditer(text):
                prefix = text[: match.start()].rstrip()
                if prefix and (_REPORTED_PREFIX.search(prefix) or prefix.endswith(('"', "'", "“", "‘"))):
                    continue
                if _HYPOTHETICAL_IDENTITY_CONTEXT.search(prefix[-180:]):
                    continue
                name = _identity_name(text, match.end())
                if name is None:
                    continue
                key = (name.casefold(), segment_id)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    SpeakerIdentityEvidence(
                        name=name,
                        segment_id=segment_id,
                        start_ms=_get(segment, "start_ms"),
                        end_ms=_get(segment, "end_ms"),
                        quote=text[match.start() : match.end() + len(name)].strip(),
                        pattern=pattern,
                    )
                )
    return results


def apply_explicit_identity_evidence(
    segments: Sequence[Any],
    evidence: Sequence[SpeakerIdentityEvidence] | None = None,
    *,
    include_introduction_pattern: bool = False,
) -> tuple[list[Any], list[SpeakerIdentityEvidence]]:
    """Apply non-conflicting explicit names only to their source segments."""

    claims = list(evidence) if evidence is not None else extract_explicit_self_identifications(
        segments, include_introduction_pattern=include_introduction_pattern
    )
    by_segment: dict[str, list[SpeakerIdentityEvidence]] = {}
    for claim in claims:
        if claim.segment_id:
            by_segment.setdefault(claim.segment_id, []).append(claim)
    output: list[Any] = []
    for segment in segments:
        segment_id = _get(segment, "segment_id", _get(segment, "id"))
        candidates = by_segment.get(str(segment_id), []) if segment_id else []
        names = {claim.name for claim in candidates}
        if len(names) != 1:
            if candidates:
                uncertainty_items = list(_get(segment, "uncertainty_items", []) or [])
                if "uncertain_speaker_identity" not in uncertainty_items:
                    uncertainty_items.append("uncertain_speaker_identity")
                output.append(_copy_with(segment, uncertainty_items=uncertainty_items))
                continue
            output.append(segment)
            continue
        current = _get(segment, "speaker_label")
        if current and not str(current).casefold().startswith("speaker "):
            output.append(segment)
            continue
        output.append(_copy_with(segment, speaker_label=next(iter(names))))
    return output, claims


def repair_automatic_identity_labels(
    segments: Sequence[Any],
) -> tuple[list[Any], list[dict[str, str]]]:
    """Remove stale automatic identity labels unsupported by current evidence.

    This migration is deliberately limited to automatically transcribed
    segments.  Neutral ``Speaker N`` labels and human/manual labels are left
    untouched; an exact current self-identification claim is retained on its
    source segment only.
    """

    claims = extract_explicit_self_identifications(segments)
    supported = {
        str(claim.segment_id): claim.name
        for claim in claims
        if claim.segment_id
    }
    output: list[Any] = []
    corrections: list[dict[str, str]] = []
    for segment in segments:
        current = _get(segment, "speaker_label")
        status = str(_get(segment, "verification_status", ""))
        segment_id = _get(segment, "segment_id", _get(segment, "id"))
        if (
            current
            and str(current).casefold() not in {"speaker 1", "speaker 2", "speaker 3"}
            and status == "automatically_transcribed"
            and str(segment_id) not in supported
        ):
            output.append(_copy_with(segment, speaker_label=None))
            corrections.append(
                {
                    "segment_id": str(segment_id),
                    "old_label": str(current),
                    "reason": "automatic_identity_label_lacks_current_exact_self_identification",
                }
            )
        else:
            output.append(segment)
    return output, corrections


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _copy_with(record: Any, **updates: Any) -> Any:
    if isinstance(record, Mapping):
        copied: MutableMapping[str, Any] = dict(record)
        copied.update(updates)
        return copied
    if hasattr(record, "model_copy"):
        return record.model_copy(update=updates)
    if is_dataclass(record):
        return replace(record, **updates)  # type: ignore[type-var]
    copied = record.__class__.__new__(record.__class__)
    copied.__dict__.update(record.__dict__)
    copied.__dict__.update(updates)
    return copied


def _label(record: Any) -> str:
    value = _get(record, "backend_label", _get(record, "speaker_label", _get(record, "speaker")))
    return str(value) if value not in (None, "") else "unknown"


def neutralize_speaker_labels(
    turns: Sequence[Any],
    *,
    manual_mapping: Mapping[str, str] | None = None,
    mapping_evidence: Mapping[str, str] | None = None,
) -> tuple[list[SpeakerTurn], dict[str, str]]:
    """Convert backend labels to neutral labels in first-appearance order.

    User/manual names are accepted only when accompanied by non-empty evidence.
    Backend labels are always retained separately for auditability.
    """

    manual = dict(manual_mapping or {})
    evidence = dict(mapping_evidence or {})
    for backend, name in manual.items():
        if not name.strip():
            raise DiarizationError(f"Manual speaker mapping for {backend!r} is empty")
        if not evidence.get(backend, "").strip():
            raise DiarizationError(f"Manual speaker mapping for {backend!r} requires evidence")
    indexed = list(enumerate(turns))
    indexed.sort(
        key=lambda item: (
            int(_get(item[1], "start_ms", 0)),
            int(_get(item[1], "end_ms", 0)),
            item[0],
        )
    )
    neutral: dict[str, str] = {}
    output: list[SpeakerTurn] = []
    for original_index, turn in indexed:
        start = int(_get(turn, "start_ms", 0))
        end = int(_get(turn, "end_ms", 0))
        if start < 0 or end <= start:
            raise DiarizationError(f"Invalid diarization range at input index {original_index}")
        backend = _label(turn)
        if backend not in neutral:
            neutral[backend] = f"Speaker {len(neutral) + 1}"
        display = manual.get(backend, neutral[backend])
        payload = json.dumps([backend, start, end, original_index], separators=(",", ":")).encode(
            "utf-8"
        )
        output.append(
            SpeakerTurn(
                turn_id=str(_get(turn, "turn_id", f"turn_{sha256(payload).hexdigest()[:16]}")),
                start_ms=start,
                end_ms=end,
                speaker_label=display,
                backend_label=backend,
                confidence=float(_get(turn, "confidence"))
                if _get(turn, "confidence") is not None
                else None,
                boundary_uncertain=bool(_get(turn, "boundary_uncertain", False)),
                mapping_evidence=evidence.get(backend)
                if backend in manual
                else "neutralized backend label",
            )
        )
    for index, turn in enumerate(output):
        turn.overlaps_turn_ids = [
            other.turn_id
            for other_index, other in enumerate(output)
            if other_index != index
            and other.start_ms < turn.end_ms
            and other.end_ms > turn.start_ms
        ]
    return output, {backend: manual.get(backend, label) for backend, label in neutral.items()}


def assign_speakers_to_segments(
    segments: Sequence[Any],
    turns: Sequence[Any],
    *,
    boundary_tolerance_ms: int = 120,
) -> list[Any]:
    """Assign the greatest-overlap neutral turn while marking ambiguous ties."""

    output: list[Any] = []
    for segment in segments:
        start, end = _get(segment, "start_ms"), _get(segment, "end_ms")
        if start is None or end is None:
            output.append(segment)
            continue
        overlaps: list[tuple[int, int, Any]] = []
        for order, turn in enumerate(turns):
            turn_start, turn_end = int(_get(turn, "start_ms")), int(_get(turn, "end_ms"))
            overlap = max(0, min(int(end), turn_end) - max(int(start), turn_start))
            if overlap:
                overlaps.append((overlap, -order, turn))
        if not overlaps:
            output.append(segment)
            continue
        overlaps.sort(key=lambda item: (-item[0], -item[1]))
        best_overlap, _, best = overlaps[0]
        uncertainties = list(_get(segment, "uncertainty_items", []) or [])
        ambiguous = (
            len(overlaps) > 1 and abs(best_overlap - overlaps[1][0]) <= boundary_tolerance_ms
        )
        if ambiguous or bool(_get(best, "boundary_uncertain", False)):
            if "uncertain_speaker_boundary" not in uncertainties:
                uncertainties.append("uncertain_speaker_boundary")
        output.append(
            _copy_with(
                segment,
                speaker_label=str(_get(best, "speaker_label", "Speaker 1")),
                uncertainty_items=uncertainties,
            )
        )
    return output


def apply_diarization(
    segments: Sequence[Any],
    turns: Sequence[Any],
    *,
    manual_mapping: Mapping[str, str] | None = None,
    mapping_evidence: Mapping[str, str] | None = None,
    boundary_tolerance_ms: int = 120,
) -> tuple[list[Any], list[SpeakerTurn], dict[str, str]]:
    """Neutralize backend output, then attach turns to transcript segments."""

    neutral_turns, mapping = neutralize_speaker_labels(
        turns,
        manual_mapping=manual_mapping,
        mapping_evidence=mapping_evidence,
    )
    assigned = assign_speakers_to_segments(
        segments,
        neutral_turns,
        boundary_tolerance_ms=boundary_tolerance_ms,
    )
    LOGGER.info(
        "applied_diarization",
        extra={
            "turn_count": len(neutral_turns),
            "speaker_count": len(mapping),
            "segment_count": len(segments),
        },
    )
    return assigned, neutral_turns, mapping


neutralize_speakers = neutralize_speaker_labels


__all__ = [
    "DiarizationError",
    "SpeakerIdentityEvidence",
    "SpeakerTurn",
    "apply_diarization",
    "assign_speakers_to_segments",
    "neutralize_speaker_labels",
    "neutralize_speakers",
    "extract_explicit_self_identifications",
    "apply_explicit_identity_evidence",
]
