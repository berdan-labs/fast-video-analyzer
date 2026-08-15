from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, cast

from .errors import BlockedError, InputError, ValidationFailure

if TYPE_CHECKING:
    from .schemas import OCRObservation as CanonicalOCRObservation


_TESSERACT_VERSION_CACHE: dict[str, tuple[tuple[int, int, int, int], str]] = {}
_TESSERACT_VERSION_LOCK = Lock()


def _executable_signature(path: str) -> tuple[int, int, int, int] | None:
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(getattr(stat, "st_ctime_ns", 0)),
        int(getattr(stat, "st_ino", 0)),
    )


@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float | None
    bounding_box: tuple[int, int, int, int]
    page: int
    block: int
    paragraph: int
    line: int
    word: int


@dataclass(frozen=True)
class OCRObservation:
    observation_id: str
    frame_id: str
    crop_id: str | None
    bounding_region: tuple[int, int, int, int] | None
    raw_engine_text: str
    normalized_interpretation: str
    confidence: float | None
    alternatives: tuple[Mapping[str, object], ...]
    language: str | None
    uncertain_characters: tuple[Mapping[str, object], ...]
    engine: str
    engine_version: str
    human_decision: str | None
    tokens: tuple[OCRToken, ...]


@dataclass(frozen=True)
class OCRRunResult:
    status: str
    observations: tuple[OCRObservation, ...]
    reason: str | None = None


def serialize_observation(observation: OCRObservation) -> dict[str, object]:
    """Serialize the complete local OCR result for a resumable checkpoint.

    The canonical project intentionally stores a compact schema observation, while
    the checkpoint keeps token boxes and engine uncertainty so a resumed run is
    byte-for-byte equivalent to the cold path.  This function is deliberately
    explicit instead of relying on ``dataclasses.asdict`` so the on-disk format is
    stable and easy to validate before use.
    """

    return {
        "observation_id": observation.observation_id,
        "frame_id": observation.frame_id,
        "crop_id": observation.crop_id,
        "bounding_region": list(observation.bounding_region)
        if observation.bounding_region is not None
        else None,
        "raw_engine_text": observation.raw_engine_text,
        "normalized_interpretation": observation.normalized_interpretation,
        "confidence": observation.confidence,
        "alternatives": [dict(item) for item in observation.alternatives],
        "language": observation.language,
        "uncertain_characters": [dict(item) for item in observation.uncertain_characters],
        "engine": observation.engine,
        "engine_version": observation.engine_version,
        "human_decision": observation.human_decision,
        "tokens": [
            {
                "text": token.text,
                "confidence": token.confidence,
                "bounding_box": list(token.bounding_box),
                "page": token.page,
                "block": token.block,
                "paragraph": token.paragraph,
                "line": token.line,
                "word": token.word,
            }
            for token in observation.tokens
        ],
    }


def deserialize_observation(payload: Mapping[str, object]) -> OCRObservation:
    """Restore a validated checkpoint observation.

    Checkpoint files are treated as untrusted state: malformed values produce a
    ``ValidationFailure`` and the caller falls back to OCR instead of allowing a
    corrupt cache to affect evidence selection.
    """

    def required_text(name: str) -> str:
        value = payload.get(name)
        if not isinstance(value, str):
            raise ValidationFailure(f"OCR checkpoint field {name!r} must be text")
        return value

    def optional_text(name: str) -> str | None:
        value = payload.get(name)
        if value is not None and not isinstance(value, str):
            raise ValidationFailure(f"OCR checkpoint field {name!r} must be text or null")
        return value

    raw_region = payload.get("bounding_region")
    if raw_region is None:
        region: tuple[int, int, int, int] | None = None
    elif (
        isinstance(raw_region, (list, tuple))
        and len(raw_region) == 4
        and all(isinstance(value, int) and not isinstance(value, bool) for value in raw_region)
    ):
        region = (
            int(raw_region[0]),
            int(raw_region[1]),
            int(raw_region[2]),
            int(raw_region[3]),
        )
    else:
        raise ValidationFailure("OCR checkpoint bounding_region is invalid")

    raw_alternatives = payload.get("alternatives", [])
    if not isinstance(raw_alternatives, list) or not all(
        isinstance(value, Mapping) for value in raw_alternatives
    ):
        raise ValidationFailure("OCR checkpoint alternatives are invalid")
    raw_uncertain = payload.get("uncertain_characters", [])
    if not isinstance(raw_uncertain, list) or not all(
        isinstance(value, Mapping) for value in raw_uncertain
    ):
        raise ValidationFailure("OCR checkpoint uncertain_characters are invalid")
    raw_tokens = payload.get("tokens", [])
    if not isinstance(raw_tokens, list):
        raise ValidationFailure("OCR checkpoint tokens are invalid")
    tokens: list[OCRToken] = []
    for raw_token in raw_tokens:
        if not isinstance(raw_token, Mapping):
            raise ValidationFailure("OCR checkpoint token is invalid")
        token_text = raw_token.get("text")
        raw_box = raw_token.get("bounding_box")
        if not isinstance(token_text, str):
            raise ValidationFailure("OCR checkpoint token text is invalid")
        if not (
            isinstance(raw_box, (list, tuple))
            and len(raw_box) == 4
            and all(isinstance(value, int) and not isinstance(value, bool) for value in raw_box)
        ):
            raise ValidationFailure("OCR checkpoint token bounding_box is invalid")
        page = raw_token.get("page")
        block = raw_token.get("block")
        paragraph = raw_token.get("paragraph")
        line = raw_token.get("line")
        word = raw_token.get("word")
        values = (page, block, paragraph, line, word)
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise ValidationFailure("OCR checkpoint token coordinates are invalid")
        typed_values = (
            cast(int, page),
            cast(int, block),
            cast(int, paragraph),
            cast(int, line),
            cast(int, word),
        )
        confidence = raw_token.get("confidence")
        if confidence is not None and not isinstance(confidence, (int, float)):
            raise ValidationFailure("OCR checkpoint token confidence is invalid")
        tokens.append(
            OCRToken(
                text=token_text,
                confidence=float(confidence) if confidence is not None else None,
                bounding_box=(
                    int(raw_box[0]),
                    int(raw_box[1]),
                    int(raw_box[2]),
                    int(raw_box[3]),
                ),
                page=typed_values[0],
                block=typed_values[1],
                paragraph=typed_values[2],
                line=typed_values[3],
                word=typed_values[4],
            )
        )

    confidence = payload.get("confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        raise ValidationFailure("OCR checkpoint confidence is invalid")
    return OCRObservation(
        observation_id=required_text("observation_id"),
        frame_id=required_text("frame_id"),
        crop_id=optional_text("crop_id"),
        bounding_region=region,
        raw_engine_text=required_text("raw_engine_text"),
        normalized_interpretation=required_text("normalized_interpretation"),
        confidence=float(confidence) if confidence is not None else None,
        alternatives=tuple(dict(value) for value in raw_alternatives),
        language=optional_text("language"),
        uncertain_characters=tuple(dict(value) for value in raw_uncertain),
        engine=required_text("engine"),
        engine_version=required_text("engine_version"),
        human_decision=optional_text("human_decision"),
        tokens=tuple(tokens),
    )


class OCRAdapter(ABC):
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def recognize(
        self,
        image_path: str | Path,
        *,
        frame_id: str,
        observation_id: str,
        crop_id: str | None = None,
        language: str | None = None,
    ) -> OCRObservation:
        raise NotImplementedError


def normalize_ocr_text(raw: str) -> str:
    """Normalize representation only; ambiguous characters are never corrected."""
    normalized = unicodedata.normalize("NFC", raw).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def parse_tesseract_tsv(
    payload: str,
    *,
    observation_id: str,
    frame_id: str,
    crop_id: str | None,
    language: str | None,
    engine_version: str,
    uncertainty_threshold: float = 80.0,
) -> OCRObservation:
    # Tesseract emits literal quote characters in the text column (for
    # example, a quoted name).  The default CSV quote handling treats an
    # unescaped quote as the start of a multiline field and can swallow every
    # following TSV row into one OCR token.  TSV has no quoting contract here;
    # treat quotes as ordinary evidence characters so row boundaries remain
    # deterministic and token geometry is not corrupted.
    reader = csv.DictReader(io.StringIO(payload), delimiter="\t", quoting=csv.QUOTE_NONE)
    required = {
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "word_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValidationFailure("Tesseract TSV output is missing required columns")
    tokens: list[OCRToken] = []
    uncertain: list[Mapping[str, object]] = []
    line_groups: dict[tuple[int, int, int, int], list[str]] = {}
    for row in reader:
        text = row.get("text") or ""
        if not text.strip():
            continue
        try:
            raw_confidence = float(row.get("conf") or -1)
            confidence = raw_confidence if raw_confidence >= 0 else None
            box = (int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"]))
            page, block, paragraph, line, word = (
                int(row["page_num"]),
                int(row["block_num"]),
                int(row["par_num"]),
                int(row["line_num"]),
                int(row["word_num"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ValidationFailure("Tesseract TSV contains an invalid numeric field") from exc
        token = OCRToken(text, confidence, box, page, block, paragraph, line, word)
        tokens.append(token)
        line_groups.setdefault((page, block, paragraph, line), []).append(text)
        if confidence is None or confidence < uncertainty_threshold:
            uncertain.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bounding_box": list(box),
                    "reason": "engine confidence below threshold",
                }
            )
    raw_text = "\n".join(" ".join(words) for _, words in sorted(line_groups.items()))
    confidences = [token.confidence for token in tokens if token.confidence is not None]
    confidence = sum(confidences) / len(confidences) / 100 if confidences else None
    if tokens:
        left = min(token.bounding_box[0] for token in tokens)
        top = min(token.bounding_box[1] for token in tokens)
        right = max(token.bounding_box[0] + token.bounding_box[2] for token in tokens)
        bottom = max(token.bounding_box[1] + token.bounding_box[3] for token in tokens)
        region: tuple[int, int, int, int] | None = (left, top, right - left, bottom - top)
    else:
        region = None
    return OCRObservation(
        observation_id=observation_id,
        frame_id=frame_id,
        crop_id=crop_id,
        bounding_region=region,
        raw_engine_text=raw_text,
        normalized_interpretation=normalize_ocr_text(raw_text),
        confidence=confidence,
        alternatives=(),
        language=language,
        uncertain_characters=tuple(uncertain),
        engine="tesseract",
        engine_version=engine_version,
        human_decision=None,
        tokens=tuple(tokens),
    )


class TesseractOCRAdapter(OCRAdapter):
    def __init__(
        self,
        *,
        executable: str = "tesseract",
        default_language: str | None = None,
        page_segmentation_mode: int = 3,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not 0 <= page_segmentation_mode <= 13:
            raise InputError("Tesseract page segmentation mode must be between 0 and 13")
        configured = os.environ.get("VSR_TESSERACT_PATH") if executable == "tesseract" else None
        candidates = [
            configured,
            shutil.which(executable),
            executable if Path(executable).is_file() else None,
        ]
        if os.name == "nt" and executable == "tesseract":
            candidates.extend(
                [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                ]
            )
        self.executable = next(
            (str(Path(value)) for value in candidates if value and Path(value).is_file()),
            executable,
        )
        self.default_language = default_language
        self.page_segmentation_mode = page_segmentation_mode
        self.timeout_seconds = timeout_seconds
        self._version: str | None = None
        self._version_lock = Lock()

    def available(self) -> bool:
        return shutil.which(self.executable) is not None or Path(self.executable).is_file()

    def version(self) -> str:
        if self._version is not None:
            return self._version
        with self._version_lock:
            if self._version is not None:
                return self._version
            if not self.available():
                raise BlockedError(f"Tesseract executable was not found: {self.executable}")
            signature = _executable_signature(self.executable)
            with _TESSERACT_VERSION_LOCK:
                cached = _TESSERACT_VERSION_CACHE.get(self.executable)
                if signature is not None and cached is not None and cached[0] == signature:
                    self._version = cached[1]
                else:
                    try:
                        completed = subprocess.run(
                            [self.executable, "--version"],
                            check=False,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=10,
                            shell=False,
                        )
                    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                        raise BlockedError(
                            f"Unable to execute Tesseract: {self.executable}"
                        ) from exc
                    if completed.returncode != 0:
                        raise ValidationFailure("Tesseract version check failed")
                    self._version = completed.stdout.splitlines()[0].strip() or "unknown"
                    if signature is not None:
                        _TESSERACT_VERSION_CACHE[self.executable] = (signature, self._version)
        return self._version

    def recognize(
        self,
        image_path: str | Path,
        *,
        frame_id: str,
        observation_id: str,
        crop_id: str | None = None,
        language: str | None = None,
    ) -> OCRObservation:
        image = Path(image_path).expanduser()
        if not image.is_file():
            raise InputError(f"OCR image does not exist: {image}")
        selected_language = language or self.default_language
        command = [self.executable, str(image), "stdout"]
        if selected_language:
            command.extend(["-l", selected_language])
        command.extend(["--psm", str(self.page_segmentation_mode), "tsv"])
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise BlockedError(f"Tesseract executable was not found: {self.executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise BlockedError(f"OCR exceeded the {self.timeout_seconds:g}s timeout") from exc
        if completed.returncode != 0:
            detail = re.sub(r"\s+", " ", completed.stderr).strip()[-1000:]
            raise ValidationFailure(
                f"Tesseract failed with exit code {completed.returncode}: {detail}"
            )
        return parse_tesseract_tsv(
            completed.stdout,
            observation_id=observation_id,
            frame_id=frame_id,
            crop_id=crop_id,
            language=selected_language,
            engine_version=self.version(),
        )


def run_optional_ocr(
    adapter: OCRAdapter | None,
    image_path: str | Path,
    *,
    frame_id: str,
    observation_id: str,
    crop_id: str | None = None,
    language: str | None = None,
) -> OCRRunResult:
    if adapter is None:
        return OCRRunResult("unavailable", (), "No OCR adapter was configured")
    if not adapter.available():
        return OCRRunResult("unavailable", (), "The configured OCR engine is unavailable")
    observation = adapter.recognize(
        image_path,
        frame_id=frame_id,
        observation_id=observation_id,
        crop_id=crop_id,
        language=language,
    )
    return OCRRunResult("completed", (observation,))


def ocr_changed(left: OCRObservation | str | None, right: OCRObservation | str | None) -> bool:
    def text(value: OCRObservation | str | None) -> str:
        if isinstance(value, OCRObservation):
            return value.normalized_interpretation
        return normalize_ocr_text(value or "")

    return text(left) != text(right)


def to_schema_observation(observation: OCRObservation) -> CanonicalOCRObservation:
    from .schemas import OCRObservation as CanonicalOCRObservation

    uncertain = [
        f"{item.get('text', '')} ({item.get('reason', 'uncertain')}; "
        f"confidence={item.get('confidence')})"
        for item in observation.uncertain_characters
    ]
    if observation.bounding_region:
        x, y, width, height = observation.bounding_region
        region: tuple[float, float, float, float] | None = (
            float(x),
            float(y),
            float(width),
            float(height),
        )
    else:
        region = None
    return CanonicalOCRObservation(
        observation_id=observation.observation_id,
        frame_id=observation.frame_id,
        crop_id=observation.crop_id,
        bounding_region=region,
        raw_engine_text=observation.raw_engine_text,
        normalized_interpretation=observation.normalized_interpretation,
        confidence=observation.confidence,
        alternatives=[],
        language=observation.language,
        uncertain_characters=uncertain,
        engine=observation.engine,
        engine_version=observation.engine_version,
        human_decision=observation.human_decision,
    )
