"""Model-independent ASR contract and the production faster-whisper boundary."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from hashlib import sha256
from pathlib import Path
from threading import Event, Thread
from typing import Any

from .ids import deterministic_id
from .security import atomic_write_json, sha256_file
from .subtitle_parse import ParsedTranscriptSegment

LOGGER = logging.getLogger(__name__)

ASRProgressCallback = Callable[[Mapping[str, Any]], None]

# Shared ASR receipts are written once per newly processed chunk.  A full
# directory inventory after every write scales as O(chunks * cache_entries),
# which is surprisingly expensive for long recordings or a warm cache shared
# by several projects.  Keep a conservative process-local size ledger and
# reconcile it periodically; the existing full inventory remains the authority
# whenever the estimated budget is crossed or the ledger reaches its interval.
_ASR_PRUNE_INTERVAL = 32
_ASR_PRUNE_STATE: dict[Path, tuple[int, int, dict[Path, int]]] = {}


class ASRError(RuntimeError):
    """Base error for automatic speech-recognition adapters."""


class ModelDownloadPermissionError(ASRError):
    """Raised when model weights would be downloaded without permission."""


class ASRDependencyError(ASRError):
    """Raised when a requested production ASR backend is unavailable."""


@dataclass(slots=True)
class ASRResult:
    segments: list[Any]
    language: str | None = None
    language_probability: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> Any:
        return self.segments[index]

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return asdict(self)


class ASRAdapter(ABC):
    """Boundary implemented by deterministic CI and real production adapters."""

    is_production: bool = False
    backend_name: str = "abstract"

    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        *,
        interval_start_ms: int | None = None,
        interval_end_ms: int | None = None,
        context_padding_ms: int = 0,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> ASRResult:
        raise NotImplementedError


class ModelIndependentASRAdapter(ASRAdapter):
    """Adapter around a supplied deterministic recognizer for ordinary CI.

    A callable is mandatory; there is deliberately no canned transcript or
    fallback output.  This adapter is never considered proof of Whisper model
    accuracy and must not be selected for a production run.
    """

    backend_name = "model-independent"
    is_production = False

    def __init__(
        self,
        transcribe_fn: Callable[..., ASRResult | Sequence[Any] | Mapping[str, Any]],
        *,
        name: str = "model-independent",
        cache_identity: str | None = None,
    ) -> None:
        if not callable(transcribe_fn):
            raise TypeError("transcribe_fn must be callable")
        self._transcribe_fn = transcribe_fn
        self.backend_name = name
        self.cache_identity = cache_identity or (
            f"{getattr(transcribe_fn, '__module__', 'unknown')}."
            f"{getattr(transcribe_fn, '__qualname__', transcribe_fn.__class__.__qualname__)}"
        )

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        interval_start_ms: int | None = None,
        interval_end_ms: int | None = None,
        context_padding_ms: int = 0,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> ASRResult:
        if (
            interval_start_ms is not None
            and interval_end_ms is not None
            and interval_end_ms <= interval_start_ms
        ):
            raise ASRError("ASR interval end must be after its start")
        result = self._transcribe_fn(
            Path(audio_path),
            interval_start_ms=interval_start_ms,
            interval_end_ms=interval_end_ms,
            context_padding_ms=context_padding_ms,
            language=language,
        )
        normalized = normalize_asr_result(result, source=self.backend_name)
        normalized.metadata.setdefault("backend", self.backend_name)
        normalized.metadata.setdefault("model_independent", True)
        return normalized


def ensure_production_adapter(adapter: ASRAdapter) -> None:
    """Reject mock/model-independent adapters at a production boundary."""

    if not adapter.is_production:
        raise ASRError(f"Adapter {adapter.backend_name!r} is not permitted in production")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_cuda_runtime_failure(error: BaseException) -> bool:
    """Identify errors for which a CPU retry is a safe, explicit fallback."""

    message = str(error).casefold()
    return any(
        marker in message
        for marker in ("cuda", "cublas", "cudnn", "cudart", "gpu runtime")
    )


def _env_bool(name: str, default: bool) -> bool:
    """Read an opt-in decoder toggle without making malformed env fatal."""

    raw = os.environ.get(name, "").strip().casefold()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    LOGGER.warning("Ignoring invalid boolean %s=%r; using %s", name, raw, default)
    return default


class FasterWhisperAdapter(ASRAdapter):
    """Lazy production adapter for ``faster-whisper``.

    Passing a filesystem model path is offline-safe.  Passing a registry model
    name requires explicit ``allow_model_download=True`` so a strict run can
    never start an implicit network/model-weight download.
    """

    backend_name = "faster-whisper"
    is_production = True
    # ``transcribe_checkpointed_chunks`` can safely bypass an intermediate WAV
    # when the requested chunk is the complete media span.  This is deliberately
    # an opt-in capability: other adapters may attach meaning to bounded input
    # (or may not preserve timestamps when given the original container).
    supports_full_media_passthrough = True

    def __init__(
        self,
        model_name_or_path: str | Path = "large-v3",
        *,
        model: str | Path | None = None,
        model_revision: str | None = None,
        model_signature: str | None = None,
        device: str = "auto",
        compute_type: str = "default",
        allow_model_download: bool = False,
        download_root: str | Path | None = None,
        cpu_threads: int = 0,
        num_workers: int = 1,
        decoding_settings: Mapping[str, Any] | None = None,
        inference_mode: str = "standard",
        batch_size: int = 8,
        allow_cpu_fallback: bool = True,
        fallback_compute_type: str = "int8",
    ) -> None:
        normalized_inference_mode = str(inference_mode).strip().casefold()
        if normalized_inference_mode not in {"standard", "batched"}:
            raise ValueError("inference_mode must be 'standard' or 'batched'")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an integer between 1 and 64")
        self.model_name_or_path = str(model if model is not None else model_name_or_path)
        # Explicitly constructed adapters remain backward-compatible when no
        # identity token is supplied. Automatic pipeline resolution supplies a
        # verified manifest revision and/or a lightweight file-stat signature
        # so a shared chunk can never silently survive model replacement at
        # the same filesystem path.
        self.model_revision = str(model_revision) if model_revision else None
        self.model_signature = str(model_signature) if model_signature else None
        self.device = device
        self.compute_type = compute_type
        self.allow_model_download = allow_model_download
        self.download_root = str(download_root) if download_root is not None else None
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.decoding_settings = dict(decoding_settings or {})
        self.inference_mode = normalized_inference_mode
        self.batch_size = batch_size
        self.allow_cpu_fallback = allow_cpu_fallback
        self.fallback_compute_type = fallback_compute_type
        self.load_diagnostic: str | None = None
        self._model: Any = None
        self._batched_pipeline: Any = None
        self._cuda_runtime_handles: list[Any] = []
        self._cuda_runtime_directories: list[str] = []
        self._cuda_runtime_prepared = False
        effective_vad = self.decoding_settings.get(
            "vad_filter", _env_bool("VSR_FASTER_WHISPER_VAD_FILTER", True)
        )
        effective_context = self.decoding_settings.get(
            "condition_on_previous_text",
            _env_bool("VSR_FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT", False),
        )
        # Checkpoint identity must describe effective decoder behavior, not
        # whether a caller supplied a default explicitly. Without this
        # canonical form, ``{}`` and an equivalent explicit settings mapping
        # create different cache keys and force an unnecessary full decode.
        self.checkpoint_decoding_settings = {
            "vad_filter": bool(effective_vad),
            "condition_on_previous_text": bool(effective_context),
        }
        # Include effective decoder policy in both the pipeline run key and
        # the chunk checkpoint key. A safer decoder setting must never reuse
        # audio produced under a hallucination-prone policy.
        identity_tokens = [
            f"{self.backend_name}|{self.model_name_or_path}",
            f"vad={bool(effective_vad)}|condition_on_previous_text={bool(effective_context)}",
            f"inference={self.inference_mode}|batch={self.batch_size}",
        ]
        if self.model_revision is not None:
            identity_tokens.append(f"revision={self.model_revision}")
        if self.model_signature is not None:
            identity_tokens.append(f"signature={self.model_signature}")
        self.cache_identity = "|".join(identity_tokens)

    def _prepare_cuda_runtime(self) -> None:
        """Expose optional pip-provided CUDA DLLs to Python on Windows.

        The base skill intentionally does not install CUDA runtimes.  When a
        developer has installed an isolated ``nvidia-*`` runtime package, its
        DLL directory is not always on ``PATH``; registering it here enables
        faster-whisper to use the existing GPU without changing the offline
        model/download policy.  Missing directories are a normal CPU-fallback
        condition.
        """

        if self._cuda_runtime_prepared:
            return
        self._cuda_runtime_prepared = True
        if os.name != "nt" or not hasattr(os, "add_dll_directory"):
            return
        relative_directories = (
            Path("nvidia") / "cublas" / "bin",
            Path("nvidia") / "cudnn" / "bin",
            Path("nvidia") / "cuda_nvrtc" / "bin",
            Path("nvidia") / "cuda_runtime" / "bin",
        )
        seen: set[str] = set()
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        path_keys = {entry.casefold() for entry in path_entries if entry}
        for entry in sys.path:
            root = Path(entry)
            if not root.is_dir():
                continue
            for relative in relative_directories:
                candidate = (root / relative).resolve()
                key = str(candidate).casefold()
                if key in seen or not candidate.is_dir():
                    continue
                seen.add(key)
                try:
                    self._cuda_runtime_handles.append(os.add_dll_directory(str(candidate)))
                except OSError:
                    continue
                self._cuda_runtime_directories.append(str(candidate))
                if key not in path_keys:
                    path_entries.insert(0, str(candidate))
                    path_keys.add(key)
        os.environ["PATH"] = os.pathsep.join(path_entries)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        model_path = Path(self.model_name_or_path)
        if not model_path.exists() and not self.allow_model_download:
            raise ModelDownloadPermissionError(
                "A local faster-whisper model path is required unless model download "
                "permission is explicit"
            )
        self._prepare_cuda_runtime()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRDependencyError(
                "faster-whisper is required for the production ASR adapter"
            ) from exc
        try:
            self._model = WhisperModel(
                self.model_name_or_path,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
            )
        except Exception as exc:
            cuda_like = str(self.device).casefold() in {"cuda", "auto"}
            if not self.allow_cpu_fallback or not cuda_like:
                raise ASRDependencyError(
                    f"Unable to load faster-whisper on device={self.device!r}: {exc}"
                ) from exc
            self.load_diagnostic = (
                f"CUDA faster-whisper load failed ({exc}); retried locally on CPU "
                f"compute_type={self.fallback_compute_type}."
            )
            self.device = "cpu"
            self.compute_type = self.fallback_compute_type
            try:
                self._model = WhisperModel(
                    self.model_name_or_path,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=self.download_root,
                    cpu_threads=self.cpu_threads,
                    num_workers=self.num_workers,
                )
            except Exception as fallback_exc:
                raise ASRDependencyError(
                    f"Unable to load faster-whisper on CUDA or CPU: {fallback_exc}"
                ) from fallback_exc
        return self._model

    def _load_batched_pipeline(self, model: Any) -> Any:
        """Build the optional batched decoder around an already-loaded model.

        Batched inference is deliberately never selected implicitly.  It can
        change decoder scheduling and therefore output boundaries, so callers
        must opt in and review the resulting transcript.  The model itself is
        shared with the standard path; this adds no second copy of the weights.
        """

        if self._batched_pipeline is not None:
            return self._batched_pipeline
        try:
            from faster_whisper import BatchedInferencePipeline
        except ImportError as exc:
            raise ASRDependencyError(
                "faster-whisper BatchedInferencePipeline is unavailable; use "
                "inference_mode='standard' or upgrade faster-whisper"
            ) from exc
        try:
            self._batched_pipeline = BatchedInferencePipeline(model)
        except Exception as exc:
            raise ASRDependencyError(
                f"Unable to initialize faster-whisper batched inference: {exc}"
            ) from exc
        return self._batched_pipeline

    def close(self) -> None:
        """Release the native CTranslate2 model before another worker starts."""
        self._model = None
        self._batched_pipeline = None

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        interval_start_ms: int | None = None,
        interval_end_ms: int | None = None,
        context_padding_ms: int = 0,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> ASRResult:
        path = Path(audio_path)
        if not path.is_file():
            raise ASRError(f"Audio input does not exist: {path}")
        if context_padding_ms < 0:
            raise ASRError("context_padding_ms cannot be negative")
        if (
            interval_start_ms is not None
            and interval_end_ms is not None
            and interval_end_ms <= interval_start_ms
        ):
            raise ASRError("ASR interval end must be after its start")
        if (interval_start_ms is None) != (interval_end_ms is None):
            raise ASRError("A bounded ASR request requires both interval boundaries")
        if interval_start_ms is not None and interval_end_ms is not None:
            # Faster-whisper accepts paths but does not guarantee that every backend
            # honors a bounded seek identically. Extracting the requested audio makes
            # the production boundary explicit and testable.
            from .transcript_repair import extract_interval_audio

            with tempfile.TemporaryDirectory(prefix="vsr-asr-interval-") as temporary:
                extraction = extract_interval_audio(
                    path,
                    Path(temporary) / "interval.wav",
                    interval_start_ms,
                    interval_end_ms,
                    context_padding_ms=context_padding_ms,
                )
                local_result = self.transcribe(
                    extraction.output_path,
                    language=language,
                    word_timestamps=word_timestamps,
                )
                local_result.segments = offset_transcript_timestamps(
                    local_result.segments, extraction.actual_start_ms
                )
                local_result.metadata.update(
                    {
                        "interval_start_ms": interval_start_ms,
                        "interval_end_ms": interval_end_ms,
                        "context_padding_ms": context_padding_ms,
                        "extracted_start_ms": extraction.actual_start_ms,
                        "extracted_end_ms": extraction.actual_end_ms,
                    }
                )
                return local_result
        model = self._load_model()
        settings: dict[str, Any] = {
            "word_timestamps": word_timestamps,
            # Long recordings often contain pauses, applause, or repeated
            # room noise. VAD suppresses Whisper's silence hallucinations;
            # disabling cross-window text carry prevents a repeated phrase in
            # one chunk from cascading through the next chunk. Both remain
            # explicit environment overrides for compatibility experiments.
            "vad_filter": _env_bool("VSR_FASTER_WHISPER_VAD_FILTER", True),
            "condition_on_previous_text": _env_bool(
                "VSR_FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT", False
            ),
        }
        settings.update(self.decoding_settings)
        # These are adapter controls rather than faster-whisper decoder
        # arguments.  Keeping them out of ``settings`` avoids a TypeError when
        # callers pass a shared settings mapping to both inference modes.
        settings.pop("inference_mode", None)
        settings.pop("batch_size", None)
        batched_vad_override = False
        if (
            self.inference_mode == "batched"
            and settings.get("vad_filter") is False
            and not settings.get("clip_timestamps")
        ):
            # BatchedInferencePipeline requires either VAD or explicit clips
            # for media longer than its 30-second single-clip shortcut.  Keep
            # standard decoding untouched; the experimental path uses its
            # built-in speech chunking rather than truncating long media.
            settings["vad_filter"] = True
            batched_vad_override = True
        elif self.inference_mode == "batched" and settings.get("vad_filter") is True:
            # Batched mode's contract includes VAD even when the safe global
            # default already enabled it; retain the telemetry marker used by
            # callers to distinguish this guarded path from standard decode.
            batched_vad_override = True
        if language is not None:
            # Whisper's multilingual decoder uses the ISO-639-1 `tl` code for
            # Filipino, while the public CLI accepts the standards-based
            # `fil` spelling as well.
            whisper_language = (
                "tl"
                if language.casefold().split("-")[0] in {"fil", "tl"}
                else language
            )
            settings["language"] = whisper_language
        try:
            if self.inference_mode == "batched":
                batched_pipeline = self._load_batched_pipeline(model)
                segments_iter, info = batched_pipeline.transcribe(
                    str(path), batch_size=self.batch_size, **settings
                )
            else:
                segments_iter, info = model.transcribe(str(path), **settings)
            # faster-whisper may defer CUDA/cuBLAS loading until the generator
            # is consumed, so force that work inside the fallback boundary.
            segments_list = list(segments_iter)
        except Exception as exc:
            cuda_like = str(self.device).casefold() in {"cuda", "auto"}
            if not self.allow_cpu_fallback or not cuda_like or not _is_cuda_runtime_failure(exc):
                raise ASRDependencyError(
                    f"faster-whisper transcription failed on device={self.device!r}: {exc}"
                ) from exc
            self.load_diagnostic = (
                f"CUDA faster-whisper decode failed ({exc}); retried locally on CPU "
                f"compute_type={self.fallback_compute_type}."
            )
            self._model = None
            self._batched_pipeline = None
            self.device = "cpu"
            self.compute_type = self.fallback_compute_type
            return self.transcribe(
                path,
                language=language,
                word_timestamps=word_timestamps,
            )
        offset_ms = interval_start_ms or 0
        normalized: list[ParsedTranscriptSegment] = []
        for index, segment in enumerate(segments_list):
            start_ms = int(round(float(segment.start) * 1000)) + offset_ms
            end_ms = int(round(float(segment.end) * 1000)) + offset_ms
            raw_text = str(segment.text)
            segment_id = _stable_asr_id(self.backend_name, index, start_ms, end_ms, raw_text)
            words: list[dict[str, Any]] = []
            for word_index, word in enumerate(getattr(segment, "words", None) or []):
                word_start = (
                    int(round(float(word.start) * 1000)) + offset_ms
                    if word.start is not None
                    else None
                )
                word_end = (
                    int(round(float(word.end) * 1000)) + offset_ms if word.end is not None else None
                )
                words.append(
                    {
                        "word_id": _stable_asr_id(
                            segment_id,
                            word_index,
                            word_start,
                            word_end,
                            str(word.word),
                            namespace="word",
                        ),
                        "text": str(word.word),
                        "start_ms": word_start,
                        "end_ms": word_end,
                        "confidence": float(word.probability)
                        if word.probability is not None
                        else None,
                        "source": self.backend_name,
                        "language": getattr(info, "language", language),
                        "uncertainty_flags": []
                        if word.probability is None or word.probability >= 0.5
                        else ["low_confidence"],
                    }
                )
            confidence_values = [
                word["confidence"] for word in words if word["confidence"] is not None
            ]
            normalized.append(
                ParsedTranscriptSegment(
                    segment_id=segment_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    timing_provenance="faster_whisper_word_timestamps",
                    raw_text=raw_text,
                    normalized_text=raw_text.strip(),
                    language=getattr(info, "language", language),
                    words=words,
                    confidence=sum(confidence_values) / len(confidence_values)
                    if confidence_values
                    else None,
                    verification_status="automatically_transcribed",
                    uncertainty_items=[]
                    if not confidence_values or min(confidence_values) >= 0.5
                    else ["low_confidence_wording"],
                    substantive=bool(raw_text.strip()),
                )
            )
        metadata = {
            "backend": self.backend_name,
            "model": self.model_name_or_path,
            "model_revision": self.model_revision,
            "model_signature": self.model_signature,
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "num_workers": self.num_workers,
            "inference_mode": self.inference_mode,
            "batch_size": self.batch_size if self.inference_mode == "batched" else None,
            "accuracy_warning": (
                "batched_inference_requires_transcript_review"
                if self.inference_mode == "batched"
                else None
            ),
            "batched_vad_override": batched_vad_override,
            "decoding_settings": settings,
            "model_path_or_revision": self.model_name_or_path,
            "package_versions": {
                "faster-whisper": _package_version("faster-whisper"),
                "ctranslate2": _package_version("ctranslate2"),
            },
            "interval_start_ms": interval_start_ms,
            "interval_end_ms": interval_end_ms,
            "context_padding_ms": context_padding_ms,
            "load_diagnostic": self.load_diagnostic,
            "cuda_runtime_directories": list(self._cuda_runtime_directories),
        }
        return ASRResult(
            normalized,
            language=getattr(info, "language", language),
            language_probability=float(getattr(info, "language_probability", 0.0))
            if getattr(info, "language_probability", None) is not None
            else None,
            metadata=metadata,
        )


def _stable_asr_id(
    source: str,
    index: int,
    start: int | None,
    end: int | None,
    text: str,
    *,
    namespace: str = "transcript",
) -> str:
    return deterministic_id(namespace, source, index, start, end, text)


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


def normalize_asr_result(
    result: ASRResult | Sequence[Any] | Mapping[str, Any], *, source: str = "asr"
) -> ASRResult:
    """Normalize adapter output while preserving supplied segment objects."""

    if isinstance(result, ASRResult):
        return result
    if isinstance(result, Mapping):
        raw_segments = result.get("segments", [])
        if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
            raise ASRError("ASR result 'segments' must be a sequence")
        return ASRResult(
            list(raw_segments),
            language=str(result["language"]) if result.get("language") is not None else None,
            language_probability=float(result["language_probability"])
            if result.get("language_probability") is not None
            else None,
            metadata=dict(result.get("metadata", {})),
        )
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise ASRError("ASR adapter must return ASRResult, a segment sequence, or a result mapping")
    return ASRResult(list(result), metadata={"backend": source})


def offset_transcript_timestamps(segments: Sequence[Any], offset_ms: int) -> list[Any]:
    """Offset segment and word timestamps onto the original media timeline."""

    output: list[Any] = []
    for segment in segments:
        updates: dict[str, Any] = {}
        start = _get(segment, "start_ms")
        end = _get(segment, "end_ms")
        if start is not None:
            updates["start_ms"] = int(start) + offset_ms
        if end is not None:
            updates["end_ms"] = int(end) + offset_ms
        words = _get(segment, "words")
        if words is not None:
            adjusted_words: list[Any] = []
            for word in words:
                word_updates: dict[str, Any] = {}
                word_start = _get(word, "start_ms")
                word_end = _get(word, "end_ms")
                if word_start is not None:
                    word_updates["start_ms"] = int(word_start) + offset_ms
                if word_end is not None:
                    word_updates["end_ms"] = int(word_end) + offset_ms
                adjusted_words.append(_copy_with(word, **word_updates))
            updates["words"] = adjusted_words
        output.append(_copy_with(segment, **updates))
    return output


_TOKEN_RE = re.compile(r"\S+")


def _text(record: Any) -> str:
    for key in ("human_verified_text", "repaired_text", "normalized_text", "raw_text", "text"):
        value = _get(record, key)
        if value is not None:
            return str(value)
    return ""


def _word_text(word: Any) -> str:
    return str(_get(word, "text", _get(word, "word", "")))


def _normalized_token(value: str) -> str:
    return value.strip().casefold()


def _duplicate_word_prefix(previous: Any, current: Any) -> int:
    """Find a timestamp-supported suffix/prefix duplicate at a chunk boundary."""

    previous_words = list(_get(previous, "words", []) or [])
    current_words = list(_get(current, "words", []) or [])
    if not previous_words or not current_words:
        return 0
    maximum = min(len(previous_words), len(current_words))
    for width in range(maximum, 0, -1):
        left = previous_words[-width:]
        right = current_words[:width]
        if [_normalized_token(_word_text(word)) for word in left] != [
            _normalized_token(_word_text(word)) for word in right
        ]:
            continue
        timing_supported = True
        for old_word, new_word in zip(left, right, strict=True):
            old_start, old_end = _get(old_word, "start_ms"), _get(old_word, "end_ms")
            new_start, new_end = _get(new_word, "start_ms"), _get(new_word, "end_ms")
            if None in (old_start, old_end, new_start, new_end):
                timing_supported = False
                break
            if min(int(old_end), int(new_end)) <= max(int(old_start), int(new_start)):
                timing_supported = False
                break
        if timing_supported:
            return width
    return 0


def _trim_duplicate_words(previous: Any, current: Any) -> Any | None:
    duplicate_count = _duplicate_word_prefix(previous, current)
    if duplicate_count == 0:
        return current
    words = list(_get(current, "words", []) or [])[duplicate_count:]
    if not words:
        return None
    raw_text = "".join(_word_text(word) for word in words)
    if raw_text and not any(value[:1].isspace() for value in (_word_text(word) for word in words)):
        raw_text = " ".join(_word_text(word) for word in words)
    first_start = _get(words[0], "start_ms")
    return _copy_with(
        current,
        start_ms=first_start if first_start is not None else _get(current, "start_ms"),
        raw_text=raw_text,
        normalized_text=raw_text.strip(),
        words=words,
    )


def _overlap_ratio(left: Any, right: Any) -> float:
    left_start, left_end = _get(left, "start_ms"), _get(left, "end_ms")
    right_start, right_end = _get(right, "start_ms"), _get(right, "end_ms")
    if None in (left_start, left_end, right_start, right_end):
        return 0.0
    overlap = max(0, min(int(left_end), int(right_end)) - max(int(left_start), int(right_start)))
    denominator = max(1, min(int(left_end) - int(left_start), int(right_end) - int(right_start)))
    return overlap / denominator


def merge_overlapping_segments(
    segments: Sequence[Any], *, minimum_overlap_ratio: float = 0.2
) -> list[Any]:
    """Merge chunk-boundary ASR output without treating repetition as a bag of words.

    Text-identical segments with meaningful timestamp overlap are removed. A
    partial suffix/prefix is trimmed only when matching word intervals overlap;
    legitimate repeated phrases at different times remain present.
    """

    indexed = list(enumerate(segments))
    indexed.sort(
        key=lambda item: (
            _get(item[1], "start_ms") is None,
            _get(item[1], "start_ms") or 0,
            _get(item[1], "end_ms") or 0,
            item[0],
        )
    )
    merged: list[Any] = []
    for _, segment in indexed:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        if (
            " ".join(_text(previous).casefold().split())
            == " ".join(_text(segment).casefold().split())
            and _overlap_ratio(previous, segment) >= minimum_overlap_ratio
        ):
            previous_confidence = _get(previous, "confidence")
            current_confidence = _get(segment, "confidence")
            if current_confidence is not None and (
                previous_confidence is None or current_confidence > previous_confidence
            ):
                merged[-1] = segment
            continue
        trimmed = _trim_duplicate_words(previous, segment)
        if trimmed is not None:
            merged.append(trimmed)
    return merged


def merge_chunk_results(
    chunks: Sequence[tuple[int, ASRResult | Sequence[Any] | Mapping[str, Any]]],
) -> ASRResult:
    """Offset completed chunk results and deduplicate only timestamp-overlapping copies."""

    all_segments: list[Any] = []
    metadata: dict[str, Any] = {"chunks": []}
    language: str | None = None
    for offset_ms, raw_result in sorted(chunks, key=lambda item: item[0]):
        result = normalize_asr_result(raw_result)
        all_segments.extend(offset_transcript_timestamps(result.segments, offset_ms))
        metadata["chunks"].append({"offset_ms": offset_ms, "metadata": result.metadata})
        language = language or result.language
    return ASRResult(merge_overlapping_segments(all_segments), language=language, metadata=metadata)


def checkpoint_cache_key(
    adapter: ASRAdapter,
    media_path: str | Path,
    *,
    interval_start_ms: int,
    interval_end_ms: int,
    chunk_ms: int,
    overlap_ms: int,
    language: str | None,
    media_sha256: str | None = None,
) -> str:
    """Key checkpoint state by media bytes, chunk config, and model/backend config."""

    path = Path(media_path)
    adapter_config = {
        "adapter_class": f"{adapter.__class__.__module__}.{adapter.__class__.__qualname__}",
        "backend": getattr(adapter, "backend_name", None),
        "model": getattr(adapter, "model_name_or_path", None),
        "device": getattr(adapter, "device", None),
        "compute_type": getattr(adapter, "compute_type", None),
        "inference_mode": getattr(adapter, "inference_mode", "standard"),
        "batch_size": (
            getattr(adapter, "batch_size", None)
            if getattr(adapter, "inference_mode", "standard") == "batched"
            else None
        ),
        "decoding_settings": getattr(
            adapter,
            "checkpoint_decoding_settings",
            getattr(adapter, "decoding_settings", None),
        ),
        "cache_identity": getattr(adapter, "cache_identity", None),
    }
    # Do not perturb cache keys for explicitly constructed legacy adapters that
    # have no model identity token; automatic Whisper resolution adds these
    # fields only when a revision/signature is actually available.
    for identity_key in ("model_revision", "model_signature"):
        identity_value = getattr(adapter, identity_key, None)
        if identity_value is not None:
            adapter_config[identity_key] = identity_value
    language_hint_policy = os.environ.get("VSR_ASR_LANGUAGE_HINT", "").strip().lower()
    payload = {
        "orchestration_schema": "checkpointed-asr-v4-language-hint-worker-policy",
        "media_sha256": media_sha256 or sha256_file(path),
        "media_size": path.stat().st_size,
        "interval_start_ms": interval_start_ms,
        "interval_end_ms": interval_end_ms,
        "chunk_ms": chunk_ms,
        "overlap_ms": overlap_ms,
        "language": language,
        "language_hint_policy": language_hint_policy or "off",
        "num_workers": getattr(adapter, "num_workers", None),
        "adapter": adapter_config,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _chunk_ranges(
    start_ms: int, end_ms: int, chunk_ms: int, overlap_ms: int
) -> list[tuple[int, int]]:
    if start_ms < 0 or end_ms <= start_ms:
        raise ASRError("Checkpointed ASR interval must have a valid positive range")
    if chunk_ms <= 0 or overlap_ms < 0 or overlap_ms >= chunk_ms:
        raise ASRError("Chunk size must be positive and overlap must be smaller than the chunk")
    stride = chunk_ms - overlap_ms
    ranges: list[tuple[int, int]] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(end_ms, cursor + chunk_ms)
        ranges.append((cursor, chunk_end))
        if chunk_end == end_ms:
            break
        cursor += stride
    return ranges


def _serialize_asr_record(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if hasattr(record, "model_dump"):
        return dict(record.model_dump(mode="json"))
    if is_dataclass(record):
        return asdict(record)  # type: ignore[arg-type]
    raise ASRError(f"ASR checkpoint cannot serialize record type {type(record)!r}")


def _segments_digest(segments: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(segments),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _read_chunk_checkpoint(
    path: Path,
    *,
    cache_key: str,
    chunk_index: int,
    start_ms: int,
    end_ms: int,
) -> ASRResult | None:
    loaded = _read_chunk_checkpoint_payload(
        path,
        cache_key=cache_key,
        chunk_index=chunk_index,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    return loaded[0] if loaded is not None else None


def _read_chunk_checkpoint_payload(
    path: Path,
    *,
    cache_key: str,
    chunk_index: int,
    start_ms: int,
    end_ms: int,
) -> tuple[ASRResult, dict[str, Any]] | None:
    """Read and validate one checkpoint while retaining its parsed payload.

    Shared-cache hits need to be materialized into the project checkpoint.
    Returning the validated payload avoids a second JSON parse on the warm
    resume path while preserving the existing digest and schema gates.
    """

    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("cache_key") != cache_key
        or payload.get("chunk_index") != chunk_index
        or payload.get("start_ms") != start_ms
        or payload.get("end_ms") != end_ms
    ):
        return None
    segments = payload.get("segments")
    if not isinstance(segments, list) or not all(isinstance(item, Mapping) for item in segments):
        return None
    segment_records = [dict(item) for item in segments]
    if payload.get("segments_digest") != _segments_digest(segment_records):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    language = payload.get("language")
    probability = payload.get("language_probability")
    result = ASRResult(
        segment_records,
        language=str(language) if language is not None else None,
        language_probability=float(probability) if probability is not None else None,
        metadata=dict(metadata),
    )
    return result, dict(payload)


def _shared_asr_cache_limit() -> int:
    """Return the total shared ASR-cache budget in bytes."""

    default_limit = 512 * 1024 * 1024
    raw_limit = os.environ.get("VSR_ASR_SHARED_CACHE_MAX_BYTES", "").strip()
    if not raw_limit:
        return default_limit
    try:
        return max(0, int(raw_limit))
    except ValueError:
        LOGGER.warning("Ignoring invalid VSR_ASR_SHARED_CACHE_MAX_BYTES=%r", raw_limit)
        return default_limit


def _prune_shared_asr_cache(
    cache_root: Path, *, current_path: Path, cache_limit: int
) -> None:
    """Bound flat shared ASR chunk receipts by removing oldest cache entries.

    A newly processed chunk used to trigger a complete ``glob``/``stat`` pass
    over every shared receipt.  That made the metadata work quadratic in the
    number of chunks when a cache was warm.  The process-local ledger below
    accounts for the just-written path with one ``stat`` and defers a complete
    reconciliation to every ``_ASR_PRUNE_INTERVAL`` writes (or immediately
    when the estimated total crosses the limit).  A new process always starts
    with a full inventory, so stale state cannot survive process boundaries.

    The ledger is only an acceleration hint: pruning still uses a full,
    deterministic inventory before deleting anything, and an inventory failure
    discards the hint so the next write retries the safe path.
    """

    if cache_limit <= 0 or not cache_root.is_dir():
        # Do not carry a stale size estimate across a removed/recreated cache
        # directory (or a temporary budget opt-out in the same process).
        try:
            _ASR_PRUNE_STATE.pop(cache_root.resolve(), None)
        except OSError:
            pass
        return
    root = cache_root.resolve()
    current_resolved = current_path.resolve()

    state = _ASR_PRUNE_STATE.get(root)
    if state is not None:
        writes, total_bytes, sizes = state
        current_size = 0
        try:
            if current_path.is_file() and not current_path.is_symlink():
                current_size = int(current_path.stat().st_size)
        except OSError:
            # The writer may have been interrupted between the atomic replace
            # and this bookkeeping call.  Remove the stale ledger entry and
            # let the periodic full inventory establish the next baseline.
            current_size = 0
        previous_size = sizes.get(current_resolved, 0)
        total_bytes = max(0, total_bytes - previous_size) + current_size
        if current_size:
            sizes[current_resolved] = current_size
        else:
            sizes.pop(current_resolved, None)
        writes += 1
        _ASR_PRUNE_STATE[root] = (writes, total_bytes, sizes)
        if total_bytes <= cache_limit and writes < _ASR_PRUNE_INTERVAL:
            return

    try:
        entries = [
            item
            for item in root.glob("*.json")
            if item.is_file() and not item.is_symlink()
        ]
        entry_sizes = {item.resolve(): int(item.stat().st_size) for item in entries}
        total_bytes = sum(entry_sizes.values())
        if total_bytes <= cache_limit:
            _ASR_PRUNE_STATE[root] = (0, total_bytes, entry_sizes)
            return
        ordered = sorted(entries, key=lambda item: (item.stat().st_mtime_ns, item.as_posix()))
        for item in ordered:
            if total_bytes <= cache_limit:
                break
            if item.resolve() == current_resolved:
                continue
            try:
                size = item.stat().st_size
                item.unlink()
                total_bytes -= size
                entry_sizes.pop(item.resolve(), None)
            except OSError:
                LOGGER.warning("Unable to prune shared ASR checkpoint: %s", item)
        _ASR_PRUNE_STATE[root] = (0, total_bytes, entry_sizes)
    except OSError:
        _ASR_PRUNE_STATE.pop(root, None)
        LOGGER.info("Skipping shared ASR-cache pruning after receipt inspection failure")


def transcribe_checkpointed_chunks(
    adapter: ASRAdapter,
    media_path: str | Path,
    *,
    duration_ms: int,
    checkpoint_dir: str | Path,
    interval_start_ms: int = 0,
    interval_end_ms: int | None = None,
    chunk_ms: int = 900_000,
    overlap_ms: int = 15_000,
    language: str | None = None,
    media_sha256: str | None = None,
    shared_cache_dir: str | Path | None = None,
    progress_callback: ASRProgressCallback | None = None,
    progress_heartbeat_seconds: float | None = None,
) -> ASRResult:
    """Transcribe bounded overlapping chunks and atomically resume valid checkpoints.

    ``progress_callback`` is deliberately observational: callback failures are
    logged and never invalidate an otherwise valid ASR checkpoint.  This keeps
    progress/ETA reporting from becoming a new reconstruction failure mode.
    While a native decoder is running, an optional heartbeat is emitted from a
    daemon thread so a long or stalled chunk remains distinguishable from a
    dead process. Heartbeats do not interrupt, retry, or change decoder output.
    """

    source = Path(media_path)
    if not source.is_file():
        raise ASRError(f"Checkpointed ASR input does not exist: {source}")
    end_ms = duration_ms if interval_end_ms is None else interval_end_ms
    if duration_ms <= 0 or end_ms > duration_ms:
        raise ASRError("Checkpointed ASR interval exceeds the supplied media duration")
    heartbeat_value: str | float
    if progress_heartbeat_seconds is None:
        heartbeat_value = os.environ.get("VSR_ASR_PROGRESS_HEARTBEAT_SECONDS", "").strip() or "60"
    else:
        heartbeat_value = progress_heartbeat_seconds
    try:
        heartbeat_seconds = float(heartbeat_value)
    except (TypeError, ValueError) as exc:
        raise ASRError("ASR progress heartbeat interval must be a finite non-negative number") from exc
    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds < 0:
        raise ASRError("ASR progress heartbeat interval must be a finite non-negative number")
    ranges = _chunk_ranges(interval_start_ms, end_ms, chunk_ms, overlap_ms)
    cache_key = checkpoint_cache_key(
        adapter,
        source,
        interval_start_ms=interval_start_ms,
        interval_end_ms=end_ms,
        chunk_ms=chunk_ms,
        overlap_ms=overlap_ms,
        language=language,
        media_sha256=media_sha256,
    )
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    shared_cache_limit = _shared_asr_cache_limit()
    shared_directory: Path | None = None
    if shared_cache_dir is not None and shared_cache_limit > 0:
        candidate_shared_directory = Path(shared_cache_dir).expanduser()
        try:
            candidate_shared_directory.mkdir(parents=True, exist_ok=True)
            if candidate_shared_directory.is_dir():
                shared_directory = candidate_shared_directory.resolve()
        except OSError as exc:
            # A shared cache is an acceleration layer only.  Never turn a
            # read-only/restricted cache location into a reconstruction failure.
            LOGGER.warning("ASR shared cache unavailable at %s: %s", candidate_shared_directory, exc)
    chunk_results: list[tuple[int, ASRResult]] = []
    resumed: list[int] = []
    shared_cache_hits: list[int] = []
    processed: list[int] = []
    chunk_timings: list[dict[str, Any]] = []
    total_chunks = len(ranges)
    overall_started = time.monotonic()
    resolved_language = language
    auto_language_hint = language is None and os.environ.get(
        "VSR_ASR_LANGUAGE_HINT", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    language_hint_activated = False

    def notify(payload: Mapping[str, Any]) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(payload)
        except Exception:  # pragma: no cover - defensive boundary for telemetry only
            LOGGER.warning("ASR progress callback failed", exc_info=True)

    for index, (start_ms, chunk_end_ms) in enumerate(ranges):
        chunk_started = time.monotonic()
        checkpoint = directory / f"{cache_key}.chunk-{index:06d}.json"
        shared_checkpoint = (
            shared_directory / f"{cache_key}.chunk-{index:06d}.json"
            if shared_directory is not None
            else None
        )
        notify(
            {
                "event": "chunk_started",
                "backend": getattr(adapter, "backend_name", None),
                "chunk_index": index,
                "total_chunks": total_chunks,
                "start_ms": start_ms,
                "end_ms": chunk_end_ms,
                "completed_chunks": len(chunk_results),
                "fraction": round(len(chunk_results) / total_chunks, 6)
                if total_chunks
                else 1.0,
            }
        )
        result = _read_chunk_checkpoint(
            checkpoint,
            cache_key=cache_key,
            chunk_index=index,
            start_ms=start_ms,
            end_ms=chunk_end_ms,
        )
        disposition = "resumed"
        shared_readable = True
        if result is None and shared_checkpoint is not None:
            try:
                shared_readable = shared_checkpoint.stat().st_size <= shared_cache_limit
            except FileNotFoundError:
                # A missing entry is a normal cold-cache state and remains a
                # valid destination for this chunk's first write.
                shared_readable = True
            except OSError:
                shared_readable = False
        if result is None and shared_checkpoint is not None and shared_readable:
            loaded_shared = _read_chunk_checkpoint_payload(
                shared_checkpoint,
                cache_key=cache_key,
                chunk_index=index,
                start_ms=start_ms,
                end_ms=chunk_end_ms,
            )
            if loaded_shared is not None:
                result, validated_shared_payload = loaded_shared
                # Materialize the shared hit into the project-local checkpoint
                # so a project remains independently resumable/offline even if
                # the shared cache is later evicted.
                try:
                    # Checkpoint payloads are machine-consumed and can be
                    # large (word-level timestamps on long recordings).
                    # Keep materialized shared hits compact to avoid
                    # duplicating indentation whitespace in every project
                    # while preserving the exact parsed payload/digest.
                    atomic_write_json(checkpoint, validated_shared_payload, compact=True)
                except OSError:
                    # The validated result is still usable; a local copy is a
                    # convenience and must not invalidate an otherwise good hit.
                    LOGGER.warning("Unable to materialize ASR shared checkpoint %s", shared_checkpoint)
                shared_cache_hits.append(index)
                disposition = "resumed_shared"
        if result is None:
            full_media_passthrough = bool(
                start_ms == 0
                and chunk_end_ms == duration_ms
                and getattr(adapter, "supports_full_media_passthrough", False)
            )
            heartbeat_stop = Event()
            heartbeat_thread: Thread | None = None
            if progress_callback is not None and heartbeat_seconds > 0:

                def emit_heartbeats(
                    *,
                    stop_event: Event = heartbeat_stop,
                    chunk_index: int = index,
                    chunk_start_ms: int = start_ms,
                    chunk_end: int = chunk_end_ms,
                    started_at: float = chunk_started,
                    checkpoint_path: Path = checkpoint,
                    interval_seconds: float = heartbeat_seconds,
                ) -> None:
                    while not stop_event.wait(interval_seconds):
                        notify(
                            {
                                "event": "chunk_heartbeat",
                                "backend": getattr(adapter, "backend_name", None),
                                "chunk_index": chunk_index,
                                "total_chunks": total_chunks,
                                "start_ms": chunk_start_ms,
                                "end_ms": chunk_end,
                                "completed_chunks": len(chunk_results),
                                "fraction": round(len(chunk_results) / total_chunks, 6)
                                if total_chunks
                                else 1.0,
                                "elapsed_seconds": round(
                                    time.monotonic() - started_at, 6
                                ),
                                "heartbeat_interval_seconds": interval_seconds,
                                "checkpoint": str(checkpoint_path),
                            }
                        )

                heartbeat_thread = Thread(
                    target=emit_heartbeats,
                    name=f"vsr-asr-heartbeat-{index}",
                    daemon=True,
                )
                heartbeat_thread.start()
            try:
                if full_media_passthrough:
                    # The complete-media request is semantically equivalent to the
                    # bounded [0, duration] request for this explicitly opted-in
                    # adapter. Avoiding a temporary WAV saves a full container
                    # decode/remux while preserving decoder settings and timestamps.
                    raw_result = adapter.transcribe(
                        source,
                        language=resolved_language,
                        word_timestamps=True,
                    )
                else:
                    raw_result = adapter.transcribe(
                        source,
                        interval_start_ms=start_ms,
                        interval_end_ms=chunk_end_ms,
                        context_padding_ms=0,
                        language=resolved_language,
                        word_timestamps=True,
                    )
            finally:
                heartbeat_stop.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join()
            result = normalize_asr_result(raw_result, source=adapter.backend_name)
            if full_media_passthrough:
                result.metadata.setdefault("full_media_passthrough", True)
            segment_records = [_serialize_asr_record(segment) for segment in result.segments]
            checkpoint_payload = {
                "schema_version": "1.0",
                "cache_key": cache_key,
                "chunk_index": index,
                "start_ms": start_ms,
                "end_ms": chunk_end_ms,
                "segments": segment_records,
                "segments_digest": _segments_digest(segment_records),
                "language": result.language,
                "language_probability": result.language_probability,
                "metadata": result.metadata,
            }
            # ASR checkpoints are content-addressed, validated machine state;
            # compact JSON materially reduces SSD write/read amplification for
            # multi-hour videos without changing parsing or resume semantics.
            atomic_write_json(checkpoint, checkpoint_payload, compact=True)
            if shared_checkpoint is not None:
                try:
                    encoded_size = len(
                        json.dumps(
                            checkpoint_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        ).encode("utf-8")
                    )
                    if encoded_size <= shared_cache_limit:
                        atomic_write_json(shared_checkpoint, checkpoint_payload, compact=True)
                        _prune_shared_asr_cache(
                            shared_checkpoint.parent,
                            current_path=shared_checkpoint,
                            cache_limit=shared_cache_limit,
                        )
                except OSError as exc:
                    LOGGER.warning("Unable to write ASR shared checkpoint %s: %s", shared_checkpoint, exc)
            result = ASRResult(
                segment_records,
                language=result.language,
                language_probability=result.language_probability,
                metadata=dict(result.metadata),
            )
            processed.append(index)
            disposition = "processed"
        else:
            resumed.append(index)
        if auto_language_hint and resolved_language is None:
            observed_language = result.language
            observed_probability = result.language_probability
            if (
                observed_language
                and observed_probability is not None
                and observed_probability >= 0.9
            ):
                resolved_language = observed_language
                language_hint_activated = True
        # Bounded adapter timestamps are already on the original media timeline.
        chunk_results.append((0, result))
        elapsed_seconds = time.monotonic() - chunk_started
        completed_chunks = len(chunk_results)
        chunk_timings.append(
            {
                "chunk_index": index,
                "start_ms": start_ms,
                "end_ms": chunk_end_ms,
                "status": disposition,
                "elapsed_seconds": round(elapsed_seconds, 6),
            }
        )
        total_elapsed = time.monotonic() - overall_started
        average_seconds = total_elapsed / completed_chunks if completed_chunks else 0.0
        remaining_seconds = average_seconds * max(total_chunks - completed_chunks, 0)
        notify(
            {
                "event": "chunk_completed",
                "backend": getattr(adapter, "backend_name", None),
                "chunk_index": index,
                "total_chunks": total_chunks,
                "start_ms": start_ms,
                "end_ms": chunk_end_ms,
                "status": disposition,
                "shared_cache_hit": index in shared_cache_hits,
                "completed_chunks": completed_chunks,
                "processed_chunks": len(processed),
                "resumed_chunks": len(resumed),
                "fraction": round(completed_chunks / total_chunks, 6)
                if total_chunks
                else 1.0,
                "elapsed_seconds": round(total_elapsed, 6),
                "estimated_remaining_seconds": round(remaining_seconds, 6),
                "checkpoint": str(checkpoint),
                "chunk_timings": list(chunk_timings),
            }
        )
    merged = merge_chunk_results(chunk_results)
    model_metadata = next((result.metadata for _, result in chunk_results if result.metadata), {})
    merged.metadata.update(
        {
            "backend": getattr(adapter, "backend_name", None),
            "model": getattr(adapter, "model_name_or_path", model_metadata.get("model")),
            "checkpoint_cache_key": cache_key,
            "checkpoint_dir": str(directory),
            "chunk_ranges_ms": [list(interval) for interval in ranges],
            "processed_chunk_indexes": processed,
            "resumed_chunk_indexes": resumed,
            "shared_cache_hit_indexes": shared_cache_hits,
            "shared_cache_dir": str(shared_directory) if shared_directory is not None else None,
            "model_metadata": dict(model_metadata),
            "language_strategy": (
                "explicit"
                if language is not None
                else (
                    "first_chunk_high_confidence_hint"
                    if language_hint_activated
                    else "per_chunk_detection"
                )
            ),
            "progress": {
                "total_chunks": total_chunks,
                "processed_chunks": len(processed),
                "resumed_chunks": len(resumed),
                "elapsed_seconds": round(time.monotonic() - overall_started, 6),
                "chunk_timings": chunk_timings,
            },
        }
    )
    notify(
        {
            "event": "completed",
            "backend": getattr(adapter, "backend_name", None),
            "total_chunks": total_chunks,
            "completed_chunks": total_chunks,
            "processed_chunks": len(processed),
            "resumed_chunks": len(resumed),
            "fraction": 1.0,
            "elapsed_seconds": merged.metadata["progress"]["elapsed_seconds"],
            "estimated_remaining_seconds": 0.0,
            "chunk_timings": list(chunk_timings),
        }
    )
    return merged


# Concise aliases used by callers and tests.
offset_timestamps = offset_transcript_timestamps
merge_overlaps = merge_overlapping_segments
transcribe_with_checkpoints = transcribe_checkpointed_chunks


__all__ = [
    "ASRAdapter",
    "ASRDependencyError",
    "ASRError",
    "ASRResult",
    "ASRProgressCallback",
    "FasterWhisperAdapter",
    "ModelDownloadPermissionError",
    "ModelIndependentASRAdapter",
    "checkpoint_cache_key",
    "ensure_production_adapter",
    "merge_chunk_results",
    "merge_overlapping_segments",
    "merge_overlaps",
    "normalize_asr_result",
    "offset_timestamps",
    "offset_transcript_timestamps",
    "transcribe_checkpointed_chunks",
    "transcribe_with_checkpoints",
]
