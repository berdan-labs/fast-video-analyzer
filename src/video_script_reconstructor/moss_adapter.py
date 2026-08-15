"""Production MOSS speaker-aware transcription adapter."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .ids import deterministic_id
from .model_store import verify_model
from .subtitle_parse import ParsedTranscriptSegment
from .whisper_adapter import (
    ASRAdapter,
    ASRDependencyError,
    ASRError,
    ASRResult,
    offset_transcript_timestamps,
)

RESULT_PREFIX = "VSR_RESULT\t"


class MossTranscribeDiarizeAdapter(ASRAdapter):
    """Run hash-pinned MOSS custom code in a dedicated offline worker."""

    backend_name = "moss-transcribe-diarize"
    is_production = True

    def __init__(
        self,
        *,
        worker_python: str | Path | None = None,
        model_root: str | Path | None = None,
        model_name: str = "moss-transcribe-diarize-0.9b",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 4096,
        timeout_seconds: float = 1800.0,
    ) -> None:
        configured = worker_python or os.environ.get("VSR_MOSS_SPEECH_PYTHON")
        if configured is None:
            raise ASRDependencyError(
                "MOSS requires an isolated worker Python; set VSR_MOSS_SPEECH_PYTHON "
                "or pass worker_python"
            )
        self.worker_python = Path(configured).expanduser().resolve()
        if not self.worker_python.is_file():
            raise ASRDependencyError(f"MOSS worker Python is absent: {self.worker_python}")
        self.model_root = Path(model_root).expanduser().resolve() if model_root else None
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self.cache_identity = (
            f"{model_name}|device={device}|dtype={dtype}|max_new_tokens={max_new_tokens}"
        )
        self._model_status: dict[str, Any] | None = None

    def _status(self) -> dict[str, Any]:
        if self._model_status is None:
            status = verify_model(self.model_name, self.model_root)
            if not status.get("verified"):
                reason = (
                    status.get("reason") or status.get("missing_files") or "integrity check failed"
                )
                raise ASRDependencyError(
                    f"Local model {self.model_name!r} is not hash-verified: {reason}"
                )
            self._model_status = status
        return self._model_status

    def _invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUTF8": "1",
            }
        )
        package_root = str(Path(__file__).resolve().parents[1])
        inherited = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = package_root + (os.pathsep + inherited if inherited else "")
        try:
            completed = subprocess.run(
                [str(self.worker_python), "-m", "video_script_reconstructor.moss_speech_worker"],
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise ASRError(f"MOSS worker timed out after {self.timeout_seconds:g}s") from exc
        payload: dict[str, Any] | None = None
        for line in reversed(completed.stdout.splitlines()):
            if not line.startswith(RESULT_PREFIX):
                continue
            try:
                candidate = json.loads(line[len(RESULT_PREFIX) :])
            except json.JSONDecodeError as exc:
                raise ASRError("MOSS worker returned malformed JSON") from exc
            if isinstance(candidate, dict):
                payload = candidate
            break
        if payload is None:
            diagnostic = (completed.stderr or completed.stdout).strip()[-1200:]
            raise ASRError(f"MOSS worker returned no result: {diagnostic}")
        if completed.returncode != 0 or not payload.get("ok"):
            detail = payload.get("error") or completed.stderr.strip()[-1200:]
            raise ASRError(f"MOSS worker failed: {detail}")
        return payload

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
        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise ASRError(f"Audio input does not exist: {path}")
        if context_padding_ms < 0:
            raise ASRError("context_padding_ms cannot be negative")
        if (interval_start_ms is None) != (interval_end_ms is None):
            raise ASRError("A bounded ASR request requires both interval boundaries")
        if (
            interval_start_ms is not None
            and interval_end_ms is not None
            and interval_end_ms <= interval_start_ms
        ):
            raise ASRError("ASR interval end must be after its start")
        if interval_start_ms is not None and interval_end_ms is not None:
            from .transcript_repair import extract_interval_audio

            with tempfile.TemporaryDirectory(prefix="vsr-moss-interval-") as temporary:
                extraction = extract_interval_audio(
                    path,
                    Path(temporary) / "interval.wav",
                    interval_start_ms,
                    interval_end_ms,
                    context_padding_ms=context_padding_ms,
                )
                local_result = self.transcribe(extraction.output_path, language=language)
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

        status = self._status()
        payload = self._invoke(
            {
                "mode": "transcribe-diarize",
                "model_path": status["directory"],
                "audio_path": str(path),
                "device": self.device,
                "dtype": self.dtype,
                "max_new_tokens": self.max_new_tokens,
            }
        )
        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ASRError("MOSS returned no parseable timestamped speaker segments")
        segments: list[ParsedTranscriptSegment] = []
        speaker_mapping: dict[str, str] = {}
        for index, item in enumerate(raw_segments):
            if not isinstance(item, dict):
                raise ASRError("MOSS returned a malformed segment")
            start_ms = int(round(float(item["start"]) * 1000))
            end_ms = int(round(float(item["end"]) * 1000))
            text = str(item["text"]).strip()
            backend_speaker = str(item["speaker"])
            if backend_speaker not in speaker_mapping:
                speaker_mapping[backend_speaker] = f"Speaker {len(speaker_mapping) + 1}"
            speaker = speaker_mapping[backend_speaker]
            if start_ms < 0 or end_ms <= start_ms or not text:
                raise ASRError("MOSS returned invalid segment evidence")
            segments.append(
                ParsedTranscriptSegment(
                    segment_id=deterministic_id(
                        "transcript",
                        self.backend_name,
                        index,
                        start_ms,
                        end_ms,
                        backend_speaker,
                        text,
                    ),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    timing_provenance="moss_generated_segment_timestamps",
                    raw_text=text,
                    normalized_text=text,
                    speaker_label=speaker,
                    language=language,
                    words=[],
                    confidence=None,
                    verification_status="automatically_transcribed",
                    uncertainty_items=["model_generated_speaker_label"],
                    substantive=True,
                )
            )
        prompt = str(payload.get("prompt", ""))
        return ASRResult(
            segments,
            language=language,
            language_probability=None,
            metadata={
                "backend": self.backend_name,
                "model": self.model_name,
                "model_path_or_revision": status.get("revision"),
                "model_directory": status.get("directory"),
                "device": payload.get("device", self.device),
                "dtype": payload.get("dtype", self.dtype),
                "package_versions": payload.get("package_versions", {}),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "raw_model_output": payload.get("raw_output"),
                "generated_tokens": payload.get("generated_tokens"),
                "speaker_labels_are_anonymous": True,
                "speaker_label_mapping": speaker_mapping,
                "offline": True,
            },
        )
