from __future__ import annotations

import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import video_script_reconstructor.whisper_adapter as whisper_adapter_module
from video_script_reconstructor.subtitle_parse import ParsedTranscriptSegment
from video_script_reconstructor.transcript_repair import ExtractionResult
from video_script_reconstructor.whisper_adapter import (
    ASRError,
    ASRResult,
    FasterWhisperAdapter,
    ModelDownloadPermissionError,
    ModelIndependentASRAdapter,
    checkpoint_cache_key,
    ensure_production_adapter,
    merge_chunk_results,
    merge_overlapping_segments,
    offset_transcript_timestamps,
    transcribe_checkpointed_chunks,
)


def seg(
    identifier: str, start: int, end: int, text: str, confidence: float = 0.8
) -> ParsedTranscriptSegment:
    return ParsedTranscriptSegment(identifier, start, end, "asr", text, text, confidence=confidence)


def test_model_independent_adapter_calls_supplied_recognizer(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixture")
    calls = []

    def recognize(path: Path, **kwargs: object):
        calls.append((path, kwargs))
        return [seg("a", 0, 500, "recognized")]

    adapter = ModelIndependentASRAdapter(recognize)
    result = adapter.transcribe(audio, interval_start_ms=100, interval_end_ms=600, language="en")
    assert result[0].raw_text == "recognized"
    assert calls[0][1]["interval_start_ms"] == 100
    with pytest.raises(ASRError):
        ensure_production_adapter(adapter)


def test_timestamp_offset_includes_words() -> None:
    source = {
        "segment_id": "a",
        "start_ms": 10,
        "end_ms": 50,
        "raw_text": "word",
        "words": [{"word_id": "w", "text": "word", "start_ms": 12, "end_ms": 40}],
    }
    adjusted = offset_transcript_timestamps([source], 1000)
    assert adjusted[0]["start_ms"] == 1010
    assert adjusted[0]["words"][0]["end_ms"] == 1040
    assert source["start_ms"] == 10


def test_overlap_merge_deduplicates_only_temporally_overlapping_copies() -> None:
    merged = merge_overlapping_segments(
        [seg("a", 0, 1000, "repeat"), seg("b", 800, 1200, "repeat"), seg("c", 2000, 2500, "repeat")]
    )
    assert [item.segment_id for item in merged] == ["a", "c"]
    chunks = merge_chunk_results(
        [(0, [seg("d", 0, 1000, "one")]), (800, [seg("e", 0, 400, "one")])]
    )
    assert len(chunks) == 1


def test_overlap_merge_trims_timestamp_supported_boundary_words() -> None:
    first = {
        "segment_id": "a",
        "start_ms": 0,
        "end_ms": 1000,
        "raw_text": "hello world",
        "normalized_text": "hello world",
        "words": [
            {"text": "hello", "start_ms": 0, "end_ms": 500},
            {"text": "world", "start_ms": 500, "end_ms": 1000},
        ],
    }
    second = {
        "segment_id": "b",
        "start_ms": 800,
        "end_ms": 1500,
        "raw_text": "world again",
        "normalized_text": "world again",
        "words": [
            {"text": "world", "start_ms": 800, "end_ms": 1000},
            {"text": "again", "start_ms": 1000, "end_ms": 1500},
        ],
    }
    merged = merge_overlapping_segments([first, second])
    assert [item["normalized_text"] for item in merged] == ["hello world", "again"]
    assert merged[1]["start_ms"] == 1000


def test_faster_whisper_bounded_request_extracts_and_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"media")
    requests: list[tuple[int, int, int]] = []

    def fake_extract(source, output, start, end, *, context_padding_ms):
        clip = Path(output)
        clip.write_bytes(b"clip")
        requests.append((start, end, context_padding_ms))
        return ExtractionResult(clip, start, end, 800, 2200, context_padding_ms, ("fake",))

    class FakeModel:
        def transcribe(self, path, **settings):
            assert settings["language"] == "tl"
            return iter(
                [SimpleNamespace(start=0.2, end=0.4, text=" bounded", words=[])]
            ), SimpleNamespace(language="en", language_probability=0.99)

    monkeypatch.setattr(
        "video_script_reconstructor.transcript_repair.extract_interval_audio", fake_extract
    )
    adapter = FasterWhisperAdapter(model="local")
    adapter._model = FakeModel()
    result = adapter.transcribe(
        media,
        interval_start_ms=1000,
        interval_end_ms=2000,
        context_padding_ms=200,
        language="fil",
    )
    assert requests == [(1000, 2000, 200)]
    assert (result[0].start_ms, result[0].end_ms) == (1000, 1200)
    assert result.metadata["extracted_start_ms"] == 800
    assert result.metadata["cpu_threads"] == 0
    assert result.metadata["num_workers"] == 1


def test_faster_whisper_batched_mode_is_explicit_and_preserves_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"media")
    standard_calls: list[dict[str, object]] = []
    batched_calls: list[dict[str, object]] = []

    class FakeModel:
        def transcribe(self, path, **settings):
            standard_calls.append({"path": path, **settings})
            return iter([SimpleNamespace(start=0.1, end=0.4, text=" standard", words=[])]), (
                SimpleNamespace(language="tl", language_probability=0.99)
            )

    class FakeBatchedPipeline:
        def __init__(self, model) -> None:
            assert isinstance(model, FakeModel)

        def transcribe(self, path, **settings):
            batched_calls.append({"path": path, **settings})
            assert settings["batch_size"] == 4
            assert settings["vad_filter"] is True
            assert "inference_mode" not in settings
            return iter([SimpleNamespace(start=0.1, end=0.4, text=" batched", words=[])]), (
                SimpleNamespace(language="tl", language_probability=0.99)
            )

    fake_module = ModuleType("faster_whisper")
    fake_module.BatchedInferencePipeline = FakeBatchedPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    standard = FasterWhisperAdapter(model="local")
    standard._model = FakeModel()
    standard_result = standard.transcribe(media, language="fil")
    assert standard_result[0].raw_text.strip() == "standard"
    assert standard_result.metadata["inference_mode"] == "standard"
    assert standard_result.metadata["batch_size"] is None
    assert standard_calls[0]["language"] == "tl"

    batched = FasterWhisperAdapter(
        model="local",
        inference_mode="batched",
        batch_size=4,
        decoding_settings={"inference_mode": "batched"},
    )
    batched._model = FakeModel()
    batched_result = batched.transcribe(media, language="fil")
    assert batched_result[0].raw_text.strip() == "batched"
    assert batched_result.metadata["inference_mode"] == "batched"
    assert batched_result.metadata["batch_size"] == 4
    assert batched_result.metadata["accuracy_warning"] == (
        "batched_inference_requires_transcript_review"
    )
    assert batched_result.metadata["batched_vad_override"] is True
    assert batched_calls[0]["language"] == "tl"


def test_batched_configuration_error_does_not_trigger_cuda_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"media")

    class FakeModel:
        pass

    class FailingBatchedPipeline:
        def __init__(self, model) -> None:
            pass

        def transcribe(self, path, **settings):
            raise RuntimeError("No clip timestamps found")

    fake_module = ModuleType("faster_whisper")
    fake_module.BatchedInferencePipeline = FailingBatchedPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    adapter = FasterWhisperAdapter(
        model="local", device="cuda", inference_mode="batched", allow_cpu_fallback=True
    )
    adapter._model = FakeModel()

    with pytest.raises(ASRError, match="No clip timestamps found"):
        adapter.transcribe(media, language="fil")
    assert adapter.device == "cuda"


def test_faster_whisper_inference_mode_changes_checkpoint_identity(tmp_path: Path) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"media")
    standard = FasterWhisperAdapter(model="local")
    batched = FasterWhisperAdapter(model="local", inference_mode="batched", batch_size=4)
    common = {
        "media_path": media,
        "interval_start_ms": 0,
        "interval_end_ms": 1_000,
        "chunk_ms": 1_000,
        "overlap_ms": 0,
        "language": "fil",
        "media_sha256": "a" * 64,
    }
    assert checkpoint_cache_key(standard, **common) != checkpoint_cache_key(batched, **common)


def test_faster_whisper_worker_count_changes_checkpoint_identity(tmp_path: Path) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"media")
    one = FasterWhisperAdapter(model="local", num_workers=1)
    two = FasterWhisperAdapter(model="local", num_workers=2)
    common = {
        "media_path": media,
        "interval_start_ms": 0,
        "interval_end_ms": 1_000,
        "chunk_ms": 1_000,
        "overlap_ms": 0,
        "language": "fil",
        "media_sha256": "a" * 64,
    }
    assert checkpoint_cache_key(one, **common) != checkpoint_cache_key(two, **common)


def test_faster_whisper_model_revision_changes_checkpoint_identity(tmp_path: Path) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"media")
    one = FasterWhisperAdapter(model="local", model_revision="revision-a")
    two = FasterWhisperAdapter(model="local", model_revision="revision-b")
    common = {
        "media_path": media,
        "interval_start_ms": 0,
        "interval_end_ms": 1_000,
        "chunk_ms": 1_000,
        "overlap_ms": 0,
        "language": "fil",
        "media_sha256": "a" * 64,
    }
    assert one.cache_identity != two.cache_identity
    assert checkpoint_cache_key(one, **common) != checkpoint_cache_key(two, **common)


def test_faster_whisper_model_signature_changes_checkpoint_identity(tmp_path: Path) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"media")
    one = FasterWhisperAdapter(model="local", model_signature="stat-a")
    two = FasterWhisperAdapter(model="local", model_signature="stat-b")
    common = {
        "media_path": media,
        "interval_start_ms": 0,
        "interval_end_ms": 1_000,
        "chunk_ms": 1_000,
        "overlap_ms": 0,
        "language": "fil",
        "media_sha256": "a" * 64,
    }
    assert checkpoint_cache_key(one, **common) != checkpoint_cache_key(two, **common)


def test_checkpointed_full_media_whisper_request_skips_intermediate_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"media")

    class FakeModel:
        def transcribe(self, path, **settings):
            assert str(path) == str(media)
            return iter(
                [SimpleNamespace(start=0.1, end=0.4, text=" Mabuhay", words=[])]
            ), SimpleNamespace(language="tl", language_probability=0.99)

    def fail_extract(*_args, **_kwargs):
        raise AssertionError("full-media passthrough unexpectedly extracted a WAV")

    monkeypatch.setattr(
        "video_script_reconstructor.transcript_repair.extract_interval_audio", fail_extract
    )
    adapter = FasterWhisperAdapter(model="local")
    adapter._model = FakeModel()
    result = transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=1_000,
        checkpoint_dir=tmp_path / "checkpoints",
        chunk_ms=2_000,
        overlap_ms=0,
    )

    assert len(result.segments) == 1
    assert result.segments[0]["start_ms"] == 100
    assert result.language == "tl"
    assert result.metadata["model_metadata"]["full_media_passthrough"] is True


def test_faster_whisper_requires_download_permission_for_registry_model(tmp_path: Path) -> None:
    adapter = FasterWhisperAdapter(model="large-v3", allow_model_download=False)
    with pytest.raises(ModelDownloadPermissionError):
        adapter._load_model()


def test_checkpointed_chunks_resume_without_reinvoking_adapter(tmp_path: Path) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"checkpoint-media-content")
    calls: list[tuple[int, int]] = []

    def recognize(path: Path, **kwargs: object):
        start = int(kwargs["interval_start_ms"])
        end = int(kwargs["interval_end_ms"])
        calls.append((start, end))
        return [
            {
                "segment_id": f"segment-{start}",
                "start_ms": start,
                "end_ms": end,
                "raw_text": f"chunk {start}",
                "normalized_text": f"chunk {start}",
                "words": [],
            }
        ]

    adapter = ModelIndependentASRAdapter(recognize, name="checkpoint-fixture")
    first = transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=5000,
        checkpoint_dir=tmp_path / "checkpoints",
        chunk_ms=3000,
        overlap_ms=1000,
        language="en",
    )
    assert calls == [(0, 3000), (2000, 5000)]
    assert first.metadata["processed_chunk_indexes"] == [0, 1]
    assert first.metadata["resumed_chunk_indexes"] == []
    checkpoint_files = sorted((tmp_path / "checkpoints").glob("*.json"))
    assert len(checkpoint_files) == 2

    second = transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=5000,
        checkpoint_dir=tmp_path / "checkpoints",
        chunk_ms=3000,
        overlap_ms=1000,
        language="en",
    )
    assert calls == [(0, 3000), (2000, 5000)]
    assert second.metadata["processed_chunk_indexes"] == []
    assert second.metadata["resumed_chunk_indexes"] == [0, 1]
    assert second.metadata["checkpoint_cache_key"] == first.metadata["checkpoint_cache_key"]

    checkpoint_files[0].write_text("{corrupt", encoding="utf-8")
    third = transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=5000,
        checkpoint_dir=tmp_path / "checkpoints",
        chunk_ms=3000,
        overlap_ms=1000,
        language="en",
    )
    assert len(calls) == 3
    assert third.metadata["processed_chunk_indexes"] == [0]
    assert third.metadata["resumed_chunk_indexes"] == [1]

    changed_config = transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=5000,
        checkpoint_dir=tmp_path / "checkpoints",
        chunk_ms=3000,
        overlap_ms=1000,
        language="fr",
    )
    assert len(calls) == 5
    assert changed_config.metadata["checkpoint_cache_key"] != first.metadata["checkpoint_cache_key"]


def test_checkpointed_chunks_reuse_valid_shared_cache_across_projects(tmp_path: Path) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"shared-cache-media-content")
    calls: list[tuple[int, int]] = []

    def recognize(path: Path, **kwargs: object):
        start = int(kwargs["interval_start_ms"])
        end = int(kwargs["interval_end_ms"])
        calls.append((start, end))
        return [
            {
                "segment_id": f"segment-{start}",
                "start_ms": start,
                "end_ms": end,
                "raw_text": f"chunk {start}",
                "normalized_text": f"chunk {start}",
                "words": [],
            }
        ]

    adapter = ModelIndependentASRAdapter(recognize, name="shared-cache-fixture")
    shared = tmp_path / "shared"
    first = transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=5000,
        checkpoint_dir=tmp_path / "project-a" / "checkpoints",
        chunk_ms=3000,
        overlap_ms=1000,
        language="en",
        shared_cache_dir=shared,
    )
    assert calls == [(0, 3000), (2000, 5000)]
    assert first.metadata["shared_cache_hit_indexes"] == []
    assert len(list(shared.glob("*.json"))) == 2

    second = transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=5000,
        checkpoint_dir=tmp_path / "project-b" / "checkpoints",
        chunk_ms=3000,
        overlap_ms=1000,
        language="en",
        shared_cache_dir=shared,
    )
    assert calls == [(0, 3000), (2000, 5000)]
    assert second.metadata["processed_chunk_indexes"] == []
    assert second.metadata["resumed_chunk_indexes"] == [0, 1]
    assert second.metadata["shared_cache_hit_indexes"] == [0, 1]
    assert len(list((tmp_path / "project-b" / "checkpoints").glob("*.json"))) == 2


def test_checkpointed_asr_json_is_compact_for_local_and_shared_materialization(
    tmp_path: Path,
) -> None:
    """Compact checkpoints reduce SSD amplification without changing payloads."""

    media = tmp_path / "media.bin"
    media.write_bytes(b"compact-asr-media")
    shared = tmp_path / "shared"

    adapter = ModelIndependentASRAdapter(
        lambda _path, **kwargs: [
            {
                "start_ms": int(kwargs["interval_start_ms"]),
                "end_ms": int(kwargs["interval_end_ms"]),
                "raw_text": "Mabuhay, kumusta?",
                "normalized_text": "Mabuhay, kumusta?",
                "words": [],
            }
        ],
        name="compact-cache-fixture",
    )
    first = transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=1000,
        checkpoint_dir=tmp_path / "project-a" / "checkpoints",
        chunk_ms=1000,
        overlap_ms=0,
        shared_cache_dir=shared,
    )

    local_checkpoint = next((tmp_path / "project-a" / "checkpoints").glob("*.json"))
    shared_checkpoint = next(shared.glob("*.json"))
    assert local_checkpoint.read_text(encoding="utf-8").count("\n") == 1
    assert shared_checkpoint.read_text(encoding="utf-8").count("\n") == 1

    second = transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=1000,
        checkpoint_dir=tmp_path / "project-b" / "checkpoints",
        chunk_ms=1000,
        overlap_ms=0,
        shared_cache_dir=shared,
    )
    materialized = next((tmp_path / "project-b" / "checkpoints").glob("*.json"))
    assert materialized.read_text(encoding="utf-8").count("\n") == 1
    assert second.metadata["shared_cache_hit_indexes"] == [0]
    assert second.segments == first.segments


def test_shared_asr_cache_budget_skips_oversized_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"shared-cache-budget-media")
    shared = tmp_path / "shared"
    monkeypatch.setenv("VSR_ASR_SHARED_CACHE_MAX_BYTES", "1")
    adapter = ModelIndependentASRAdapter(
        lambda _path, **kwargs: [
            {
                "start_ms": int(kwargs["interval_start_ms"]),
                "end_ms": int(kwargs["interval_end_ms"]),
                "raw_text": "chunk",
                "words": [],
            }
        ],
        name="shared-budget-fixture",
    )
    transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=1000,
        checkpoint_dir=tmp_path / "project" / "checkpoints",
        chunk_ms=1000,
        overlap_ms=0,
        shared_cache_dir=shared,
    )
    assert not list(shared.glob("*.json"))


def test_shared_asr_cache_pruning_reconciles_periodically_not_per_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm shared caches amortize directory scans across chunk writes."""

    root = tmp_path / "shared"
    root.mkdir()
    current = root / "chunk.json"
    current.write_bytes(b"seed")
    original_glob = Path.glob
    glob_calls = 0

    def counted_glob(path: Path, pattern: str):
        nonlocal glob_calls
        if path == root and pattern == "*.json":
            glob_calls += 1
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", counted_glob)
    monkeypatch.setattr(whisper_adapter_module, "_ASR_PRUNE_INTERVAL", 4)
    for index in range(7):
        current.write_bytes(f"chunk-{index}".encode())
        whisper_adapter_module._prune_shared_asr_cache(
            root,
            current_path=current,
            cache_limit=1024 * 1024,
        )

    # One initial inventory plus a periodic reconciliation after the fourth
    # ledger update; seven writes therefore need only two scans, rather than
    # seven.
    assert glob_calls == 2


def test_checkpointed_chunks_propagate_high_confidence_language_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"language-hint-media")
    requested_languages: list[str | None] = []

    def recognize(path: Path, **kwargs: object) -> ASRResult:
        requested_languages.append(
            str(kwargs["language"]) if kwargs.get("language") is not None else None
        )
        start = int(kwargs["interval_start_ms"])
        end = int(kwargs["interval_end_ms"])
        return ASRResult(
            [
                {
                    "segment_id": f"segment-{start}",
                    "start_ms": start,
                    "end_ms": end,
                    "raw_text": "Mabuhay",
                    "normalized_text": "Mabuhay",
                    "words": [],
                }
            ],
            language="tl",
            language_probability=0.98,
        )

    adapter = ModelIndependentASRAdapter(recognize, name="language-hint-fixture")
    monkeypatch.setenv("VSR_ASR_LANGUAGE_HINT", "1")
    result = transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=6000,
        checkpoint_dir=tmp_path / "checkpoints",
        chunk_ms=2000,
        overlap_ms=0,
    )

    assert requested_languages == [None, "tl", "tl"]
    assert result.metadata["language_strategy"] == "first_chunk_high_confidence_hint"
    assert result.language == "tl"

    resumed = transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=6000,
        checkpoint_dir=tmp_path / "checkpoints",
        chunk_ms=2000,
        overlap_ms=0,
    )
    assert requested_languages == [None, "tl", "tl"]
    assert resumed.metadata["language_strategy"] == "first_chunk_high_confidence_hint"


def test_checkpointed_chunks_keep_language_detection_per_chunk_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"language-default-media")
    requested_languages: list[str | None] = []

    def recognize(path: Path, **kwargs: object) -> ASRResult:
        requested_languages.append(
            str(kwargs["language"]) if kwargs.get("language") is not None else None
        )
        start = int(kwargs["interval_start_ms"])
        end = int(kwargs["interval_end_ms"])
        return ASRResult(
            [{"start_ms": start, "end_ms": end, "raw_text": "Mabuhay", "words": []}],
            language="tl",
            language_probability=0.98,
        )

    monkeypatch.delenv("VSR_ASR_LANGUAGE_HINT", raising=False)
    result = transcribe_checkpointed_chunks(
        ModelIndependentASRAdapter(recognize, name="language-default-fixture"),
        media,
        duration_ms=6000,
        checkpoint_dir=tmp_path / "checkpoints",
        chunk_ms=2000,
        overlap_ms=0,
    )

    assert requested_languages == [None, None, None]
    assert result.metadata["language_strategy"] == "per_chunk_detection"


def test_checkpoint_cache_key_reuses_precomputed_media_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"media bytes")
    adapter = ModelIndependentASRAdapter(lambda _path, **_kwargs: [], name="digest-fixture")

    def fail_rehash(_path: Path) -> str:
        raise AssertionError("checkpoint cache key unexpectedly rehashed the media")

    monkeypatch.setattr(whisper_adapter_module, "sha256_file", fail_rehash)
    key = checkpoint_cache_key(
        adapter,
        media,
        interval_start_ms=0,
        interval_end_ms=1_000,
        chunk_ms=1_000,
        overlap_ms=0,
        language="fil",
        media_sha256="a" * 64,
    )
    assert len(key) == 64


def test_language_hint_policy_is_part_of_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"media bytes")
    adapter = ModelIndependentASRAdapter(lambda _path, **_kwargs: [], name="policy-fixture")
    monkeypatch.delenv("VSR_ASR_LANGUAGE_HINT", raising=False)
    off = checkpoint_cache_key(
        adapter,
        media,
        interval_start_ms=0,
        interval_end_ms=1_000,
        chunk_ms=1_000,
        overlap_ms=0,
        language=None,
        media_sha256="a" * 64,
    )
    monkeypatch.setenv("VSR_ASR_LANGUAGE_HINT", "1")
    on = checkpoint_cache_key(
        adapter,
        media,
        interval_start_ms=0,
        interval_end_ms=1_000,
        chunk_ms=1_000,
        overlap_ms=0,
        language=None,
        media_sha256="a" * 64,
    )
    assert off != on


def test_checkpointed_chunks_reports_progress_without_affecting_resume(tmp_path: Path) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"progress-media-content")
    events: list[dict[str, object]] = []

    def recognize(path: Path, **kwargs: object):
        start = int(kwargs["interval_start_ms"])
        end = int(kwargs["interval_end_ms"])
        return [
            {
                "segment_id": f"segment-{start}",
                "start_ms": start,
                "end_ms": end,
                "raw_text": f"chunk {start}",
                "normalized_text": f"chunk {start}",
                "words": [],
            }
        ]

    adapter = ModelIndependentASRAdapter(recognize, name="progress-fixture")
    result = transcribe_checkpointed_chunks(
        adapter,
        media,
        duration_ms=5000,
        checkpoint_dir=tmp_path / "checkpoints",
        chunk_ms=3000,
        overlap_ms=1000,
        language="en",
        progress_callback=lambda payload: events.append(dict(payload)),
    )

    assert [event["event"] for event in events] == [
        "chunk_started",
        "chunk_completed",
        "chunk_started",
        "chunk_completed",
        "completed",
    ]
    assert events[-1]["fraction"] == 1.0
    progress = result.metadata["progress"]
    assert progress["total_chunks"] == 2
    assert len(progress["chunk_timings"]) == 2


def test_checkpointed_chunks_emits_heartbeat_during_native_call(tmp_path: Path) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"heartbeat-media-content")
    events: list[dict[str, object]] = []

    def recognize(path: Path, **kwargs: object):
        time.sleep(0.08)
        start = int(kwargs["interval_start_ms"])
        end = int(kwargs["interval_end_ms"])
        return [
            {
                "segment_id": f"segment-{start}",
                "start_ms": start,
                "end_ms": end,
                "raw_text": f"chunk {start}",
                "normalized_text": f"chunk {start}",
                "words": [],
            }
        ]

    transcribe_checkpointed_chunks(
        ModelIndependentASRAdapter(recognize, name="heartbeat-fixture"),
        media,
        duration_ms=1000,
        checkpoint_dir=tmp_path / "checkpoints",
        chunk_ms=1000,
        overlap_ms=0,
        progress_callback=lambda payload: events.append(dict(payload)),
        progress_heartbeat_seconds=0.01,
    )

    heartbeats = [event for event in events if event["event"] == "chunk_heartbeat"]
    assert heartbeats
    assert all(event["chunk_index"] == 0 for event in heartbeats)
    assert all(float(event["elapsed_seconds"]) >= 0 for event in heartbeats)
    assert events[0]["event"] == "chunk_started"
    assert events[-1]["event"] == "completed"


def test_checkpointed_chunks_rejects_invalid_heartbeat_interval(tmp_path: Path) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"heartbeat-validation")
    adapter = ModelIndependentASRAdapter(lambda *_args, **_kwargs: [], name="fixture")

    with pytest.raises(ASRError, match="heartbeat interval"):
        transcribe_checkpointed_chunks(
            adapter,
            media,
            duration_ms=1000,
            checkpoint_dir=tmp_path / "checkpoints",
            progress_heartbeat_seconds=-1,
        )
