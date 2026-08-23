from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import video_script_reconstructor.pipeline as pipeline_module
from video_script_reconstructor.errors import BlockedError, ValidationFailure
from video_script_reconstructor.ocr import (
    OCRAdapter,
    OCRObservation,
    OCRToken,
    deserialize_observation,
    serialize_observation,
)
from video_script_reconstructor.pipeline import (
    _load_or_run_ocr,
    _ocr_checkpoint_flush_interval,
    _ocr_workers,
    _paddle_ocr_batch_workers,
    _partition_round_robin,
    _prune_shared_json_cache,
    _restore_ocr_cache,
    _StreamingOCRPrefetch,
    _write_ocr_cache,
)


def test_streaming_ocr_prefetch_drains_bounded_batches(tmp_path: Path) -> None:
    adapter = BatchOCRAdapter()
    frames = []
    for index in range(3):
        path = tmp_path / f"frame-{index}.png"
        path.write_bytes(f"frame-{index}".encode())
        frames.append(SimpleNamespace(frame_id=f"F{index + 1:06d}", path=path))

    prefetch = _StreamingOCRPrefetch(adapter, batch_size=2)
    for frame in frames:
        prefetch.submit(frame)
    prefetch.finish()

    assert adapter.batch_sizes == [2, 1]
    assert set(prefetch.observations) == {"F000001", "F000002", "F000003"}
    assert prefetch.metrics == {
        "submitted_count": 3,
        "recognized_count": 3,
        "batch_count": 2,
        "worker_count": 1,
        "recognize_seconds_total": prefetch.metrics["recognize_seconds_total"],
        "error": None,
    }
    assert prefetch.metrics["recognize_seconds_total"] >= 0.0


def test_streaming_ocr_prefetch_fanout_is_bounded_and_complete(tmp_path: Path) -> None:
    adapter = SpawnableBatchOCRAdapter()
    frames = []
    for index in range(5):
        path = tmp_path / f"fanout-{index}.png"
        path.write_bytes(f"frame-{index}".encode())
        frames.append(SimpleNamespace(frame_id=f"F{index + 1:06d}", path=path))

    prefetch = _StreamingOCRPrefetch(adapter, batch_size=2, worker_count=2)
    for frame in frames:
        prefetch.submit(frame)
    prefetch.finish()

    assert set(prefetch.observations) == {frame.frame_id for frame in frames}
    assert sorted(adapter.batch_sizes) == [1, 2, 2]
    assert prefetch.metrics == {
        "submitted_count": 5,
        "recognized_count": 5,
        "batch_count": 3,
        "worker_count": 2,
        "recognize_seconds_total": prefetch.metrics["recognize_seconds_total"],
        "error": None,
    }
    assert prefetch.metrics["recognize_seconds_total"] >= 0.0


def test_streaming_ocr_prefetch_failure_is_optional_and_non_blocking(tmp_path: Path) -> None:
    class FailingBatchAdapter(BatchOCRAdapter):
        def recognize_many(
            self,
            images: list[Path],
            *,
            frame_ids: list[str],
            observation_ids: list[str],
            language: str | None = None,
        ) -> dict[str, OCRObservation]:
            raise RuntimeError("fixture streaming failure")

    path = tmp_path / "frame.png"
    path.write_bytes(b"frame")
    prefetch = _StreamingOCRPrefetch(FailingBatchAdapter(), batch_size=1)
    prefetch.submit(SimpleNamespace(frame_id="F000001", path=path))
    prefetch.submit(SimpleNamespace(frame_id="F000002", path=path))
    prefetch.finish()

    assert prefetch.observations == {}
    assert prefetch.metrics["error"] == "fixture streaming failure"


def test_streaming_ocr_prefetch_fanout_failure_is_bounded_and_records_clear_error(
    tmp_path: Path,
) -> None:
    """A failing fan-out batch stops dispatch, reports one clear error, and
    never records partial or cross-frame canonical references."""

    class FanoutFailingAdapter(SpawnableBatchOCRAdapter):
        def recognize_many(
            self,
            images: list[Path],
            *,
            frame_ids: list[str],
            observation_ids: list[str],
            language: str | None = None,
        ) -> dict[str, OCRObservation]:
            self.batch_sizes.append(len(images))
            self.calls.extend(str(image) for image in images)
            if len(self.batch_sizes) > 1:
                raise RuntimeError("fixture fan-out failure")
            return {
                frame_id: _observation(frame_id, observation_id, image.stem)
                for image, frame_id, observation_id in zip(
                    images, frame_ids, observation_ids, strict=True
                )
            }

    frames = []
    for index in range(4):
        path = tmp_path / f"fanout-{index}.png"
        path.write_bytes(f"frame-{index}".encode())
        frames.append(SimpleNamespace(frame_id=f"F{index + 1:06d}", path=path))

    prefetch = _StreamingOCRPrefetch(FanoutFailingAdapter(), batch_size=1, worker_count=2)
    for frame in frames:
        prefetch.submit(frame)
    prefetch.finish()

    assert prefetch.metrics["error"] == "fixture fan-out failure"
    assert prefetch.metrics["worker_count"] == 2
    # The failed batch and every batch after it contribute nothing; only
    # complete batches may surface, each attributed to its own frame.
    assert "F000004" not in prefetch.observations
    expected_text = {
        frame.frame_id: Path(frame.path).stem
        for frame in frames
        if frame.frame_id in prefetch.observations
    }
    assert set(prefetch.observations) == set(expected_text)
    for frame_id, observation in prefetch.observations.items():
        assert observation.frame_id == frame_id
        assert observation.normalized_interpretation == expected_text[frame_id]


def test_paddle_ocr_batch_worker_policy_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VSR_PADDLE_OCR_WORKERS", raising=False)
    assert _paddle_ocr_batch_workers() == 1
    monkeypatch.setenv("VSR_PADDLE_OCR_WORKERS", "999")
    assert _paddle_ocr_batch_workers() == 2
    monkeypatch.setenv("VSR_PADDLE_OCR_WORKERS", "invalid")
    assert _paddle_ocr_batch_workers() == 1
    # Adapters without an independent worker factory retain the sequential path.
    monkeypatch.setenv("VSR_PADDLE_OCR_WORKERS", "4")
    assert _paddle_ocr_batch_workers(BatchOCRAdapter()) == 1
    assert _paddle_ocr_batch_workers(SpawnableBatchOCRAdapter()) == 2


def test_partition_round_robin_is_deterministic_and_total() -> None:
    items = [10, 11, 12, 13, 14, 15, 16]
    shards = _partition_round_robin(items, 3)
    assert shards == [[10, 13, 16], [11, 14], [12, 15]]
    assert sorted(item for shard in shards for item in shard) == items
    assert _partition_round_robin(items, 1) == [items]
    assert _partition_round_robin([], 4) == [[], [], [], []]
    assert _partition_round_robin(items, 0) == [items]


def test_ocr_checkpoint_accepts_prefetched_observations(tmp_path: Path) -> None:
    frames = _frames(tmp_path, ["a", "b"])
    prefetched = {
        "F000001": _observation("F000001", "P000001", "prefetched-a"),
        "F000002": _observation("F000002", "P000002", "prefetched-b"),
    }
    adapter = BatchOCRAdapter()
    result, metrics = _load_or_run_ocr(
        tmp_path / "source.mp4",
        tmp_path,
        frames,
        adapter=adapter,
        adapter_key="fixture-prefetch",
        source_sha256="source-digest",
        prefetched_by_frame_id=prefetched,
        prefetch_metrics={"submitted_count": 2, "recognized_count": 2, "batch_count": 1},
    )

    assert adapter.batch_sizes == []
    assert metrics["prefetch_hit_count"] == 2
    assert metrics["prefetch_batch_count"] == 1
    assert metrics["prefetch_recognized_count"] == 2
    assert result["F000001"].normalized_interpretation == "prefetched-a"
    assert result["F000001"].observation_id == "O000001"


def _observation(frame_id: str, observation_id: str, text: str) -> OCRObservation:
    return OCRObservation(
        observation_id=observation_id,
        frame_id=frame_id,
        crop_id=None,
        bounding_region=(1, 2, 30, 40),
        raw_engine_text=f" {text}\n",
        normalized_interpretation=text,
        confidence=0.91,
        alternatives=({"text": text, "confidence": 0.91},),
        language="eng",
        uncertain_characters=({"text": "?", "reason": "fixture"},),
        engine="fixture-ocr",
        engine_version="1",
        human_decision=None,
        tokens=(OCRToken(text, 91.0, (1, 2, 30, 40), 1, 1, 1, 1, 1),),
    )


class CountingOCRAdapter(OCRAdapter):
    def __init__(self, *, empty: bool = False) -> None:
        self.calls: list[str] = []
        self.empty = empty

    def available(self) -> bool:
        return True

    def recognize(
        self,
        image_path: str | Path,
        *,
        frame_id: str,
        observation_id: str,
        crop_id: str | None = None,
        language: str | None = None,
    ) -> OCRObservation:
        self.calls.append(str(image_path))
        if self.empty:
            return _observation(frame_id, observation_id, "")
        return _observation(frame_id, observation_id, Path(image_path).stem)


class BatchOCRAdapter(CountingOCRAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        # Mirror the production adapter's instrumentation contract so pipeline
        # metric aggregation is exercised end to end.
        self.batch_roundtrip_seconds_total = 0.0
        self.batch_roundtrip_count = 0

    def recognize_many(
        self,
        images: list[Path],
        *,
        frame_ids: list[str],
        observation_ids: list[str],
        language: str | None = None,
    ) -> dict[str, OCRObservation]:
        started = time.perf_counter()
        self.batch_sizes.append(len(images))
        self.calls.extend(str(image) for image in images)
        results = {
            frame_id: _observation(frame_id, observation_id, image.stem)
            for image, frame_id, observation_id in zip(
                images, frame_ids, observation_ids, strict=True
            )
        }
        self.batch_roundtrip_seconds_total += time.perf_counter() - started
        self.batch_roundtrip_count += 1
        return results


class SpawnableBatchOCRAdapter(BatchOCRAdapter):
    def __init__(self, *, shared: SpawnableBatchOCRAdapter | None = None) -> None:
        super().__init__()
        if shared is not None:
            self.batch_sizes = shared.batch_sizes
            self.calls = shared.calls
        self.closed = False

    def available(self) -> bool:
        return True

    def spawn_worker(self) -> SpawnableBatchOCRAdapter:
        return type(self)(shared=self)

    def close(self) -> None:
        self.closed = True


class FailingOCRAdapter(CountingOCRAdapter):
    def __init__(self, fail_frame_id: str) -> None:
        super().__init__()
        self.fail_frame_id = fail_frame_id

    def recognize(
        self,
        image_path: str | Path,
        *,
        frame_id: str,
        observation_id: str,
        crop_id: str | None = None,
        language: str | None = None,
    ) -> OCRObservation:
        if frame_id == self.fail_frame_id:
            self.calls.append(str(image_path))
            raise RuntimeError("fixture OCR worker failure")
        return super().recognize(
            image_path,
            frame_id=frame_id,
            observation_id=observation_id,
            crop_id=crop_id,
            language=language,
        )


def _frames(project: Path, values: list[str]) -> list[dict[str, object]]:
    frame_dir = project / "evidence" / "full"
    frame_dir.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, object]] = []
    for index, value in enumerate(values, 1):
        path = frame_dir / f"frame-{value}.png"
        path.write_bytes(value.encode())
        result.append(
            {
                "frame_id": f"F{index:06d}",
                "full_frame_path": path.relative_to(project).as_posix(),
                "pixel_hash": {"algorithm": "fixture", "value": f"pixel-{value}"},
            }
        )
    return result


def test_ocr_observation_checkpoint_round_trips_complete_tokens() -> None:
    original = _observation("F000001", "O000001", "VISIBLE")
    restored = deserialize_observation(serialize_observation(original))
    assert restored == original


def test_ocr_checkpoint_budget_matches_compact_persisted_json(tmp_path: Path) -> None:
    """A payload accepted by the byte budget must be restorable at that budget."""

    observation = _observation("F000001", "O000001", "VISIBLE")
    checkpoint = tmp_path / "ocr.json"
    # Use the compact envelope size as the exact budget.  Before the compact
    # write, the same payload was pretty-printed and exceeded this limit
    # despite _write_ocr_cache accepting it.
    compact_payload = {
        "schema_version": "1.0",
        "cache_key": "key",
        "source_sha256": "source",
        "adapter_identity": "adapter",
        "entries": [
            {
                "pixel_hash": "pixel-hash",
                "observation": serialize_observation(observation),
            }
        ],
    }
    cache_limit = len(
        (
            json.dumps(
                compact_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )
    assert _write_ocr_cache(
        checkpoint,
        cache_key_value="key",
        source_digest="source",
        adapter_identity="adapter",
        entries={"pixel-hash": observation},
        cache_limit=cache_limit,
    )
    assert checkpoint.stat().st_size <= cache_limit
    restored = _restore_ocr_cache(
        checkpoint,
        cache_key_value="key",
        source_digest="source",
        adapter_identity="adapter",
        cache_limit=cache_limit,
    )
    assert restored is not None
    assert restored["pixel-hash"] == observation


def test_ocr_checkpoint_reuses_results_and_remaps_current_ids(tmp_path: Path) -> None:
    first_frames = _frames(tmp_path, ["a", "b"])
    first_adapter = CountingOCRAdapter()
    first, first_metrics = _load_or_run_ocr(
        tmp_path / "source.mp4",
        tmp_path,
        first_frames,
        adapter=first_adapter,
        adapter_key="fixture",
        source_sha256="source-digest",
    )
    assert len(first_adapter.calls) == 2
    assert first_metrics["cache_miss_count"] == 2
    assert first["F000001"].normalized_interpretation == "frame-a"

    second_frames = _frames(tmp_path, ["b", "a"])
    second_frames[0]["frame_id"] = "F000101"
    second_frames[1]["frame_id"] = "F000102"
    second_adapter = CountingOCRAdapter()
    second, second_metrics = _load_or_run_ocr(
        tmp_path / "source.mp4",
        tmp_path,
        second_frames,
        adapter=second_adapter,
        adapter_key="fixture",
        source_sha256="source-digest",
    )
    assert second_adapter.calls == []
    assert second_metrics["cache_hit_count"] == 2
    assert second["F000101"].frame_id == "F000101"
    assert second["F000101"].observation_id == "O000001"
    assert second["F000101"].normalized_interpretation == "frame-b"
    assert second["F000102"].observation_id == "O000002"


def test_ocr_checkpoint_reuses_results_from_shared_cache(tmp_path: Path) -> None:
    shared = tmp_path / "shared-visual"
    first_project = tmp_path / "project-a"
    first_frames = _frames(first_project, ["a", "b"])
    first_adapter = CountingOCRAdapter()
    _load_or_run_ocr(
        tmp_path / "source.mp4",
        first_project,
        first_frames,
        adapter=first_adapter,
        adapter_key="fixture-shared",
        source_sha256="source-digest",
        shared_cache_dir=shared,
    )
    assert len(first_adapter.calls) == 2

    second_project = tmp_path / "project-b"
    second_frames = _frames(second_project, ["a", "b"])
    second_adapter = CountingOCRAdapter()
    second, metrics = _load_or_run_ocr(
        tmp_path / "source.mp4",
        second_project,
        second_frames,
        adapter=second_adapter,
        adapter_key="fixture-shared",
        source_sha256="source-digest",
        shared_cache_dir=shared,
    )
    assert second_adapter.calls == []
    assert metrics["shared_cache_hit"] is True
    assert metrics["cache_hit_count"] == 2
    assert second["F000001"].normalized_interpretation == "frame-a"
    assert list((shared / "ocr").glob("*.json"))


def test_ocr_shared_warm_hit_parses_checkpoint_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validated shared OCR payloads are reused for local materialization."""

    shared = tmp_path / "shared-visual"
    first_project = tmp_path / "project-a"
    frames = _frames(first_project, ["a", "b"])
    _load_or_run_ocr(
        tmp_path / "source.mp4",
        first_project,
        frames,
        adapter=CountingOCRAdapter(),
        adapter_key="fixture-warm-parse",
        source_sha256="source-digest",
        shared_cache_dir=shared,
    )

    original_loads = pipeline_module.json.loads
    load_count = 0

    def counted_loads(value: str | bytes | bytearray, *args: object, **kwargs: object):
        nonlocal load_count
        load_count += 1
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(pipeline_module.json, "loads", counted_loads)
    second_project = tmp_path / "project-b"
    second, metrics = _load_or_run_ocr(
        tmp_path / "source.mp4",
        second_project,
        _frames(second_project, ["a", "b"]),
        adapter=CountingOCRAdapter(),
        adapter_key="fixture-warm-parse",
        source_sha256="source-digest",
        shared_cache_dir=shared,
    )

    assert metrics["shared_cache_hit"] is True
    assert second["F000001"].normalized_interpretation == "frame-a"
    # A warm shared hit needs one JSON decode for validation; the same parsed
    # envelope is now used for local checkpoint materialization.
    assert load_count == 1


def test_ocr_local_warm_hit_does_not_rewrite_complete_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local complete hit avoids redundant checkpoint and shared writes."""

    project = tmp_path / "project"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    _load_or_run_ocr(
        source,
        project,
        _frames(project, ["a", "b"]),
        adapter=CountingOCRAdapter(),
        adapter_key="fixture-warm-local",
        source_sha256="source-digest",
    )
    writes: list[Path] = []
    original_write = pipeline_module._write_ocr_cache

    def counted_write(path: Path, **kwargs: object) -> bool:
        writes.append(path)
        return original_write(path, **kwargs)

    monkeypatch.setattr(pipeline_module, "_write_ocr_cache", counted_write)
    _result, metrics = _load_or_run_ocr(
        source,
        project,
        _frames(project, ["a", "b"]),
        adapter=CountingOCRAdapter(),
        adapter_key="fixture-warm-local",
        source_sha256="source-digest",
    )

    assert writes == []
    assert metrics["cache_hit_count"] == 2
    assert metrics["cache_miss_count"] == 0
    assert metrics["cache_reused_without_rewrite"] is True
    assert metrics["cache_written"] is False


def test_shared_json_cache_pruning_reconciles_periodically_not_per_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm OCR/survey caches amortize directory scans across receipt writes."""

    root = tmp_path / "shared-json"
    root.mkdir()
    current = root / "receipt.json"
    current.write_bytes(b"seed")
    original_glob = Path.glob
    glob_calls = 0

    def counted_glob(path: Path, pattern: str):
        nonlocal glob_calls
        if path == root and pattern == "*.json":
            glob_calls += 1
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", counted_glob)
    monkeypatch.setattr(pipeline_module, "_SHARED_JSON_PRUNE_INTERVAL", 4)
    for index in range(7):
        current.write_bytes(f"receipt-{index}".encode())
        _prune_shared_json_cache(
            root,
            current_path=current,
            cache_limit=1024 * 1024,
        )

    # One initial inventory plus a periodic reconciliation after the fourth
    # ledger update; seven writes therefore need only two scans, rather than
    # seven.
    assert glob_calls == 2


def test_ocr_shared_warm_hit_materializes_once_without_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shared complete hit materializes local state but skips final rewrites."""

    shared = tmp_path / "shared"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    first = tmp_path / "first"
    _load_or_run_ocr(
        source,
        first,
        _frames(first, ["a", "b"]),
        adapter=CountingOCRAdapter(),
        adapter_key="fixture-warm-shared",
        source_sha256="source-digest",
        shared_cache_dir=shared,
    )
    writes: list[Path] = []
    original_write = pipeline_module._write_ocr_cache

    def counted_write(path: Path, **kwargs: object) -> bool:
        writes.append(path)
        return original_write(path, **kwargs)

    monkeypatch.setattr(pipeline_module, "_write_ocr_cache", counted_write)
    second = tmp_path / "second"
    _result, metrics = _load_or_run_ocr(
        source,
        second,
        _frames(second, ["a", "b"]),
        adapter=CountingOCRAdapter(),
        adapter_key="fixture-warm-shared",
        source_sha256="source-digest",
        shared_cache_dir=shared,
    )

    assert writes == []
    assert metrics["shared_cache_hit"] is True
    assert metrics["cache_hit_count"] == 2
    assert metrics["cache_miss_count"] == 0
    assert metrics["cache_reused_without_rewrite"] is True
    assert metrics["cache_written"] is False
    assert list((second / ".state" / "checkpoints" / "ocr").glob("*.json"))


def test_ocr_local_warm_hit_backfills_evicted_shared_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing optional shared receipt is restored without local rewrites."""

    shared = tmp_path / "shared"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    frames = _frames(project, ["a", "b"])
    _load_or_run_ocr(
        source,
        project,
        frames,
        adapter=CountingOCRAdapter(),
        adapter_key="fixture-warm-backfill",
        source_sha256="source-digest",
        shared_cache_dir=shared,
    )
    for path in shared.rglob("*.json"):
        path.unlink()
    writes: list[Path] = []
    original_write = pipeline_module._write_ocr_cache

    def counted_write(path: Path, **kwargs: object) -> bool:
        writes.append(path)
        return original_write(path, **kwargs)

    monkeypatch.setattr(pipeline_module, "_write_ocr_cache", counted_write)
    _result, metrics = _load_or_run_ocr(
        source,
        project,
        _frames(project, ["a", "b"]),
        adapter=CountingOCRAdapter(),
        adapter_key="fixture-warm-backfill",
        source_sha256="source-digest",
        shared_cache_dir=shared,
    )

    assert len(writes) == 1
    assert writes[0].parent == shared / "ocr"
    assert metrics["cache_written"] is False
    assert metrics["shared_cache_written"] is True


def test_ocr_new_pixel_writes_partial_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new pixel keeps the durable write path instead of using warm-hit skip."""

    project = tmp_path / "project"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    _load_or_run_ocr(
        source,
        project,
        _frames(project, ["a", "b"]),
        adapter=CountingOCRAdapter(),
        adapter_key="fixture-partial-hit",
        source_sha256="source-digest",
    )
    writes: list[Path] = []
    original_write = pipeline_module._write_ocr_cache

    def counted_write(path: Path, **kwargs: object) -> bool:
        writes.append(path)
        return original_write(path, **kwargs)

    monkeypatch.setattr(pipeline_module, "_write_ocr_cache", counted_write)
    _result, metrics = _load_or_run_ocr(
        source,
        project,
        _frames(project, ["a", "c"]),
        adapter=CountingOCRAdapter(),
        adapter_key="fixture-partial-hit",
        source_sha256="source-digest",
    )

    assert len(writes) == 1
    assert metrics["cache_hit_count"] == 1
    assert metrics["cache_miss_count"] == 1
    assert metrics["cache_reused_without_rewrite"] is False
    assert metrics["cache_written"] is True


def test_ocr_checkpoint_processes_only_new_pixels_and_caches_blank_results(tmp_path: Path) -> None:
    first_frames = _frames(tmp_path, ["a", "b"])
    first_adapter = CountingOCRAdapter(empty=True)
    _load_or_run_ocr(
        tmp_path / "source.mp4",
        tmp_path,
        first_frames,
        adapter=first_adapter,
        adapter_key="fixture-empty",
        source_sha256="source-digest",
    )
    assert len(first_adapter.calls) == 2

    second_frames = _frames(tmp_path, ["a", "c"])
    second_adapter = CountingOCRAdapter(empty=True)
    _result, metrics = _load_or_run_ocr(
        tmp_path / "source.mp4",
        tmp_path,
        second_frames,
        adapter=second_adapter,
        adapter_key="fixture-empty",
        source_sha256="source-digest",
    )
    assert len(second_adapter.calls) == 1
    assert metrics["cache_hit_count"] == 1
    assert metrics["cache_miss_count"] == 1


def test_ocr_batch_checkpoint_skips_full_batch_when_all_pixels_are_cached(tmp_path: Path) -> None:
    frames = _frames(tmp_path, ["a", "b"])
    first_adapter = BatchOCRAdapter()
    _load_or_run_ocr(
        tmp_path / "source.mp4",
        tmp_path,
        frames,
        adapter=first_adapter,
        adapter_key="fixture-batch",
        source_sha256="source-digest",
    )
    assert len(first_adapter.calls) == 2

    second_adapter = BatchOCRAdapter()
    _load_or_run_ocr(
        tmp_path / "source.mp4",
        tmp_path,
        frames,
        adapter=second_adapter,
        adapter_key="fixture-batch",
        source_sha256="source-digest",
    )
    assert second_adapter.calls == []


def test_ocr_batch_adapter_chunks_long_runs_and_checkpoints_each_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VSR_OCR_BATCH_SIZE", "2")
    frames = _frames(tmp_path, ["a", "b", "c", "d", "e"])
    adapter = BatchOCRAdapter()
    result, metrics = _load_or_run_ocr(
        tmp_path / "source.mp4",
        tmp_path,
        frames,
        adapter=adapter,
        adapter_key="fixture-batch-chunks",
        source_sha256="source-digest",
    )
    assert adapter.batch_sizes == [2, 2, 1]
    assert metrics["batch_size"] == 2
    assert metrics["batch_count"] == 3
    assert metrics["checkpoint_flush_count"] == 3
    assert len(result) == 5


def test_ocr_checkpoint_flushes_completed_batch_before_worker_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VSR_OCR_WORKERS", "1")
    monkeypatch.setenv("VSR_OCR_CHECKPOINT_BATCH", "1")
    frames = _frames(tmp_path, ["a", "b", "c"])
    failing = FailingOCRAdapter("F000002")
    with pytest.raises(RuntimeError, match="fixture OCR worker failure"):
        _load_or_run_ocr(
            tmp_path / "source.mp4",
            tmp_path,
            frames,
            adapter=failing,
            adapter_key="fixture-incremental",
            source_sha256="source-digest",
        )

    retry = CountingOCRAdapter()
    result, metrics = _load_or_run_ocr(
        tmp_path / "source.mp4",
        tmp_path,
        frames,
        adapter=retry,
        adapter_key="fixture-incremental",
        source_sha256="source-digest",
    )
    assert [Path(path).stem for path in retry.calls] == ["frame-b", "frame-c"]
    assert metrics["cache_hit_count"] == 1
    assert metrics["cache_miss_count"] == 2
    assert result["F000001"].normalized_interpretation == "frame-a"


def test_corrupt_ocr_checkpoint_is_a_miss(tmp_path: Path) -> None:
    frames = _frames(tmp_path, ["a"])
    adapter = CountingOCRAdapter()
    _load_or_run_ocr(
        tmp_path / "source.mp4",
        tmp_path,
        frames,
        adapter=adapter,
        adapter_key="fixture-corrupt",
        source_sha256="source-digest",
    )
    checkpoint = next((tmp_path / ".state" / "checkpoints" / "ocr").glob("*.json"))
    checkpoint.write_text("{not-json", encoding="utf-8")
    retry_adapter = CountingOCRAdapter()
    _result, metrics = _load_or_run_ocr(
        tmp_path / "source.mp4",
        tmp_path,
        frames,
        adapter=retry_adapter,
        adapter_key="fixture-corrupt",
        source_sha256="source-digest",
    )
    assert len(retry_adapter.calls) == 1
    assert metrics["cache_hit_count"] == 0


def test_ocr_worker_policy_is_bounded_and_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VSR_OCR_WORKERS", "12")
    assert _ocr_workers() == 12
    monkeypatch.setenv("VSR_OCR_WORKERS", "999")
    assert _ocr_workers() == 16
    monkeypatch.setenv("VSR_OCR_WORKERS", "invalid")
    assert 1 <= _ocr_workers() <= 16


def test_ocr_default_pool_scales_only_on_large_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VSR_OCR_WORKERS", raising=False)
    monkeypatch.setattr("video_script_reconstructor.pipeline.os.cpu_count", lambda: 12)
    assert _ocr_workers() == 6
    monkeypatch.setattr("video_script_reconstructor.pipeline.os.cpu_count", lambda: 16)
    assert _ocr_workers() == 12


def test_ocr_checkpoint_flush_interval_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VSR_OCR_CHECKPOINT_BATCH", raising=False)
    assert _ocr_checkpoint_flush_interval() == 16
    monkeypatch.setenv("VSR_OCR_CHECKPOINT_BATCH", "0")
    assert _ocr_checkpoint_flush_interval() == 1
    monkeypatch.setenv("VSR_OCR_CHECKPOINT_BATCH", "999")
    assert _ocr_checkpoint_flush_interval() == 64
    monkeypatch.setenv("VSR_OCR_CHECKPOINT_BATCH", "invalid")
    assert _ocr_checkpoint_flush_interval() == 16


class ClosingSpawnableBatchOCRAdapter(OCRAdapter):
    """Track spawned children so ownership cleanup can be asserted."""

    def __init__(self, *, shared: ClosingSpawnableBatchOCRAdapter | None = None) -> None:
        self.batch_sizes: list[int] = [] if shared is None else shared.batch_sizes
        self.calls: list[str] = [] if shared is None else shared.calls
        self.children: list[ClosingSpawnableBatchOCRAdapter] = []
        if shared is not None:
            shared.children.append(self)
        self.closed = False

    def available(self) -> bool:
        return True

    def recognize(
        self,
        image_path: str | Path,
        *,
        frame_id: str,
        observation_id: str,
        crop_id: str | None = None,
        language: str | None = None,
    ) -> OCRObservation:
        return _observation(frame_id, observation_id, Path(image_path).stem)

    def recognize_many(
        self,
        images: list[Path],
        *,
        frame_ids: list[str],
        observation_ids: list[str],
        language: str | None = None,
    ) -> dict[str, OCRObservation]:
        self.batch_sizes.append(len(images))
        self.calls.extend(str(image) for image in images)
        return {
            frame_id: _observation(frame_id, observation_id, image.stem)
            for image, frame_id, observation_id in zip(
                images, frame_ids, observation_ids, strict=True
            )
        }

    def spawn_worker(self) -> ClosingSpawnableBatchOCRAdapter:
        return type(self)(shared=self)

    def close(self) -> None:
        self.closed = True


def test_sharded_paddle_batches_match_sequential_outputs_and_close_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opt-in sharding is a pure scheduling change: identical observations,
    deterministic per-frame IDs, and every owned worker closed afterwards."""

    monkeypatch.delenv("VSR_PADDLE_OCR_WORKERS", raising=False)
    values = ["a", "b", "c", "d", "e"]
    sequential_project = tmp_path / "sequential"
    sequential_result, _sequential_metrics = _load_or_run_ocr(
        tmp_path / "source.mp4",
        sequential_project,
        _frames(sequential_project, values),
        adapter=SpawnableBatchOCRAdapter(),
        adapter_key="fixture-shard",
        source_sha256="source-digest",
    )

    monkeypatch.setenv("VSR_PADDLE_OCR_WORKERS", "3")
    monkeypatch.setenv("VSR_OCR_BATCH_SIZE", "2")
    sharded_project = tmp_path / "sharded"
    adapter = ClosingSpawnableBatchOCRAdapter()
    result, metrics = _load_or_run_ocr(
        tmp_path / "source.mp4",
        sharded_project,
        _frames(sharded_project, values),
        adapter=adapter,
        adapter_key="fixture-shard",
        source_sha256="source-digest",
    )

    assert result == sequential_result
    assert metrics["worker_count"] == 2
    assert metrics["batch_count"] == 3
    assert metrics["checkpoint_flush_count"] == 3
    assert sorted(adapter.batch_sizes) == [1, 2, 2]
    for index, frame_id in enumerate(("F000001", "F000002", "F000003", "F000004", "F000005"), 1):
        assert result[frame_id].observation_id == f"O{index:06d}"
        assert result[frame_id].normalized_interpretation == f"frame-{values[index - 1]}"
    assert len(adapter.children) == 1
    assert all(child.closed for child in adapter.children)


def test_sharded_paddle_failure_propagates_and_closes_owned_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shard failure is never swallowed; owned workers are still closed and
    a retry resumes the remaining pixels through the normal checkpoint path."""

    class ShardFailingAdapter(ClosingSpawnableBatchOCRAdapter):
        def recognize_many(
            self,
            images: list[Path],
            *,
            frame_ids: list[str],
            observation_ids: list[str],
            language: str | None = None,
        ) -> dict[str, OCRObservation]:
            self.batch_sizes.append(len(images))
            self.calls.extend(str(image) for image in images)
            if "F000002" in frame_ids:
                raise RuntimeError("fixture shard failure")
            return {
                frame_id: _observation(frame_id, observation_id, image.stem)
                for image, frame_id, observation_id in zip(
                    images, frame_ids, observation_ids, strict=True
                )
            }

    monkeypatch.setenv("VSR_PADDLE_OCR_WORKERS", "3")
    monkeypatch.setenv("VSR_OCR_BATCH_SIZE", "1")
    frames = _frames(tmp_path, ["a", "b", "c", "d"])
    adapter = ShardFailingAdapter()
    with pytest.raises(RuntimeError, match="fixture shard failure"):
        _load_or_run_ocr(
            tmp_path / "source.mp4",
            tmp_path,
            frames,
            adapter=adapter,
            adapter_key="fixture-shard-failure",
            source_sha256="source-digest",
        )
    assert len(adapter.children) == 1
    assert all(child.closed for child in adapter.children)

    retry_adapter = SpawnableBatchOCRAdapter()
    retry_result, _retry_metrics = _load_or_run_ocr(
        tmp_path / "source.mp4",
        tmp_path,
        frames,
        adapter=retry_adapter,
        adapter_key="fixture-shard-failure",
        source_sha256="source-digest",
    )
    assert set(retry_result) == {"F000001", "F000002", "F000003", "F000004"}
    for index, frame_id in enumerate(("F000001", "F000002", "F000003", "F000004"), 1):
        assert retry_result[frame_id].observation_id == f"O{index:06d}"


def test_sharded_paddle_spawn_failure_closes_partial_workers_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If independent worker creation fails mid-fan-out, already-created
    workers are closed and the failure propagates to the caller."""

    class ExhaustedSpawnerAdapter(ClosingSpawnableBatchOCRAdapter):
        def __init__(self, *, shared: ExhaustedSpawnerAdapter | None = None) -> None:
            super().__init__(shared=shared)
            self.spawn_calls = 0

        def spawn_worker(self) -> ExhaustedSpawnerAdapter:
            self.spawn_calls += 1
            if self.spawn_calls > 1:
                raise BlockedError("fixture spawn exhaustion")
            return super().spawn_worker()

    monkeypatch.setenv("VSR_PADDLE_OCR_WORKERS", "3")
    monkeypatch.setenv("VSR_OCR_BATCH_SIZE", "1")
    monkeypatch.setattr(pipeline_module, "_paddle_ocr_batch_workers", lambda adapter=None: 3)
    frames = _frames(tmp_path, ["a", "b", "c"])
    adapter = ExhaustedSpawnerAdapter()
    with pytest.raises(BlockedError, match="fixture spawn exhaustion"):
        _load_or_run_ocr(
            tmp_path / "source.mp4",
            tmp_path,
            frames,
            adapter=adapter,
            adapter_key="fixture-spawn-failure",
            source_sha256="source-digest",
        )
    assert adapter.spawn_calls == 2
    assert len(adapter.children) == 1
    assert adapter.children[0].closed


def test_sharded_paddle_rejected_worker_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spawned worker that fails capability validation is not leaked."""

    class UnavailableSpawnerAdapter(ClosingSpawnableBatchOCRAdapter):
        def spawn_worker(self) -> UnavailableSpawnerAdapter:
            worker = super().spawn_worker()
            worker.available = lambda: False  # type: ignore[method-assign]
            return worker

    monkeypatch.setenv("VSR_PADDLE_OCR_WORKERS", "2")
    monkeypatch.setenv("VSR_OCR_BATCH_SIZE", "1")
    frames = _frames(tmp_path, ["a", "b"])
    adapter = UnavailableSpawnerAdapter()
    with pytest.raises(ValidationFailure, match="independent worker is unavailable"):
        _load_or_run_ocr(
            tmp_path / "source.mp4",
            tmp_path,
            frames,
            adapter=adapter,
            adapter_key="fixture-unavailable-worker",
            source_sha256="source-digest",
        )
    assert len(adapter.children) == 1
    assert adapter.children[0].closed
