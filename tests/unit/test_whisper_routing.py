from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from video_script_reconstructor.pipeline import (
    _asr_cpu_threads,
    _auto_asr_adapters,
    _faster_whisper_compute_type,
    _faster_whisper_model_identity,
    _faster_whisper_num_workers,
)
from video_script_reconstructor.whisper_adapter import FasterWhisperAdapter


def test_faster_whisper_retries_cpu_when_cuda_runtime_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "large-v3"
    model_dir.mkdir()
    calls: list[tuple[str, str]] = []

    class FakeWhisperModel:
        def __init__(self, path, *, device, compute_type, **kwargs):
            calls.append((str(device), str(compute_type)))
            if device == "cuda":
                raise RuntimeError("cublas64_12.dll is missing")

    fake_module = ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    adapter = FasterWhisperAdapter(
        model=model_dir,
        device="cuda",
        compute_type="float16",
        allow_cpu_fallback=True,
    )

    adapter._load_model()

    assert calls == [("cuda", "float16"), ("cpu", "int8")]
    assert adapter.device == "cpu"
    assert adapter.compute_type == "int8"
    assert adapter.load_diagnostic is not None
    assert "cublas64_12.dll" in adapter.load_diagnostic


def test_faster_whisper_can_disable_cpu_fallback(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "large-v3"
    model_dir.mkdir()

    class FakeWhisperModel:
        def __init__(self, path, **kwargs):
            raise RuntimeError("CUDA unavailable")

    fake_module = ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    adapter = FasterWhisperAdapter(
        model=model_dir,
        device="cuda",
        allow_cpu_fallback=False,
    )

    try:
        adapter._load_model()
    except Exception as exc:
        assert "Unable to load faster-whisper" in str(exc)
    else:
        raise AssertionError("expected CUDA load failure")


def test_faster_whisper_retries_cpu_when_decode_defers_cuda_failure(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")

    class DeferredCudaFailureModel:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, path, **settings):
            self.calls += 1
            if self.calls == 2:
                return iter(
                    [
                        type(
                            "Segment",
                            (),
                            {"start": 0.0, "end": 0.5, "text": " Filipino", "words": []},
                        )()
                    ]
                ), type("Info", (), {"language": "tl", "language_probability": 0.9})()
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    adapter = FasterWhisperAdapter(model="local", device="cuda", compute_type="float16")
    fake_model = DeferredCudaFailureModel()
    adapter._model = fake_model
    adapter._load_model = lambda: fake_model  # type: ignore[method-assign]
    result = adapter.transcribe(audio, language="fil")

    assert result[0].raw_text.strip() == "Filipino"
    assert adapter.device == "cpu"
    assert adapter.load_diagnostic is not None


def test_filipino_hint_places_whisper_before_qwen(monkeypatch) -> None:
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline._developer_worker_path",
        lambda *args, **kwargs: Path("missing-worker.exe"),
    )
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.importlib.util.find_spec",
        lambda name: SimpleNamespace() if name == "faster_whisper" else None,
    )
    monkeypatch.setattr(
        "video_script_reconstructor.model_store.verify_model",
        lambda name: {
            "offline_ready": name == "faster-whisper-large-v3",
            "directory": "C:/models/large-v3",
        },
    )

    # The worker/model probes are intentionally patched at their import sites;
    # this test only asserts the language-aware ordering contract.
    adapters = _auto_asr_adapters(language="fil")
    assert [adapter.backend_name for adapter in adapters] == ["faster-whisper"]


def test_auto_whisper_receives_bounded_cpu_thread_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline._developer_worker_path",
        lambda *args, **kwargs: Path("missing-worker.exe"),
    )
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.importlib.util.find_spec",
        lambda name: SimpleNamespace() if name == "faster_whisper" else None,
    )
    monkeypatch.setattr("video_script_reconstructor.pipeline.shutil.which", lambda name: None)
    monkeypatch.setattr("video_script_reconstructor.pipeline.os.cpu_count", lambda: 16)
    monkeypatch.setattr(
        "video_script_reconstructor.model_store.verify_model",
        lambda name: {
            "offline_ready": name == "faster-whisper-large-v3",
            "directory": "C:/models/large-v3",
        },
    )

    adapters = _auto_asr_adapters(language="fil", compare_candidates=False)
    assert len(adapters) == 1
    assert adapters[0].cpu_threads == 8
    assert adapters[0].num_workers == 1


def test_faster_whisper_worker_policy_is_vram_guarded_and_overridable(monkeypatch) -> None:
    monkeypatch.delenv("VSR_FASTER_WHISPER_NUM_WORKERS", raising=False)
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.shutil.which",
        lambda name: "nvidia-smi.exe" if name == "nvidia-smi" else None,
    )
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="12288\n", returncode=0),
    )
    assert _faster_whisper_num_workers(duration_ms=120_000) == 2
    assert _faster_whisper_num_workers(duration_ms=600_000) == 1

    monkeypatch.setenv("VSR_FASTER_WHISPER_NUM_WORKERS", "1")
    assert _faster_whisper_num_workers(duration_ms=600_000) == 1
    monkeypatch.setenv("VSR_FASTER_WHISPER_NUM_WORKERS", "999")
    assert _faster_whisper_num_workers(duration_ms=600_000) == 8
    monkeypatch.setenv("VSR_FASTER_WHISPER_NUM_WORKERS", "invalid")
    assert _faster_whisper_num_workers(duration_ms=120_000) == 2


def test_faster_whisper_compute_type_policy_follows_host_device(monkeypatch) -> None:
    monkeypatch.delenv("VSR_FASTER_WHISPER_COMPUTE_TYPE", raising=False)
    monkeypatch.setattr("video_script_reconstructor.pipeline.shutil.which", lambda name: None)
    assert _faster_whisper_compute_type() == "int8"
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.shutil.which",
        lambda name: "nvidia-smi.exe" if name == "nvidia-smi" else None,
    )
    assert _faster_whisper_compute_type() == "float16"


def test_faster_whisper_compute_type_override_is_explicit_and_supported(monkeypatch) -> None:
    monkeypatch.setattr("video_script_reconstructor.pipeline.shutil.which", lambda name: None)
    for supported in ("int8_float16", "int8_bfloat16", "Default"):
        monkeypatch.setenv("VSR_FASTER_WHISPER_COMPUTE_TYPE", supported)
        assert _faster_whisper_compute_type() == supported.casefold()


def test_faster_whisper_compute_type_invalid_override_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr("video_script_reconstructor.pipeline.shutil.which", lambda name: None)
    monkeypatch.setenv("VSR_FASTER_WHISPER_COMPUTE_TYPE", "bfloat16")
    assert _faster_whisper_compute_type() == "int8"
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.shutil.which",
        lambda name: "nvidia-smi.exe" if name == "nvidia-smi" else None,
    )
    monkeypatch.setenv("VSR_FASTER_WHISPER_COMPUTE_TYPE", "fp16")
    assert _faster_whisper_compute_type() == "float16"


def test_auto_whisper_applies_compute_type_override_at_construction(monkeypatch) -> None:
    monkeypatch.setenv("VSR_FASTER_WHISPER_COMPUTE_TYPE", "int8_float16")
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline._developer_worker_path",
        lambda *args, **kwargs: Path("missing-worker.exe"),
    )
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.importlib.util.find_spec",
        lambda name: SimpleNamespace() if name == "faster_whisper" else None,
    )
    monkeypatch.setattr("video_script_reconstructor.pipeline.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "video_script_reconstructor.model_store.verify_model",
        lambda name: {
            "offline_ready": name == "faster-whisper-large-v3",
            "directory": "C:/models/large-v3",
        },
    )

    adapters = _auto_asr_adapters(language="fil", compare_candidates=False)

    assert len(adapters) == 1
    assert isinstance(adapters[0], FasterWhisperAdapter)
    assert adapters[0].device == "cpu"
    assert adapters[0].compute_type == "int8_float16"


def test_auto_whisper_batched_mode_is_explicitly_configurable(monkeypatch) -> None:
    monkeypatch.setenv("VSR_FASTER_WHISPER_BATCHED", "1")
    monkeypatch.setenv("VSR_FASTER_WHISPER_BATCH_SIZE", "4")
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline._developer_worker_path",
        lambda *args, **kwargs: Path("missing-worker.exe"),
    )
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.importlib.util.find_spec",
        lambda name: SimpleNamespace() if name == "faster_whisper" else None,
    )
    monkeypatch.setattr("video_script_reconstructor.pipeline.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "video_script_reconstructor.model_store.verify_model",
        lambda name: {
            "offline_ready": name == "faster-whisper-large-v3",
            "directory": "C:/models/large-v3",
        },
    )

    adapters = _auto_asr_adapters(language="fil", compare_candidates=False)

    assert len(adapters) == 1
    assert isinstance(adapters[0], FasterWhisperAdapter)
    assert adapters[0].inference_mode == "batched"
    assert adapters[0].batch_size == 4


def test_auto_whisper_accepts_explicit_complete_external_model(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "hf-cache" / "large-v3"
    model_dir.mkdir(parents=True)
    for filename in (
        "config.json",
        "model.bin",
        "preprocessor_config.json",
        "tokenizer.json",
        "vocabulary.json",
    ):
        (model_dir / filename).write_bytes(b"fixture")
    monkeypatch.setenv("VSR_FASTER_WHISPER_LARGE_V3_PATH", str(model_dir))
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline._developer_worker_path",
        lambda *args, **kwargs: Path("missing-worker.exe"),
    )
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.importlib.util.find_spec",
        lambda name: SimpleNamespace() if name == "faster_whisper" else None,
    )
    monkeypatch.setattr(
        "video_script_reconstructor.model_store.verify_model",
        lambda name: {"offline_ready": False},
    )

    adapters = _auto_asr_adapters(language="fil", compare_candidates=False)

    assert len(adapters) == 1
    assert isinstance(adapters[0], FasterWhisperAdapter)
    assert Path(adapters[0].model_name_or_path) == model_dir.resolve()
    assert adapters[0].model_revision is None
    assert adapters[0].model_signature is not None


def test_faster_whisper_model_identity_prefers_manifest_revision_and_tracks_stats(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "large-v3"
    model_dir.mkdir()
    for filename in (
        "config.json",
        "model.bin",
        "preprocessor_config.json",
        "tokenizer.json",
        "vocabulary.json",
    ):
        (model_dir / filename).write_bytes(b"fixture")
    (model_dir / "model-manifest.json").write_text(
        '{"revision":"manifest-revision-a"}', encoding="utf-8"
    )

    revision, signature = _faster_whisper_model_identity(model_dir)

    assert revision == "manifest-revision-a"
    assert signature is not None
    assert len(signature) == 64


def test_auto_whisper_passes_verified_model_revision(monkeypatch) -> None:
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline._developer_worker_path",
        lambda *args, **kwargs: Path("missing-worker.exe"),
    )
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.importlib.util.find_spec",
        lambda name: SimpleNamespace() if name == "faster_whisper" else None,
    )
    monkeypatch.setattr("video_script_reconstructor.pipeline.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "video_script_reconstructor.model_store.verify_model",
        lambda name: {
            "offline_ready": name == "faster-whisper-large-v3",
            "directory": "C:/models/large-v3",
            "revision": "verified-revision-a",
        },
    )

    adapters = _auto_asr_adapters(language="fil", compare_candidates=False)

    assert len(adapters) == 1
    assert adapters[0].model_revision == "verified-revision-a"
    assert "revision=verified-revision-a" in adapters[0].cache_identity


def test_no_candidate_comparison_short_circuits_unused_backend_probes(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline._developer_worker_path",
        lambda *args, **kwargs: Path("missing-worker.exe"),
    )
    monkeypatch.setattr(
        "video_script_reconstructor.pipeline.importlib.util.find_spec",
        lambda name: SimpleNamespace() if name == "faster_whisper" else None,
    )

    def verify(name: str) -> dict[str, object]:
        calls.append(name)
        return {
            "offline_ready": name == "faster-whisper-large-v3",
            "directory": "C:/models/large-v3",
        }

    monkeypatch.setattr("video_script_reconstructor.model_store.verify_model", verify)
    adapters = _auto_asr_adapters(language="fil", compare_candidates=False)
    assert [adapter.backend_name for adapter in adapters] == ["faster-whisper"]
    assert calls == ["faster-whisper-large-v3"]


def test_asr_cpu_thread_policy_is_bounded_and_overridable(monkeypatch) -> None:
    monkeypatch.setattr("video_script_reconstructor.pipeline.shutil.which", lambda name: None)
    monkeypatch.setattr("video_script_reconstructor.pipeline.os.cpu_count", lambda: 16)
    monkeypatch.delenv("VSR_ASR_CPU_THREADS", raising=False)
    assert _asr_cpu_threads() == 8
    monkeypatch.setenv("VSR_ASR_CPU_THREADS", "4")
    assert _asr_cpu_threads() == 4
    monkeypatch.setenv("VSR_ASR_CPU_THREADS", "999")
    assert _asr_cpu_threads() == 32
    monkeypatch.setenv("VSR_ASR_CPU_THREADS", "invalid")
    assert _asr_cpu_threads() == 8
