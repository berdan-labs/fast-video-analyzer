from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from video_script_reconstructor import paddle_ocr_worker
from video_script_reconstructor.paddle_ocr_adapter import PaddleOCRV5Adapter


def _verified(name: str, _: Path | None) -> dict[str, object]:
    return {
        "verified": True,
        "directory": f"C:/models/{name}",
        "revision": f"{name}-revision",
    }


def test_paddle_ocr_adapter_preserves_lines_boxes_scores_and_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"fixture")
    monkeypatch.setattr("video_script_reconstructor.paddle_ocr_adapter.verify_model", _verified)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        request = json.loads(str(kwargs["input"]))
        assert request["mode"] == "recognize"
        assert request["detector_path"].endswith("pp-ocrv5-server-det")
        payload = {
            "ok": True,
            "lines": [
                {"text": "Value 42", "confidence": 0.98, "bounding_box": [10, 20, 100, 30]},
                {"text": "Flag --safe", "confidence": 0.61, "bounding_box": [10, 60, 140, 30]},
            ],
            "package_versions": {"paddleocr": "3.7.0", "paddlepaddle-gpu": "3.2.0"},
        }
        return subprocess.CompletedProcess(
            args=["worker"],
            returncode=0,
            stdout="VSR_RESULT\t" + json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = PaddleOCRV5Adapter(worker_python=sys.executable)
    observation = adapter.recognize(
        image, frame_id="F000001", observation_id="O000001", language="eng"
    )

    assert observation.raw_engine_text == "Value 42\nFlag --safe"
    assert observation.bounding_region == (10, 20, 140, 70)
    assert observation.confidence == pytest.approx(0.795)
    assert [token.bounding_box for token in observation.tokens] == [
        (10, 20, 100, 30),
        (10, 60, 140, 30),
    ]
    assert observation.uncertain_characters[0]["text"] == "Flag --safe"
    assert observation.engine == "pp-ocrv5-server"
    assert "pp-ocrv5-server-det-revision" in observation.engine_version


def test_paddle_ocr_available_requires_both_verified_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "video_script_reconstructor.paddle_ocr_adapter.verify_model",
        lambda name, _: {
            "verified": name.endswith("-det"),
            "directory": f"C:/models/{name}",
            "reason": "missing recognizer",
        },
    )
    adapter = PaddleOCRV5Adapter(worker_python=sys.executable)
    assert adapter.available() is False


def test_paddle_ocr_spawn_worker_copies_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("video_script_reconstructor.paddle_ocr_adapter.verify_model", _verified)
    adapter = PaddleOCRV5Adapter(
        worker_python=sys.executable,
        model_root="C:/models",
        detector_name="detector",
        recognizer_name="recognizer",
        device="gpu:0",
        uncertainty_threshold=0.73,
        timeout_seconds=17.0,
    )

    child = adapter.spawn_worker()

    assert child is not adapter
    assert child.worker_python == adapter.worker_python
    assert child.model_root == adapter.model_root
    assert child.detector_name == adapter.detector_name
    assert child.recognizer_name == adapter.recognizer_name
    assert child.device == adapter.device
    assert child.uncertainty_threshold == adapter.uncertainty_threshold
    assert child.timeout_seconds == adapter.timeout_seconds
    assert child.cache_identity == adapter.cache_identity


def test_persistent_worker_reuses_loaded_engine_across_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector = tmp_path / "det"
    recognizer = tmp_path / "rec"
    detector.mkdir()
    recognizer.mkdir()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    calls = {"loads": 0, "images": []}

    def fake_load(_request: dict[str, object]) -> tuple[object, str]:
        calls["loads"] += 1
        return object(), "gpu:0"

    monkeypatch.setattr(paddle_ocr_worker, "_load_engine", fake_load)
    monkeypatch.setattr(
        paddle_ocr_worker,
        "_package_versions",
        lambda: {"paddleocr": "fixture", "paddlepaddle-gpu": "fixture"},
    )

    def fake_recognize(
        _ocr: object, images: list[Path], *, device: str
    ) -> list[list[dict[str, object]]]:
        calls["images"].extend((image, device) for image in images)
        return [
            [{"text": image.name, "confidence": 1.0, "bounding_box": [0, 0, 1, 1]}]
            for image in images
        ]

    monkeypatch.setattr(paddle_ocr_worker, "_recognize_images", fake_recognize)
    session = paddle_ocr_worker._PersistentSession()
    common = {
        "mode": "recognize_batch",
        "detector_path": str(detector),
        "recognizer_path": str(recognizer),
        "device": "gpu:0",
    }
    first_result = session.handle({**common, "images": [str(first)]})
    second_result = session.handle({**common, "images": [str(second)]})

    assert first_result["ok"] is True
    assert second_result["ok"] is True
    assert calls["loads"] == 1
    assert [item[0] for item in calls["images"]] == [first.resolve(), second.resolve()]


def test_worker_uses_one_predictor_call_for_a_bounded_batch(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    calls: list[object] = []

    class Result:
        def __init__(self, text: str) -> None:
            self.json = {
                "res": {
                    "rec_texts": [text],
                    "rec_scores": [1.0],
                    "rec_boxes": [[0, 0, 1, 1]],
                }
            }

    class FakeOCR:
        def predict(self, value: object) -> list[Result]:
            calls.append(value)
            assert isinstance(value, list)
            return [Result(Path(item).name) for item in value]

    result = paddle_ocr_worker._recognize_batch(
        {
            "images": [str(first), str(second)],
            "detector_path": str(tmp_path),
            "recognizer_path": str(tmp_path),
            "device": "gpu:0",
        },
        ocr=FakeOCR(),
        loaded_device="gpu:0",
        loaded_versions={"paddleocr": "fixture"},
    )

    assert len(calls) == 1
    assert calls[0] == [str(first.resolve()), str(second.resolve())]
    assert [item["image_path"] for item in result["items"]] == [
        str(first.resolve()),
        str(second.resolve()),
    ]
    assert [item["lines"][0]["text"] for item in result["items"]] == [
        "first.png",
        "second.png",
    ]
