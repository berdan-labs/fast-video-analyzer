from __future__ import annotations

import os
from pathlib import Path

import pytest

from video_script_reconstructor.model_store import verify_model
from video_script_reconstructor.paddle_ocr_adapter import PaddleOCRV5Adapter

pytestmark = [pytest.mark.model_dependent, pytest.mark.ocr_executable]


def test_real_pp_ocrv5_reads_exact_high_impact_text_and_boxes() -> None:
    detector = verify_model("pp-ocrv5-server-det")
    recognizer = verify_model("pp-ocrv5-server-rec")
    if not detector.get("offline_ready") or not recognizer.get("offline_ready"):
        pytest.skip("PP-OCRv5 server detector/recognizer weights are unavailable")
    repository = Path(__file__).resolve().parents[2]
    default_worker = repository / ".artifacts" / "workers" / "paddleocr" / "Scripts" / "python.exe"
    worker = Path(os.environ.get("VSR_PADDLE_OCR_PYTHON", default_worker)).resolve()
    if not worker.is_file():
        pytest.skip("PP-OCRv5 worker unavailable: set VSR_PADDLE_OCR_PYTHON")
    image = (
        Path(__file__).resolve().parents[1] / "fixtures" / "generated" / "slide-lecture-after.png"
    )
    observation = PaddleOCRV5Adapter(worker_python=worker).recognize(
        image, frame_id="F000001", observation_id="O000001"
    )

    assert observation.engine == "pp-ocrv5-server"
    assert "The exact value is 42." in observation.normalized_interpretation
    assert "ENABLED" in observation.normalized_interpretation
    assert observation.confidence is not None and observation.confidence > 0.95
    assert all(
        token.bounding_box[2] > 0 and token.bounding_box[3] > 0 for token in observation.tokens
    )
    assert str(detector["revision"]) in observation.engine_version
    assert str(recognizer["revision"]) in observation.engine_version
