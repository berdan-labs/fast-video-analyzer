from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from video_script_reconstructor.ocr import TesseractOCRAdapter

pytestmark = [pytest.mark.model_dependent, pytest.mark.ocr_executable]


def _readable_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def test_real_tesseract_reads_exact_high_impact_token(tmp_path: Path) -> None:
    adapter = TesseractOCRAdapter(page_segmentation_mode=6)
    if not adapter.available():
        pytest.skip("Tesseract executable unavailable: install Tesseract and place it on PATH")
    image_path = tmp_path / "tesseract-smoke.png"
    image = Image.new("RGB", (1600, 320), "white")
    ImageDraw.Draw(image).text(
        (60, 80), "MODEL DEPENDENT OCR VALUE 42", fill="black", font=_readable_font(72)
    )
    image.save(image_path, "PNG")
    observation = adapter.recognize(
        image_path,
        frame_id="F000001",
        observation_id="O000001",
        language="eng",
    )
    assert observation.engine == "tesseract"
    assert observation.engine_version
    assert "42" in observation.normalized_interpretation
    assert observation.tokens
