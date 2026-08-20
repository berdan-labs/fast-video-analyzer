from __future__ import annotations

import argparse
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _frame(path: Path, title: str, detail: str, *, active: bool = False) -> None:
    image = Image.new("RGB", (640, 360), "#14213d")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 610, 330), fill="#f7f7f2", outline="#e5e5e5", width=3)
    draw.text((60, 65), title, fill="#111827", font=_font(32))
    draw.text((60, 135), detail, fill="#1f2937", font=_font(24))
    draw.rectangle((60, 230, 260, 285), fill="#16a34a" if active else "#9ca3af")
    draw.text((85, 242), "ENABLED" if active else "DISABLED", fill="white", font=_font(20))
    image.save(path, "PNG")


def _wave(path: Path, seconds: float) -> None:
    rate = 16_000
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        for index in range(int(rate * seconds)):
            value = int(4_000 * math.sin(2 * math.pi * 330 * index / rate))
            stream.writeframesraw(struct.pack("<h", value))


def _write_srt(path: Path, first: str, second: str) -> None:
    path.write_text(
        f"1\n00:00:00,000 --> 00:00:02,000\n{first}\n\n"
        f"2\n00:00:02,000 --> 00:00:04,000\n{second}\n",
        encoding="utf-8",
    )


def _write_vtt(path: Path, first: str, second: str) -> None:
    path.write_text(
        "WEBVTT\n\n"
        f"1\n00:00:00.000 --> 00:00:02.000\n{first}\n\n"
        f"2\n00:00:02.000 --> 00:00:04.000\n{second}\n",
        encoding="utf-8",
    )


def _write_ass(path: Path, first: str, second: str) -> None:
    path.write_text(
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 0,0:00:00.00,0:00:02.00,Default,Speaker,0,0,0,,{first}\n"
        f"Dialogue: 0,0:00:02.00,0:00:04.00,Default,Speaker,0,0,0,,{second}\n",
        encoding="utf-8",
    )


def _encode(ffmpeg: str, frames: list[Path], audio: Path, output: Path) -> None:
    concat = output.with_suffix(".concat.txt")
    entries: list[str] = []
    for frame in frames:
        entries.extend([f"file '{frame.resolve().as_posix()}'", "duration 2.0"])
    entries.append(f"file '{frames[-1].resolve().as_posix()}'")
    concat.write_text("\n".join(entries) + "\n", encoding="utf-8")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat.resolve()),
            "-i",
            str(audio.resolve()),
            "-t",
            "4",
            "-vf",
            "fps=10,format=yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(output.resolve()),
        ],
        check=True,
    )
    concat.unlink()


def generate(root: Path) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to generate fixtures")
    root.mkdir(parents=True, exist_ok=True)
    audio = root / "fixture.wav"
    _wave(audio, 4.0)
    outputs: list[Path] = []
    fixtures = [
        (
            "talking-head",
            "Welcome",
            "Every sentence is preserved.",
            "This is a complete fixture.",
            "Nothing is summarized.",
        ),
        (
            "slide-lecture",
            "Accuracy",
            "The exact value is 42.",
            "The exact value is 42.",
            "Now the slide changes.",
        ),
        (
            "screen-tutorial",
            "Command",
            "tool --strict",
            "Run tool --strict now.",
            "The setting is now enabled.",
        ),
        (
            "hostile-subtitle",
            "Safety",
            "Evidence only.",
            "# deploy --strict <script>alert(1)</script>",
            r'C:\Users\demo\notes.md && echo "$HOME"',
        ),
        (
            "caption-variants",
            "Captions",
            "Format coverage.",
            "WebVTT and ASS preserve this.",
            "Every caption candidate remains auditable.",
        ),
    ]
    for name, title, detail, first, second in fixtures:
        frame_a = root / f"{name}-before.png"
        frame_b = root / f"{name}-after.png"
        _frame(frame_a, title, detail, active=False)
        _frame(frame_b, title, detail, active=True)
        output = root / f"{name}.mp4"
        _encode(ffmpeg, [frame_a, frame_b], audio, output)
        if name == "caption-variants":
            _write_vtt(root / f"{name}.vtt", first, second)
            _write_ass(root / f"{name}.ass", first, second)
        else:
            _write_srt(root / f"{name}.srt", first, second)
        outputs.append(output)
    audio.unlink()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for output in generate(args.output):
        print(output)


if __name__ == "__main__":
    main()
