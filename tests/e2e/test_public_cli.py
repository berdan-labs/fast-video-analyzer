from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE)
    return environment


def test_public_cli_decodes_video_and_validates_single_artifact(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts" / "generate_fixtures.py"), str(fixtures)],
        check=True,
    )
    assert (fixtures / "hostile-subtitle.srt").is_file()
    assert (fixtures / "caption-variants.vtt").is_file()
    assert (fixtures / "caption-variants.ass").is_file()
    output = tmp_path / "output"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "video_script_reconstructor",
            "run",
            str(fixtures / "screen-tutorial.mp4"),
            "--subtitle",
            str(fixtures / "screen-tutorial.srt"),
            "--output",
            str(output),
            "--offline",
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 3, run.stderr
    report = json.loads(run.stdout)
    project = Path(report["project_dir"])
    assert report["status"] == "review_required"
    validate = subprocess.run(
        [sys.executable, "-m", "video_script_reconstructor", "validate", str(project)],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr
    assert json.loads(validate.stdout)["valid"] is True
    assert len(list(project.rglob("*.md"))) == 1
    assert not list(project.rglob("*.html"))
    canonical = json.loads(
        (project / ".state" / "canonical-project.json").read_text(encoding="utf-8")
    )
    image_id = canonical["frames"][0]["frame_id"]
    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "video_script_reconstructor",
            "evidence",
            "metadata",
            "verify",
            str(project),
            image_id,
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)[0]["verified"] is True
