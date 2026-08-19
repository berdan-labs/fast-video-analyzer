from __future__ import annotations

import json
from pathlib import Path

from conftest import InstalledWheel


def test_import_and_resources_come_from_installed_wheel(installed_wheel: InstalledWheel) -> None:
    program = """
import importlib.metadata
import importlib.resources
import json
import video_script_reconstructor
import long_video_analyzer
from video_script_reconstructor.config import load_config

root = importlib.resources.files("video_script_reconstructor")
resources = {
    "strict": root.joinpath("resources/configs/strict.yaml").read_text(encoding="utf-8"),
    "schema": root.joinpath("resources/configs/schema.json").read_text(encoding="utf-8"),
    "prompt": root.joinpath("resources/prompts/annotate-visual-packet.md").read_text(encoding="utf-8"),
    "reference": root.joinpath("resources/references/security.md").read_text(encoding="utf-8"),
    "template": root.joinpath("resources/assets/templates/reconstruction.md.j2").read_text(encoding="utf-8"),
    "agent": root.joinpath("resources/agents/openai.yaml").read_text(encoding="utf-8"),
    "skill": root.joinpath("resources/SKILL.md").read_text(encoding="utf-8"),
}
entry_points = [
    (item.name, item.value)
    for item in importlib.metadata.entry_points(group="console_scripts")
    if item.name in {"fast-video-analyzer", "long-video-analyzer", "video-script-reconstructor"}
]
print(json.dumps({
    "package_file": video_script_reconstructor.__file__,
    "preset": load_config("strict").preset,
    "resource_lengths": {key: len(value) for key, value in resources.items()},
    "entry_points": entry_points,
    "version": importlib.metadata.version("fast-video-analyzer"),
    "wrapper_file": long_video_analyzer.__file__,
}))
"""
    completed = installed_wheel.run([installed_wheel.python, "-c", program])
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    package_file = Path(result["package_file"]).resolve()
    assert package_file.is_relative_to(installed_wheel.environment.resolve())
    assert not package_file.is_relative_to(installed_wheel.repository.resolve())
    assert result["preset"] == "strict"
    assert min(result["resource_lengths"].values()) > 0
    entry_points = dict(result["entry_points"])
    assert entry_points["fast-video-analyzer"] == "video_script_reconstructor.cli:main"
    assert entry_points["long-video-analyzer"] == "video_script_reconstructor.cli:main"
    assert entry_points["video-script-reconstructor"] == "video_script_reconstructor.cli:main"
    assert result["version"] == "0.1.0"
    assert Path(result["wrapper_file"]).resolve().is_relative_to(installed_wheel.environment.resolve())


def test_console_and_module_entry_points_work_outside_repository(
    installed_wheel: InstalledWheel,
) -> None:
    console_help = installed_wheel.run([installed_wheel.console, "--help"])
    legacy_help = installed_wheel.run([installed_wheel.legacy_console, "--help"])
    compatibility_help = installed_wheel.run([installed_wheel.compatibility_console, "--help"])
    module_help = installed_wheel.run(
        [installed_wheel.python, "-m", "video_script_reconstructor", "--help"]
    )
    assert console_help.returncode == 0, console_help.stderr
    assert legacy_help.returncode == 0, legacy_help.stderr
    assert compatibility_help.returncode == 0, compatibility_help.stderr
    assert module_help.returncode == 0, module_help.stderr
    for output in (console_help.stdout, module_help.stdout):
        assert "doctor" in output
        assert "plan" in output
        assert "run" in output
        assert "validate" in output
        assert "models" in output
        assert "workers" in output
    assert "Fast Video Analyzer" in console_help.stdout

    workers = installed_wheel.run([installed_wheel.console, "workers", "list"])
    assert workers.returncode == 0, workers.stderr
    worker_report = json.loads(workers.stdout)
    assert {item["name"] for item in worker_report} == {
        "qwen-speech",
        "moss-speech",
        "paddle-ocr",
    }

    doctor = installed_wheel.run([installed_wheel.console, "doctor", "--offline"])
    assert doctor.returncode == 0, doctor.stderr
    report = json.loads(doctor.stdout)
    assert report["checks"]["network_policy"]["value"] == "offline"
    assert report["checks"]["ffmpeg"]["status"] in {"available", "blocking-for-strict"}


def test_installed_wheel_dependencies_are_consistent(installed_wheel: InstalledWheel) -> None:
    completed = installed_wheel.run([installed_wheel.python, "-m", "pip", "check"])
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "No broken requirements found" in completed.stdout


def test_production_fixture_validator_wraps_installed_pipeline(
    installed_wheel: InstalledWheel,
) -> None:
    sentence = "A complete transcript-only packaging sentence with token 42."
    transcript = installed_wheel.workdir / "packaging-transcript.txt"
    transcript.write_text(sentence, encoding="utf-8")
    output_root = installed_wheel.workdir / "output"
    run = installed_wheel.run(
        [
            installed_wheel.console,
            "run",
            transcript,
            "--output",
            output_root,
            "--vision-mode",
            "none",
            "--offline",
        ]
    )
    assert run.returncode == 0, run.stdout + run.stderr
    project_dir = Path(json.loads(run.stdout)["project_dir"])
    validator = installed_wheel.repository / "scripts" / "validate_fixture_output.py"
    valid = installed_wheel.run(
        [
            installed_wheel.python,
            validator,
            project_dir,
            "--expect-spoken",
            sentence,
            "--expect-token",
            "42",
            "--expect-status",
            "automatically_checked",
        ]
    )
    assert valid.returncode == 0, valid.stdout + valid.stderr
    result = json.loads(valid.stdout)
    assert result["valid"] is True
    assert result["production_valid"] is True
    assert result["checks"]["expected_spoken_count"] == 1

    invalid = installed_wheel.run(
        [installed_wheel.python, validator, project_dir, "--expect-token", "43"]
    )
    assert invalid.returncode == 4
    invalid_result = json.loads(invalid.stdout)
    assert invalid_result["valid"] is False
    assert any("expected exact token '43'" in error for error in invalid_result["errors"])
