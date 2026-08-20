from __future__ import annotations

import tarfile
import zipfile

from conftest import InstalledWheel


def test_wheel_contains_every_installed_resource(installed_wheel: InstalledWheel) -> None:
    required = {
        "video_script_reconstructor/resources/configs/strict.yaml",
        "video_script_reconstructor/resources/configs/balanced.yaml",
        "video_script_reconstructor/resources/configs/schema.json",
        "video_script_reconstructor/resources/prompts/annotate-visual-packet.md",
        "video_script_reconstructor/resources/references/security.md",
        "video_script_reconstructor/resources/assets/templates/reconstruction.md.j2",
        "video_script_reconstructor/resources/agents/openai.yaml",
        "video_script_reconstructor/resources/SKILL.md",
    }
    with zipfile.ZipFile(installed_wheel.wheel) as archive:
        names = set(archive.namelist())
    assert required <= names


def test_sdist_contains_validator_and_source_contract(installed_wheel: InstalledWheel) -> None:
    with tarfile.open(installed_wheel.sdist, "r:gz") as archive:
        names = {name.split("/", 1)[-1] for name in archive.getnames() if "/" in name}
    required = {
        "scripts/validate_fixture_output.py",
        "scripts/validate_corpus_manifest.py",
        "configs/strict.yaml",
        "configs/schema.json",
        "src/video_script_reconstructor/validate_output.py",
        "tests/acceptance_manifest.json",
        "tests/corpus_manifest.json",
        "SKILL.md",
    }
    assert required <= names
