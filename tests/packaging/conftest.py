from __future__ import annotations

import os
import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class InstalledWheel:
    repository: Path
    wheel: Path
    sdist: Path
    environment: Path
    python: Path
    console: Path
    legacy_console: Path
    workdir: Path

    def run(self, arguments: list[str | Path], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        return subprocess.run(
            [str(item) for item in arguments],
            cwd=self.workdir,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            **kwargs,
        )


@pytest.fixture(scope="session")
def installed_wheel(tmp_path_factory: pytest.TempPathFactory) -> InstalledWheel:
    root = tmp_path_factory.mktemp("installed-wheel")
    dist = root / "dist"
    dist.mkdir()
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--outdir",
            str(dist),
            str(REPOSITORY),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if build.returncode != 0:
        pytest.fail(f"wheel/sdist build failed\nSTDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}")
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        pytest.fail(f"expected one wheel and one sdist, found {wheels!r} and {sdists!r}")

    environment = root / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    if os.name == "nt":
        python = environment / "Scripts" / "python.exe"
        console = environment / "Scripts" / "long-video-analyzer.exe"
        legacy_console = environment / "Scripts" / "video-script-reconstructor.exe"
    else:
        python = environment / "bin" / "python"
        console = environment / "bin" / "long-video-analyzer"
        legacy_console = environment / "bin" / "video-script-reconstructor"
    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheels[0])],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if install.returncode != 0:
        pytest.fail(
            f"wheel installation failed\nSTDOUT:\n{install.stdout}\nSTDERR:\n{install.stderr}"
        )
    workdir = root / "outside-repository"
    workdir.mkdir()
    return InstalledWheel(
        repository=REPOSITORY,
        wheel=wheels[0],
        sdist=sdists[0],
        environment=environment,
        python=python,
        console=console,
        legacy_console=legacy_console,
        workdir=workdir,
    )
