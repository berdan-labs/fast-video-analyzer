from __future__ import annotations

import os
from pathlib import Path

import pytest


def _managed_worker_python(name: str) -> Path:
    """Resolve an installed optional worker without downloading or importing it."""

    configured_root = os.environ.get("VSR_WORKER_ROOT", "").strip()
    if configured_root:
        root = Path(configured_root).expanduser()
    elif os.name == "nt":
        root = (
            Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            / "video-script-reconstructor"
            / "workers"
        )
    else:
        root = (
            Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
            / "video-script-reconstructor"
            / "workers"
        )
    executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return (root / name / executable).expanduser().resolve()


@pytest.fixture(autouse=True)
def isolate_optional_model_store(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep ordinary tests hermetic while real-model tests use the installed store."""

    if request.node.get_closest_marker("model_dependent") is None:
        monkeypatch.setenv("VSR_MODEL_ROOT", str(tmp_path / "optional-models"))
        return

    # Model-dependent checks should exercise a managed local worker whenever it
    # is already installed, but remain safely skippable on a clean CI host. An
    # explicit environment override always wins, so callers can benchmark an
    # isolated development worker without changing the test harness.
    for variable, worker_name in (
        ("VSR_QWEN_SPEECH_PYTHON", "qwen-speech"),
        ("VSR_MOSS_SPEECH_PYTHON", "moss-speech"),
        ("VSR_PADDLE_OCR_PYTHON", "paddle-ocr"),
    ):
        if os.environ.get(variable, "").strip():
            continue
        worker = _managed_worker_python(worker_name)
        if worker.is_file():
            monkeypatch.setenv(variable, str(worker))
