"""Run the owner-controlled release-candidate model audit.

This command is deliberately separate from pull-request CI.  It runs the
model-dependent tests, evaluates the configured corpus, and writes a small
sanitized report.  Model weights, source media, pytest XML, generated projects,
and raw corpus output remain in the owner-selected work directory; they are
never copied into the report or the repository.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as SafeET

from video_script_reconstructor import __version__
from video_script_reconstructor.diagnostics import sanitize_diagnostic_value
from video_script_reconstructor.model_store import list_models

ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCHEMA_VERSION = "1.0"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_PACKAGE_NAMES = {
    "faster-whisper": "faster_whisper",
    "llama-cpp-python": "llama_cpp",
    "paddleocr": "paddleocr",
    "pillow": "PIL",
    "pytest": "pytest",
    "qwen-asr": "qwen_asr",
}
_CAPABILITY_ENV_FLAGS = (
    "VSR_FASTER_WHISPER_LARGE_V3_PATH",
    "VSR_FASTER_WHISPER_SMOKE_AUDIO",
    "VSR_QWEN_SPEECH_PYTHON",
    "VSR_MOSS_SPEECH_PYTHON",
    "VSR_PADDLE_OCR_PYTHON",
    "VSR_LOCAL_VISION_COMMAND",
    "VSR_LOCAL_VISION_MODEL",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON manifest {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest {path.name} must contain an object")
    return value


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay within the repository")
    return path.as_posix()


def load_audit_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate the public lane map without inspecting local secrets."""

    manifest_path = Path(path).expanduser().resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"model audit schema_version must be {AUDIT_SCHEMA_VERSION!r}")
    target = _safe_relative(manifest.get("pytest_target"), label="pytest_target")
    if not target.startswith("tests/"):
        raise ValueError("pytest_target must be below tests/")
    raw_lanes = manifest.get("lanes")
    if not isinstance(raw_lanes, list) or not raw_lanes:
        raise ValueError("lanes must be a non-empty list")
    lanes: list[dict[str, Any]] = []
    ids: set[str] = set()
    modules: set[str] = set()
    for index, raw_lane in enumerate(raw_lanes):
        if not isinstance(raw_lane, Mapping):
            raise ValueError(f"lanes[{index}] must be an object")
        lane_id = raw_lane.get("id")
        if not isinstance(lane_id, str) or not lane_id.strip():
            raise ValueError(f"lanes[{index}].id must be non-empty")
        if lane_id in ids:
            raise ValueError(f"duplicate audit lane id: {lane_id}")
        ids.add(lane_id)
        test_module = _safe_relative(
            raw_lane.get("test_module"), label=f"lanes[{index}].test_module"
        )
        if not test_module.startswith(f"{target}/") and Path(test_module).name != test_module:
            raise ValueError(f"lanes[{index}].test_module must identify a module in {target}")
        if not test_module.endswith(".py"):
            raise ValueError(f"lanes[{index}].test_module must be a Python module")
        if test_module in modules:
            raise ValueError(f"duplicate audit test module: {test_module}")
        modules.add(test_module)
        models = raw_lane.get("models", [])
        if not isinstance(models, list) or not all(
            isinstance(model, str) and model.strip() for model in models
        ):
            raise ValueError(f"lanes[{index}].models must be a list of names")
        description = raw_lane.get("description", lane_id)
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"lanes[{index}].description must be non-empty")
        lanes.append(
            {
                "id": lane_id,
                "description": description,
                "models": list(models),
                "test_module": test_module,
            }
        )
    corpus = manifest.get("corpus")
    if not isinstance(corpus, Mapping):
        raise ValueError("corpus must be an object")
    corpus_manifest = _safe_relative(corpus.get("manifest"), label="corpus.manifest")
    corpus_baseline = _safe_relative(corpus.get("baseline"), label="corpus.baseline")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "pytest_target": target,
        "lanes": lanes,
        "corpus": {"manifest": corpus_manifest, "baseline": corpus_baseline},
    }


def _safe_text(value: Any, *, limit: int = 500) -> str:
    sanitized = sanitize_diagnostic_value(str(value))
    return str(sanitized)[:limit]


def _git_revision() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = result.stdout.strip()
    return revision if _SHA_RE.fullmatch(revision) else None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in _SAFE_PACKAGE_NAMES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _model_evidence() -> list[dict[str, Any]]:
    """Return model identity/status fields while excluding local paths."""

    try:
        statuses = list_models()
    except Exception as exc:  # pragma: no cover - defensive host diagnostics
        return [{"inventory_error": _safe_text(exc)}]
    evidence: list[dict[str, Any]] = []
    for raw in statuses:
        evidence.append(
            {
                "name": raw.get("name"),
                "backend": raw.get("backend"),
                "source": raw.get("source"),
                "revision": _safe_text(raw.get("revision")) if raw.get("revision") else None,
                "available": bool(raw.get("available")),
                "verified": bool(raw.get("verified")),
                "offline_ready": bool(raw.get("offline_ready")),
            }
        )
    return evidence


def runtime_evidence() -> dict[str, Any]:
    """Capture reproducibility facts without values of paths or environment variables."""

    return {
        "application_version": __version__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "nvidia_smi_present": shutil.which("nvidia-smi") is not None,
        "git_revision": _git_revision(),
        "package_versions": _package_versions(),
        "configured_capability_flags": {
            name: bool(os.environ.get(name, "").strip()) for name in _CAPABILITY_ENV_FLAGS
        },
        "models": _model_evidence(),
    }


def _case_module(testcase: ET.Element) -> str:
    classname = testcase.attrib.get("classname", "").replace("\\", "/")
    if classname:
        stem = classname.rsplit(".", 1)[-1]
        if stem.endswith(".py"):
            return stem
        if stem.startswith("test_"):
            return f"{stem}.py"
    file_value = testcase.attrib.get("file", "").replace("\\", "/")
    if file_value:
        return Path(file_value).name
    return ""


def _element_message(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    message = element.attrib.get("message", "").strip()
    text = " ".join("".join(element.itertext()).split())
    combined = message or text
    return _safe_text(combined) if combined else None


def parse_junit(path: str | Path) -> list[dict[str, Any]]:
    """Parse pytest's JUnit output into path-free testcase facts."""

    try:
        root = SafeET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return []
    cases: list[dict[str, Any]] = []
    for testcase in root.iter("testcase"):
        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped = testcase.find("skipped")
        if failure is not None or error is not None:
            status = "fail"
            reason = _element_message(failure if failure is not None else error) or (
                "pytest testcase failed"
            )
        elif skipped is not None:
            status = "unavailable"
            reason = _element_message(skipped) or "capability was skipped"
        else:
            status = "pass"
            reason = None
        cases.append(
            {
                "module": _case_module(testcase),
                "name": _safe_text(testcase.attrib.get("name", ""), limit=200),
                "status": status,
                "reason": reason,
            }
        )
    return cases


def _classify_lane(lane: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    module = str(lane["test_module"])
    matches = [case for case in cases if case.get("module") == Path(module).name]
    counts = {
        "tests": len(matches),
        "passed": sum(case.get("status") == "pass" for case in matches),
        "failed": sum(case.get("status") == "fail" for case in matches),
        "unavailable": sum(case.get("status") == "unavailable" for case in matches),
    }
    if not matches:
        status = "fail"
        reason = "audit test was not collected"
    elif counts["failed"]:
        status = "fail"
        reason = next(
            str(case.get("reason"))
            for case in matches
            if case.get("status") == "fail" and case.get("reason")
        )
    elif counts["unavailable"]:
        status = "unavailable"
        reason = next(
            str(case.get("reason"))
            for case in matches
            if case.get("status") == "unavailable" and case.get("reason")
        )
        if counts["passed"]:
            reason = f"not all lane checks were available: {reason}"
    else:
        status = "pass"
        reason = None
    return {
        "id": lane["id"],
        "description": lane["description"],
        "models": list(lane["models"]),
        "test_module": module,
        "status": status,
        "reason": reason,
        **counts,
    }


def run_model_tests(
    manifest: Mapping[str, Any],
    work_root: Path,
    *,
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    """Run all model-dependent tests once and classify every required lane."""

    junit_path = work_root / "model-tests.junit.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(manifest["pytest_target"]),
        "-q",
        f"--junitxml={junit_path}",
    ]
    timed_out = False
    returncode: int | None
    runner_reason: str | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        if returncode != 0:
            output_lines = (completed.stderr or completed.stdout or "").splitlines()
            if output_lines:
                runner_reason = _safe_text(output_lines[-1])
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = None
    cases = parse_junit(junit_path)
    lanes = [_classify_lane(lane, cases) for lane in manifest["lanes"]]
    if timed_out:
        for lane in lanes:
            if lane["status"] == "fail" and lane["reason"] == "audit test was not collected":
                lane["reason"] = f"pytest timed out after {timeout_seconds} seconds"
    elif runner_reason:
        for lane in lanes:
            if lane["status"] == "fail" and lane["reason"] == "audit test was not collected":
                lane["reason"] = f"pytest runner failed: {runner_reason}"
    if any(lane["status"] == "fail" for lane in lanes) or (
        returncode is not None and returncode != 0
    ):
        status = "fail"
    elif any(lane["status"] == "unavailable" for lane in lanes):
        status = "unavailable"
    else:
        status = "pass"
    return {
        "status": status,
        "process_returncode": returncode,
        "runner_reason": runner_reason,
        "testcase_count": len(cases),
        "lanes": lanes,
        "artifact": "model-tests.junit.xml",
    }


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _corpus_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    quality = report.get("quality", {})
    performance = report.get("performance", {})
    quality = quality if isinstance(quality, Mapping) else {}
    performance = performance if isinstance(performance, Mapping) else {}
    failed_cases: list[dict[str, Any]] = []
    raw_cases = quality.get("cases", [])
    if isinstance(raw_cases, list):
        for item in raw_cases:
            if not isinstance(item, Mapping) or item.get("pass"):
                continue
            errors = item.get("errors", [])
            failed_cases.append(
                {
                    "case_id": _safe_text(item.get("case_id", ""), limit=100),
                    "errors": [_safe_text(error, limit=300) for error in errors[:3]]
                    if isinstance(errors, list)
                    else [],
                }
            )
    return {
        "status": "pass" if report.get("gate_pass") else "fail",
        "gate_pass": bool(report.get("gate_pass")),
        "scoring_version": report.get("scoring_version"),
        "model_revision": _safe_text(report.get("model_revision"), limit=200),
        "corpus_hash": _safe_text(report.get("corpus_hash"), limit=100),
        "quality_pass": bool(quality.get("pass")),
        "performance_pass": performance.get("pass"),
        "performance_required": bool(performance.get("required")),
        "case_count": len(raw_cases) if isinstance(raw_cases, list) else 0,
        "failed_cases": failed_cases,
        "artifact": "corpus-report.json",
        "markdown_artifact": "corpus-report.md",
    }


def run_corpus_evaluation(
    manifest: Mapping[str, Any],
    work_root: Path,
    *,
    corpus_manifest_path: Path | None = None,
    corpus_baseline_path: Path | None = None,
    preset: str = "strict",
    vision_mode: str = "none",
    repeat: int = 1,
    model_revision: str = "deterministic-fixtures",
    require_performance: bool = False,
    skip: bool = False,
) -> dict[str, Any]:
    """Run the manifest evaluator and retain only a sanitized summary."""

    if skip:
        return {
            "status": "unavailable",
            "gate_pass": False,
            "reason": "corpus evaluation was explicitly skipped",
            "artifact": None,
            "markdown_artifact": None,
        }
    corpus = manifest["corpus"]
    corpus_manifest = corpus_manifest_path or _resolve_repo_path(str(corpus["manifest"]))
    corpus_baseline = corpus_baseline_path or _resolve_repo_path(str(corpus["baseline"]))
    output_root = work_root / "corpus-output"
    report_json = work_root / "corpus-report.json"
    report_markdown = work_root / "corpus-report.md"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_corpus.py"),
        "--manifest",
        str(corpus_manifest),
        "--baseline",
        str(corpus_baseline),
        "--output-root",
        str(output_root),
        "--report-json",
        str(report_json),
        "--report-markdown",
        str(report_markdown),
        "--preset",
        preset,
        "--vision-mode",
        vision_mode,
        "--repeat",
        str(repeat),
        "--model-revision",
        model_revision,
    ]
    if require_performance:
        command.append("--require-performance-baseline")
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {
            "status": "fail",
            "gate_pass": False,
            "reason": _safe_text(exc),
            "process_returncode": None,
            "artifact": "corpus-report.json",
            "markdown_artifact": "corpus-report.md",
        }
    try:
        report = _load_json(report_json)
    except ValueError:
        return {
            "status": "fail",
            "gate_pass": False,
            "reason": f"corpus evaluator produced no readable report (exit {completed.returncode})",
            "process_returncode": completed.returncode,
            "artifact": "corpus-report.json",
            "markdown_artifact": "corpus-report.md",
        }
    summary = _corpus_summary(report)
    summary["process_returncode"] = completed.returncode
    if completed.returncode != 0:
        summary["status"] = "fail"
        summary["gate_pass"] = False
        summary["reason"] = "corpus evaluator exited non-zero"
    return summary


def build_report(
    manifest: Mapping[str, Any],
    *,
    runtime: Mapping[str, Any],
    model_tests: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> dict[str, Any]:
    lane_pass = all(lane.get("status") == "pass" for lane in model_tests.get("lanes", []))
    corpus_pass = corpus.get("status") == "pass" and bool(corpus.get("gate_pass"))
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "repository_revision": runtime.get("git_revision"),
        "runtime": dict(runtime),
        "model_tests": dict(model_tests),
        "corpus": dict(corpus),
        "required_lanes_pass": lane_pass,
        "corpus_gate_pass": corpus_pass,
        "gate_pass": lane_pass and corpus_pass,
        "lane_ids": [str(lane["id"]) for lane in manifest["lanes"]],
    }


def _markdown_cell(value: Any) -> str:
    return str(value if value not in (None, "") else "—").replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise release-note-ready summary without local paths."""

    runtime = report.get("runtime", {})
    runtime = runtime if isinstance(runtime, Mapping) else {}
    model_tests = report.get("model_tests", {})
    model_tests = model_tests if isinstance(model_tests, Mapping) else {}
    corpus = report.get("corpus", {})
    corpus = corpus if isinstance(corpus, Mapping) else {}
    lines = [
        "# Release-candidate model audit",
        "",
        f"- Gate: **{'PASS' if report.get('gate_pass') else 'FAIL'}**",
        f"- Repository revision: `{_markdown_cell(report.get('repository_revision'))}`",
        f"- Application: `{_markdown_cell(runtime.get('application_version'))}`",
        f"- Runtime: Python `{_markdown_cell(runtime.get('python'))}` / {_markdown_cell(runtime.get('os'))} / {_markdown_cell(runtime.get('machine'))}",
        f"- NVIDIA SMI present: `{_markdown_cell(runtime.get('nvidia_smi_present'))}`",
        "",
        "## Required lanes",
        "",
        "| Lane | Status | Models | Test cases | Reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    lanes = model_tests.get("lanes", [])
    if isinstance(lanes, list):
        for lane in lanes:
            if not isinstance(lane, Mapping):
                continue
            lines.append(
                "| `{}` | **{}** | {} | {} | {} |".format(
                    _markdown_cell(lane.get("id")),
                    _markdown_cell(str(lane.get("status", "fail")).upper()),
                    _markdown_cell(", ".join(str(item) for item in lane.get("models", []))),
                    _markdown_cell(lane.get("tests")),
                    _markdown_cell(lane.get("reason")),
                )
            )
    lines.extend(
        [
            "",
            "## Corpus evaluation",
            "",
            f"- Status: **{_markdown_cell(str(corpus.get('status', 'fail')).upper())}**",
            f"- Corpus hash: `{_markdown_cell(corpus.get('corpus_hash'))}`",
            f"- Model revision: `{_markdown_cell(corpus.get('model_revision'))}`",
            f"- Quality: `{_markdown_cell(corpus.get('quality_pass'))}`; performance: `{_markdown_cell(corpus.get('performance_pass'))}` (required: `{_markdown_cell(corpus.get('performance_required'))}`)",
            f"- Cases: `{_markdown_cell(corpus.get('case_count'))}`; failed cases: `{_markdown_cell(len(corpus.get('failed_cases', [])))}`",
            "",
            "This report records capability evidence only. Model weights, source media, generated projects, and raw benchmark output stay in the owner-controlled audit directory.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "tests" / "model_audit_manifest.json",
        help="public lane/corpus manifest (default: tests/model_audit_manifest.json)",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        help="fresh owner-local directory for raw tests, corpus output, and reports",
    )
    parser.add_argument("--report-json", type=Path, help="sanitized audit report path")
    parser.add_argument("--report-markdown", type=Path, help="sanitized Markdown report path")
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        help="owner-local corpus manifest override (defaults to the public lane map)",
    )
    parser.add_argument(
        "--corpus-baseline",
        type=Path,
        help="owner-local compatible corpus baseline override",
    )
    parser.add_argument("--pytest-timeout-seconds", type=int, default=3600)
    parser.add_argument("--preset", choices=("strict", "balanced"), default="strict")
    parser.add_argument(
        "--vision-mode", choices=("none", "host-agent", "auto", "local"), default="none"
    )
    parser.add_argument("--repeat", type=int, default=1, help="corpus evaluation repetitions")
    parser.add_argument("--model-revision", default="deterministic-fixtures")
    parser.add_argument("--require-performance-baseline", action="store_true")
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="diagnostic-only mode; the audit gate remains failed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow reuse of an existing work/report directory",
    )
    return parser


def _prepare_work_root(path: Path | None, *, force: bool) -> Path:
    if path is None:
        return Path(tempfile.mkdtemp(prefix="fast-video-analyzer-model-audit-"))
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()) and not force:
        raise ValueError(
            f"work root is not empty: {resolved.name}; choose a fresh directory or pass --force"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_artifact(path: Path, content: str, *, force: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise ValueError(f"refusing to overwrite existing audit artifact: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.pytest_timeout_seconds <= 0 or args.repeat <= 0:
        print("FAIL: timeout and repeat must be positive", file=sys.stderr)
        return 2
    try:
        manifest = load_audit_manifest(args.manifest)
        work_root = _prepare_work_root(args.work_root, force=args.force)
        runtime = runtime_evidence()
        model_tests = run_model_tests(
            manifest,
            work_root,
            timeout_seconds=args.pytest_timeout_seconds,
        )
        corpus = run_corpus_evaluation(
            manifest,
            work_root,
            corpus_manifest_path=(
                args.corpus_manifest.expanduser().resolve() if args.corpus_manifest else None
            ),
            corpus_baseline_path=(
                args.corpus_baseline.expanduser().resolve() if args.corpus_baseline else None
            ),
            preset=args.preset,
            vision_mode=args.vision_mode,
            repeat=args.repeat,
            model_revision=args.model_revision,
            require_performance=args.require_performance_baseline,
            skip=args.skip_corpus,
        )
        report = build_report(
            manifest,
            runtime=runtime,
            model_tests=model_tests,
            corpus=corpus,
        )
        report_json = (args.report_json or work_root / "model-audit.json").expanduser().resolve()
        report_markdown = (
            (args.report_markdown or work_root / "model-audit.md").expanduser().resolve()
        )
        _write_artifact(
            report_json,
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            force=args.force,
        )
        _write_artifact(report_markdown, render_markdown(report), force=args.force)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(f"Model audit: {'PASS' if report['gate_pass'] else 'FAIL'}")
    print(f"Sanitized JSON: {report_json}")
    print(f"Sanitized Markdown: {report_markdown}")
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
