"""Profile deterministic visual evidence on real media without ASR or vision models.

This deliberately calls the same visual-stage implementation used by
``run_pipeline`` but supplies one neutral scaffold block.  It is useful for
long videos whose transcript stage is unavailable in an offline environment;
it never invents transcript text or semantic claims.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from video_script_reconstructor.media_probe import probe_media
from video_script_reconstructor.pipeline import _extract_visual_evidence
from video_script_reconstructor.security import sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile the deterministic visual stage without model dependencies."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument(
        "--no-scene-detection",
        action="store_true",
        help="Disable hard-cut detection while retaining bounded periodic/adaptive sampling.",
    )
    parser.add_argument(
        "--no-adaptive-detection",
        action="store_true",
        help="Disable the 2 fps small-change survey branch.",
    )
    return parser


def _scaffold(duration_ms: int) -> list[dict[str, Any]]:
    return [
        {
            "block_id": "B000001",
            "chapter_id": "C001",
            "start_ms": 0,
            "end_ms": duration_ms,
            "speaker": "Visual profiler",
            "spoken_text": "",
            "visual_description": None,
            "on_screen_text": [],
            "relevant_non_speech_audio": [],
            "frame_ids": [],
            "transcript_segment_ids": [],
            "visual_event_ids": [],
            "image_claim_ids": [],
            "metadata_revision_ids": [],
            "metadata_sufficiency_decision_ids": [],
            "transformation_ids": [],
            "fidelity_mode": "verbatim",
            "confidence": 0.0,
            "verification_status": "unverified",
            "uncertainty": [],
            "residual_source_text": None,
        }
    ]


def profile(
    input_path: Path,
    *,
    output: Path,
    interval_seconds: float = 30.0,
    scene_detection: bool = True,
    adaptive_detection: bool = True,
) -> dict[str, Any]:
    source = input_path.resolve(strict=True)
    project_dir = output.resolve()
    (project_dir / ".state" / "vision" / "packets").mkdir(parents=True, exist_ok=True)
    (project_dir / "evidence" / "full").mkdir(parents=True, exist_ok=True)
    (project_dir / "evidence" / "crops").mkdir(parents=True, exist_ok=True)
    probe = probe_media(source)
    if probe.duration_ms is None:
        raise RuntimeError("visual profiler requires a measured media duration")
    started = time.perf_counter()
    # The production pipeline computes this immutable digest once and threads
    # it through every cache boundary.  Passing it here keeps the profiler
    # representative for long media instead of hashing the source once for
    # survey identity and again for raw-frame checkpoints.
    source_sha256 = sha256_file(source)
    progress_events: list[dict[str, Any]] = []
    frames, payloads, revisions, events, reviews, ocr = _extract_visual_evidence(
        source,
        project_dir,
        "M_PROFILE",
        probe.duration_ms,
        _scaffold(probe.duration_ms),
        survey_interval_seconds=interval_seconds,
        strict=True,
        ocr_adapter=None,
        ocr_enabled=False,
        scene_detection_enabled=scene_detection,
        frame_difference_enabled=adaptive_detection,
        progress_callback=progress_events.append,
        source_sha256=source_sha256,
    )
    return {
        "input": str(source),
        "project_dir": str(project_dir),
        "duration_ms": probe.duration_ms,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "full_frame_count": len(frames),
        "payload_count": len(payloads),
        "revision_count": len(revisions),
        "event_count": len(events),
        "review_count": len(reviews),
        "ocr_count": len(ocr),
        "crop_count": sum(len(frame.get("crops", [])) for frame in frames),
        "progress_events": progress_events,
    }


def main() -> int:
    args = _parser().parse_args()
    report = profile(
        args.input,
        output=args.output,
        interval_seconds=args.interval_seconds,
        scene_detection=not args.no_scene_detection,
        adaptive_detection=not args.no_adaptive_detection,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
