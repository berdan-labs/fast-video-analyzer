from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, cast

from .security import atomic_write_text

TOP_SECTIONS = [
    "Document map",
    "How to read this analysis",
    "Source and verification summary",
    "Chapter index",
    "Complete chronological analysis",
    "Unresolved evidence and review items",
    "Evidence image index",
    "Audit results",
    "Source-selection and correction history",
    "Reproducibility summary",
]


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return cast(dict[str, Any], value.model_dump(mode="json"))
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected model or mapping, got {type(value)!r}")


def timestamp(ms: int | None) -> str:
    if ms is None:
        return "untimed"
    value = max(0, int(ms))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def yaml_scalar(value: Any) -> str:
    text = str(value if value is not None else "")
    text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")
    return f'"{text}"'


def safe_evidence_text(value: str) -> str:
    escaped = html.escape(value, quote=False)
    escaped = re.sub(r"(?m)^(\s{0,3})([#>-]|\d+[.)])", r"\1\\\2", escaped)
    escaped = escaped.replace("[", "\\[").replace("]", "\\]")
    return escaped


def safe_fenced_block(value: str, language: str = "text") -> str:
    runs = [len(match.group(0)) for match in re.finditer(r"`+", value)]
    fence = "`" * max(3, (max(runs) + 1) if runs else 3)
    return f"{fence}{language}\n{value}\n{fence}"


def anchor_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", re.sub(r"\s+", "-", value.casefold()))


def _selected_text(segment: dict[str, Any]) -> str:
    return str(
        segment.get("human_verified_text")
        or segment.get("repaired_text")
        or segment.get("normalized_text")
        or segment.get("raw_text")
        or segment.get("text")
        or ""
    )


def render_markdown(project: Any) -> str:
    data = _dump(project)
    title = safe_evidence_text(str(data.get("source_title") or "Untitled source"))
    status = str(data.get("project_status") or "processing")
    media = _dump(data.get("media", {}))
    blocks = [_dump(item) for item in data.get("script_blocks", data.get("blocks", []))]
    chapters = [_dump(item) for item in data.get("chapters", [])]
    frames = [_dump(item) for item in data.get("frames", data.get("snapshots", []))]
    reviews = [_dump(item) for item in data.get("review_items", [])]
    ocr_by_text: dict[str, list[dict[str, Any]]] = {}
    for observation in data.get("ocr_observations", []):
        item = _dump(observation)
        normalized = str(item.get("normalized_interpretation") or "").strip()
        if normalized:
            ocr_by_text.setdefault(normalized.casefold(), []).append(item)
    audit = _dump(data.get("audit", {}))
    if not chapters:
        chapters = [
            {
                "chapter_id": "C001",
                "title": "Complete analysis (navigational)",
                "start_ms": min((b.get("start_ms") or 0 for b in blocks), default=0),
                "end_ms": max((b.get("end_ms") or 0 for b in blocks), default=0),
                "block_ids": [b.get("block_id") for b in blocks],
                "source_authored": False,
            }
        ]

    generated = str(data.get("generated_at_utc") or data.get("updated_at_utc") or "unknown")
    lines = [
        "---",
        "artifact_type: long-video-analysis",
        'schema_version: "1.0"',
        f"source_title: {yaml_scalar(data.get('source_title') or 'Untitled source')}",
        f"source_id: {yaml_scalar(media.get('media_id') or data.get('media_id') or 'unknown')}",
        f"duration: {yaml_scalar(timestamp(media.get('duration_ms')))}",
        f"generated_at_utc: {yaml_scalar(generated)}",
        f"project_status: {yaml_scalar(status)}",
        f"fidelity_mode: {yaml_scalar(data.get('fidelity_mode', 'verbatim'))}",
        f"primary_language: {yaml_scalar(data.get('primary_language') or 'und')}",
        'markdown_is_rendered_from: ".state/canonical-project.json"',
        "---",
        "",
        f"# {title} — Complete Video Analysis",
        "",
        f"> **Status: `{status}`.** {safe_evidence_text(str(data.get('status_reason') or 'See audit results and unresolved evidence below.'))}",
        "",
        "## Document map",
        "",
    ]
    for heading in TOP_SECTIONS[1:]:
        lines.append(f"- [{heading}](#{anchor_slug(heading)})")
    lines.append("- Chapters:")
    for chapter in chapters:
        cid = str(chapter.get("chapter_id", "C001"))
        lines.append(
            f"  - [{cid} — {safe_evidence_text(str(chapter.get('title', 'Chapter')))}](#{cid})"
        )

    lines.extend(
        [
            "",
            "## How to read this analysis",
            "",
            'Read each complete chronological block first. Important current image findings are already mirrored in the block. If wording or a claim remains uncertain, follow its frame and image-claim IDs. Read embedded image metadata with `long-video-analyzer evidence metadata show "<PROJECT_DIR>" <IMAGE_ID>` before requesting another visual pass. If metadata is insufficient, stale, contradictory, or lacks the needed precision, inspect the full frame, relevant crops, and adjacent before/action/after frames, then ingest a targeted enrichment observation. Finally, re-read the regenerated block and unresolved items.',
            "",
            "## Source and verification summary",
            "",
            f"- Source: `{safe_evidence_text(str(data.get('input_reference') or 'local evidence'))}`",
            f"- Duration: `{timestamp(media.get('duration_ms'))}`",
            f"- Transcript decision: {safe_evidence_text(str(data.get('transcript_source_decision') or 'See source-selection history.'))}",
            f"- Verification state: `{status}` (automatic checks never imply human verification)",
            "- This is an evidence-grounded analysis of completed media, not a recovered private or unpublished script.",
            "",
            "## Chapter index",
            "",
            "| Chapter | Title | Time range | Blocks | Evidence | Unresolved |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    frame_by_id = {str(frame.get("frame_id") or frame.get("image_id")): frame for frame in frames}
    review_by_block: dict[str, int] = {}
    for item in reviews:
        for block_id in item.get(
            "block_ids", ([item.get("block_id")] if item.get("block_id") else [])
        ):
            review_by_block[str(block_id)] = review_by_block.get(str(block_id), 0) + (
                0 if item.get("decision") else 1
            )
    block_by_id = {str(block.get("block_id")): block for block in blocks}
    for chapter in chapters:
        cid = str(chapter.get("chapter_id", "C001"))
        chapter_blocks = [
            block_by_id[item] for item in chapter.get("block_ids", []) if item in block_by_id
        ]
        if not chapter_blocks:
            chapter_blocks = [
                block for block in blocks if str(block.get("chapter_id", "C001")) == cid
            ]
        evidence_count = len(
            {frame for block in chapter_blocks for frame in block.get("frame_ids", [])}
        )
        unresolved_count = sum(
            review_by_block.get(str(block.get("block_id")), 0) for block in chapter_blocks
        )
        lines.append(
            f"| [{cid}](#{cid}) | {safe_evidence_text(str(chapter.get('title', 'Chapter')))} | "
            f"{timestamp(chapter.get('start_ms'))}–{timestamp(chapter.get('end_ms'))} | "
            f"{len(chapter_blocks)} | {evidence_count} | {unresolved_count} |"
        )

    lines.extend(["", "## Complete chronological analysis", ""])
    for chapter in chapters:
        cid = str(chapter.get("chapter_id", "C001"))
        ctitle = safe_evidence_text(str(chapter.get("title", "Chapter")))
        lines.extend(
            [
                f'<a id="{cid}"></a>',
                f"## {cid} — {ctitle} · {timestamp(chapter.get('start_ms'))}–{timestamp(chapter.get('end_ms'))}",
                "",
            ]
        )
        chapter_blocks = [block for block in blocks if str(block.get("chapter_id", "C001")) == cid]
        for block in chapter_blocks:
            block_id = str(block.get("block_id"))
            label = safe_evidence_text(
                str(block.get("speaker") or block.get("event_label") or "Event")
            )
            lines.extend(
                [
                    f'<a id="{block_id}"></a>',
                    f"### {block_id} · {timestamp(block.get('start_ms'))}–{timestamp(block.get('end_ms'))} · {label}",
                    "",
                ]
            )
            spoken = str(block.get("spoken_text", block.get("spoken", "")))
            if spoken:
                lines.extend(["**Spoken**", "", safe_evidence_text(spoken), ""])
            visual = str(block.get("visual_description", block.get("visual", "")))
            if not visual and data.get("visual_source_available") is False:
                visual = "[no visual source available]"
            elif not visual and block.get("frame_ids"):
                visual = "[visual evidence retained; semantic description pending review]"
            if visual:
                fixed_visual_markers = {
                    "[no visual source available]",
                    "[visual evidence retained; semantic description pending review]",
                }
                rendered_visual = (
                    visual if visual in fixed_visual_markers else safe_evidence_text(visual)
                )
                lines.extend(["**Visual**", "", rendered_visual, ""])
            on_screen = block.get("on_screen_text", [])
            if isinstance(on_screen, str):
                on_screen = [on_screen]
            if on_screen:
                lines.extend(["**On-screen text**", ""])
                for item in on_screen:
                    if "\n" in str(item) or "```" in str(item):
                        lines.extend([safe_fenced_block(str(item)), ""])
                    else:
                        ocr_matches = ocr_by_text.get(str(item).strip().casefold(), [])
                        if ocr_matches:
                            details = []
                            for match in ocr_matches:
                                confidence = match.get("confidence")
                                if confidence is not None:
                                    details.append(f"confidence {float(confidence):.2f}")
                                uncertain = match.get("uncertain_characters", [])
                                if uncertain:
                                    details.append(
                                        "uncertain characters: " + ", ".join(map(str, uncertain))
                                    )
                            suffix = "; ".join(details)
                            suffix = f" ({suffix})" if suffix else ""
                            lines.append(
                                f"- [OCR candidate] {safe_evidence_text(str(item))}{suffix}"
                            )
                        else:
                            lines.append(f"- {safe_evidence_text(str(item))}")
                lines.append("")
            audio = block.get("relevant_non_speech_audio", [])
            if isinstance(audio, str):
                audio = [audio]
            if audio:
                lines.extend(["**Relevant non-speech audio**", ""])
                lines.extend(f"- {safe_evidence_text(str(item))}" for item in audio)
                lines.append("")
            if block.get("frame_ids"):
                lines.extend(["**Evidence**", ""])
                for frame_id in block.get("frame_ids", []):
                    frame = frame_by_id.get(str(frame_id), {})
                    path = str(frame.get("full_frame_path") or frame.get("path") or "")
                    description = str(
                        frame.get("description")
                        or frame.get("current_factual_description")
                        or frame.get("selection_reason")
                        or "evidence frame"
                    )
                    if path:
                        lines.extend(
                            [
                                f'<a id="{frame_id}"></a>',
                                f"![{frame_id} — {safe_evidence_text(description)} at {timestamp(frame.get('actual_ms'))}]({path})",
                                "",
                                f"*{frame_id} · role: {frame.get('evidence_role', frame.get('role', 'other'))} · requested {timestamp(frame.get('requested_ms'))} · actual {timestamp(frame.get('actual_ms'))} · offset {frame.get('offset_ms', 'unknown')} ms · selected because {safe_evidence_text(str(frame.get('selection_reason') or 'it supports this block'))} · [open full-size]({path})*",
                                "",
                                f'*Embedded evidence knowledge · metadata revision `{frame.get("latest_revision_id") or "none"}` · supported image claims `{", ".join(frame.get("supported_claim_ids", [])) or "none"}` · unresolved `{len(frame.get("unresolved_claim_ids", []))}` · use `long-video-analyzer evidence metadata show "<PROJECT_DIR>" {frame_id}` for the complete structured record.*',
                                "",
                            ]
                        )
                    for crop in frame.get("crops", []):
                        crop_id = str(crop.get("crop_id") or "crop")
                        crop_path = str(crop.get("path") or "")
                        if not crop_path:
                            continue
                        crop_reason = safe_evidence_text(
                            str(crop.get("reason") or "evidence-based detail crop")
                        )
                        lines.extend(
                            [
                                f'<a id="{crop_id}"></a>',
                                f"![{crop_id} - {crop_reason}]({crop_path})",
                                "",
                                f"*[{crop_id}](#{crop_id}) - crop of [{frame_id}](#{frame_id}) - {crop_reason} - [open full-size]({crop_path})*",
                                "",
                            ]
                        )
            traces = [
                ("segments", block.get("transcript_segment_ids", block.get("segment_ids", []))),
                ("visual events", block.get("visual_event_ids", [])),
                ("frames", block.get("frame_ids", [])),
                ("image claims", block.get("image_claim_ids", [])),
                ("metadata revisions", block.get("metadata_revision_ids", [])),
                ("transformations", block.get("transformation_ids", [])),
            ]
            lines.extend(["**Trace**", ""])
            lines.append(
                " · ".join(
                    f"`{name}: {', '.join(map(str, ids)) or 'none'}`" for name, ids in traces
                )
            )
            lines.extend(
                [
                    "",
                    "**Verification**",
                    "",
                    f"`{block.get('verification_status', 'unverified')}` · confidence `{float(block.get('confidence', 0.0)):.2f}`",
                    "",
                    "**Uncertainty**",
                    "",
                ]
            )
            uncertainty = block.get("uncertainty", [])
            if isinstance(uncertainty, str):
                uncertainty = [uncertainty]
            if uncertainty:
                lines.extend(f"- {safe_evidence_text(str(item))}" for item in uncertainty)
            else:
                lines.append("- None.")
            lines.append("")

    lines.extend(["## Unresolved evidence and review items", ""])
    unresolved = [item for item in reviews if not item.get("decision")]
    if not unresolved:
        lines.append("No unresolved review items.")
    else:
        for item in unresolved:
            review_id = item.get("review_id")
            block_ids = item.get(
                "block_ids", ([item.get("block_id")] if item.get("block_id") else [])
            )
            links = ", ".join(f"[{bid}](#{bid})" for bid in block_ids)
            lines.extend(
                [
                    f'<a id="{review_id}"></a>',
                    f"### {review_id} · {item.get('severity', 'medium')} · {timestamp(item.get('start_ms'))}–{timestamp(item.get('end_ms'))}",
                    "",
                    f"- Type: `{item.get('category', 'uncertainty')}`",
                    f"- Uncertainty: {safe_evidence_text(str(item.get('problem') or 'Unspecified uncertainty'))}",
                    f"- Relevant blocks: {links or 'none'}",
                    f"- Frames/crops: {', '.join(map(str, item.get('frame_ids', []))) or 'none'}",
                    f"- Alternatives: {safe_evidence_text('; '.join(map(str, item.get('alternatives', []))) or 'none')}",
                    f"- Required action: {safe_evidence_text(str(item.get('required_action') or 'Inspect the cited evidence.'))}",
                    f"- Blocks full verification: `{bool(item.get('blocking'))}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## Evidence image index",
            "",
            "| Figure | Actual time | Role | Blocks | Selection reason | Full frame | Crops | Verification | Revision | Claims S/D/U | Sufficiency |",
            "|---|---:|---|---|---|---|---|---|---|---:|---|",
        ]
    )
    final_frames = [
        frame
        for frame in frames
        if frame.get("final", True) and not frame.get("parent_full_frame_id")
    ]
    for frame in final_frames:
        fid = str(frame.get("frame_id") or frame.get("image_id"))
        path = str(frame.get("full_frame_path") or frame.get("path") or "")
        counts = f"{len(frame.get('supported_claim_ids', []))}/{len(frame.get('disputed_claim_ids', []))}/{len(frame.get('unresolved_claim_ids', []))}"
        crop_links = ", ".join(
            f"[{crop.get('crop_id')}](#{crop.get('crop_id')})"
            for crop in frame.get("crops", [])
            if crop.get("crop_id")
        )
        lines.append(
            f"| [{fid}](#{fid}) | {timestamp(frame.get('actual_ms'))} | {frame.get('evidence_role', frame.get('role', 'other'))} | "
            f"{', '.join(map(str, frame.get('linked_block_ids', [])))} | {safe_evidence_text(str(frame.get('selection_reason') or ''))} | "
            f"[{path}]({path}) | {crop_links or 'none'} | {frame.get('verification_status', 'unverified')} | "
            f"{frame.get('latest_revision_id') or 'none'} | {counts} | {frame.get('metadata_sufficiency_state', 'not_evaluated')} |"
        )

    coverage = audit.get("source_segment_coverage", {})
    visual_coverage = audit.get("visual_event_coverage", {})
    metadata_coverage = audit.get("image_metadata_coverage", {})
    unresolved_by_severity = {
        severity: sum(
            1
            for item in reviews
            if not item.get("decision") and item.get("severity", "medium") == severity
        )
        for severity in ("critical", "high", "medium", "low")
    }
    lines.extend(
        [
            "",
            "## Audit results",
            "",
            f"- Substantive segment coverage: `{coverage.get('covered', 0)}/{coverage.get('total', 0)}`",
            f"- Missing / partial / duplicate: `{len(coverage.get('missing_ids', []))}/{len(coverage.get('partial_ids', []))}/{len(coverage.get('duplicate_ids', []))}`",
            f"- Visual event coverage: `{visual_coverage.get('covered', 0)}/{visual_coverage.get('total', 0)}`",
            f"- Generated / metadata-bearing images: `{metadata_coverage.get('generated_images', 0)}/{metadata_coverage.get('embedded_metadata_images', 0)}`",
            f"- Final evidence linked: `{metadata_coverage.get('final_evidence_images_linked', 0)}/{metadata_coverage.get('final_evidence_images', 0)}`",
            f"- Semantically analyzed images / Markdown-consumed image claims: `{metadata_coverage.get('semantically_analyzed_images', 0)}/{metadata_coverage.get('markdown_consumed_claims', 0)}`",
            f"- Unsupported spoken / visual statements: `{len(audit.get('unsupported_spoken_statements', []))}/{len(audit.get('unsupported_visual_statements', []))}`",
            f"- High-impact discrepancies: `{len(audit.get('high_impact_token_discrepancies', []))}`",
            f"- Timeline/frame-link errors: `{len(audit.get('timeline_errors', []))}`",
            "- Unresolved review items by severity: "
            + ", ".join(
                f"`{severity}={count}`" for severity, count in unresolved_by_severity.items()
            ),
            f"- Blocking failures: `{', '.join(audit.get('blocking_failures', [])) or 'none'}`",
            f"- Final status: `{audit.get('final_project_status', status)}`",
            f"- Status reason: {safe_evidence_text(str(data.get('status_reason') or 'See unresolved evidence above.'))}",
            "",
            "## Source-selection and correction history",
            "",
            f"- Transcript source decision: {safe_evidence_text(str(data.get('transcript_source_decision') or 'No candidate decision recorded.'))}",
        ]
    )
    corrections = data.get("corrections", [])
    if corrections:
        for correction in corrections:
            item = _dump(correction)
            lines.append(
                f"- `{item.get('timestamp')}` · {safe_evidence_text(str(item.get('reviewer')))} · {safe_evidence_text(str(item.get('target_id')))} · {safe_evidence_text(str(item.get('decision')))}"
            )
    else:
        lines.append("- Human corrections: none.")
    manifest = _dump(data.get("manifest", {}))
    lines.extend(
        [
            "",
            "## Reproducibility summary",
            "",
            f"- Configuration hash: `{manifest.get('source_config_hash', data.get('config_hash', 'unknown'))}`",
            f"- Code version: `{manifest.get('code_version', data.get('code_version', 'unknown'))}`",
            f"- Tools/models: {safe_evidence_text(str(data.get('tools_models_summary') or 'See .state/run-manifest.json.'))}",
            f"- Network activity: `{len(manifest.get('network_activity', []))} recorded actions`",
            "- Canonical source: `.state/canonical-project.json`",
            "",
        ]
    )
    return "\n".join(lines)


def render_to_path(project: Any, path: Path) -> None:
    atomic_write_text(path, render_markdown(project))
