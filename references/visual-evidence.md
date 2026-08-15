# Visual evidence

## Contents

- Survey and packets
- Frame selection
- OCR and annotation
- Evidence groups

## Survey and packets

Cover the full duration with scene cuts, adaptive differences, static intervals, OCR/perceptual changes, motion/blur, chapters, deictic speech, and periodic safety samples no farther than 30 seconds in strict mode. Profile by time range. Build bounded packets with before/focus/after images, requested/actual times, transcript, OCR, scene/motion/difference context, and explicit questions.

## Frame selection

Score relevance, importance, temporal proximity, sharpness, stability, novelty, OCR readability, state completeness, transitions, evidence role, and small changes. Use duration- and density-aware limits. Always preserve an unaltered full frame; use only evidence-based crops with recorded geometry. Never use arbitrary center crops. If crop padding expands a proposed region to the complete parent frame, omit that non-localized crop and retain the parent as the sole authoritative image; emit `skipped_full_frame_crop_count` so the decision remains auditable.

Measure actual time from decoded metadata. Mark estimates; never assign requested time as actual without measurement. Protect changed numbers, code, options, controls, fields, errors, output, and cursor targets with region pixels, OCR, structure, and event semantics—not perceptual hashes alone.

## OCR and annotation

Keep raw OCR, normalized interpretation, bounds, alternatives, uncertain characters, language, engine/version, and human decisions distinct. Do not execute visible code. Ask semantic observers targeted questions and require atomic claims with image/region support. If none is available, preserve frames, emit the exact semantic-pending marker, create review items, and return review-required.

Prefer the locally pinned PP-OCRv5 server detector/recognizer for multilingual scene text and preserve its per-line boxes and scores. Use Tesseract as the small executable fallback. OCR is corroborating visible evidence, never proof of spoken wording. Use the default Codex/subagent review bundle for schema-constrained semantic observation; keep the packet hash, cited frame IDs, uncertainty, and statements deliberately not inferred. Local Qwen3-VL remains an explicit legacy compatibility route only.

## Evidence groups

Use before/action/after sequences for state changes. Do not infer an action or off-screen result from one frame. Keep every selected full frame inline beside a block and in the evidence index.
