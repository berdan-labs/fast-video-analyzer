# Canonical schema guide

## Contents

- Storage rules
- Record families
- State invariants

## Storage rules

Use Pydantic models with `extra="forbid"`, integer milliseconds, nullable unknown timing, deterministic IDs, versioned JSON, and generated schemas. Keep raw, normalized, repaired, and human-verified transcript text distinct. A missing human decision remains null.

## Record families

Canonical state includes media identity; transcript candidates, words, segments, and repairs; frames/crops, OCR, visual events; evidence-image envelopes, visual-analysis observations, image claims, metadata revisions, and sufficiency decisions; script blocks, chapters, review items, run manifest, audit report, and the encompassing canonical project.

Frame records preserve requested and measured actual time, raw PTS/time base, offset, source, decoded-pixel hash, image path, parent/crop geometry, selection role, linked IDs, metadata revision, and verification state. Script blocks cite every segment/event/frame/image-claim/revision/sufficiency/transformation they consume.

## State invariants

Keep IDs stable across unchanged reruns. Preserve observations and revisions append-only. Require optimistic concurrency on metadata bases. Keep disputed/rejected/unresolved claims out of factual descriptions. Mirror every fact consumed by Markdown in both the block and supporting claim IDs.

