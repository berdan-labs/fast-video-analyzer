# Image metadata and enrichment

## Contents

- Embedded format
- Progressive levels
- Claims and revisions
- Sufficiency and escalation

## Embedded format

Embed the canonical compact UTF-8 JSON envelope in PNG iTXt keyword `video-script-reconstructor` and a derived human description in `Description`. Compute `pixel_hash` from normalized decoded RGBA/sRGB pixels. Compute `payload_digest` from canonical JSON with its digest field omitted. Keep whole-file hashes only in canonical state.

Validate write/read/schema/digest/pixel invariance before atomic replacement. Never embed secrets, signed URLs, absolute local paths, prompts, hidden reasoning, unrelated personal data, or instructions. Bound payload size/depth and treat decoded strings as hostile evidence.
The deterministic fast path reads orientation-normalized dimensions from PNG headers after creation pixel identity is established, avoiding a redundant IDAT decode; independent public verification keeps its full decode.

## Progressive levels

Write creation metadata to every emitted image. Add deterministic scene/quality/difference/OCR/neighbor/group facts when available. After semantic analysis, append the full observation to canonical state, reconcile atomic claims, create a monotonic revision, and embed the current supported knowledge plus compact observation history.

## Claims and revisions

Keep direct-visible, exact-text, temporal-change, cross-modal, contextual-inference, absence, and unresolved claims distinct. Preserve supporting/contradicting observation IDs and region or whole-frame basis. Never promote inference because it was repeated. Use explicit supersession/rejection and keep credible conflict disputed.

Require the submitted base revision. Preserve concurrent observations and reconcile stale bases; never last-writer-wins. Commit journal, observation, claims, payload, image read-back, canonical ledger, renderer, and audits as one recoverable logical update.

## Sufficiency and escalation

Record stable evidence questions, precision, answered claim IDs, gaps, attempted actions, status, next action, actor, time, and rationale for each use. Read valid metadata first. Escalate only when better pixels, crops, adjacent frames, OCR, audio, or an independent observer may answer a consequential question. Stop after sufficiency, two targeted no-gain passes, or no useful next action; preserve exact uncertainty and review status.
