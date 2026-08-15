# Markdown output contract

## Contents

- Project tree
- Document order
- Blocks and links
- Validation

## Project tree

Create `<video stem>.md` inside `<video stem> (Analyzer Outputs)`, plus `evidence/full/`, `evidence/crops/`, and `.state/`. The entire project contains exactly one `.md` file, zero HTML, and no output README or alternate script. Candidate and diagnostic frames stay under `.state/`. Explicit `--output` roots retain the legacy slugged project layout for compatibility.

## Document order

Render YAML scalars, title/status, document map, reading guide, source/verification summary, chapter index, chronological reconstruction, unresolved evidence, evidence index, audits, source/correction history, and reproducibility in that order. Link every top-level section and chapter.

## Blocks and links

Use stable chapter/block anchors. Include complete spoken wording, evidence-grounded visuals, on-screen text, relevant audio, inline full frames/crops, image revision/claim summary, exact trace IDs, verification, and uncertainty. Use forward-slash relative links. Escape hostile headings, links, and HTML; choose fences longer than any embedded backtick run.

## Validation

Fail on the wrong Markdown count, any HTML, missing/escaping/orphan images, crop without parent, invalid/mismatched metadata, pixel/hash/timestamp/revision disagreements, unsupported/stale/contradicted consumed claims, unknown IDs, missing anchors, or status disagreement.
