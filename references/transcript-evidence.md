# Transcript evidence

## Contents

- Fidelity
- Candidate validation
- Repair and ASR
- Speakers and audio events

## Fidelity

Default to verbatim. Preserve filler, false starts, repetitions, corrections, caveats, tangents, warnings, jokes, sponsor content, and meaningful disfluency. `clean-verbatim` may only apply conservative punctuation/capitalization, cue-fragment repair, proven duplicate removal, and paragraph boundaries, with transformations recorded.

Audit ordered tokens and meaning units. Bag-of-words or word count cannot distinguish “Dog bites man” from “Man bites dog.” Treat names, numbers, commands, URLs, paths, code, measurements, medical/legal/scientific/financial/safety terms as high impact.

## Candidate validation

Check encoding, placeholders, invalid ranges, non-monotonicity, overlaps, duplicates, hallucination loops, gaps, speech mismatch, drift, duration/language mismatch, reading rate, broken boundaries, high-impact tokens, and missing speech. Produce interval diagnostics and preserve rejected candidates.

## Repair and ASR

Extract only the suspect interval plus configured context; offset returned words/segments into the media timeline; align sequence-aware; replace only supported wording; preserve alternatives and repair records. Chunk full ASR with overlap/checkpoints and deduplicate boundaries without losing repeated speech. Never silently substitute a smaller Whisper model.
Chunk size is a measurable scheduling input, not an accuracy shortcut. On the
tested RTX 3060/large-v3 FC101 slice, 150-second windows with 15-second overlap
completed faster than one 600-second window and merged more words; benchmark
`--asr-chunk-seconds 150 --asr-overlap-seconds 15` on the target host before
using it, because every boundary setting is checkpoint-keyed and auditable.

Keep the heavyweight runtimes isolated. Prefer faster-whisper large-v3 for Filipino or mixed
Filipino-English speech when its verified local weights are present; pass `--language fil` or set
`VSR_PREFER_WHISPER=1`. Its adapter tries CUDA and automatically retries CPU/int8 for missing
cuBLAS/CUDA runtimes. The Qwen/MOSS/forced-alignment workers are legacy compatibility adapters,
disabled for ordinary runs; use them only with explicit opt-in when an independent corroboration
experiment is required. The default route preserves Whisper's word timestamps and routes visual
and script interpretation to the bounded Codex/subagent review workflow.
If `doctor --offline` sees an NVIDIA GPU but no cuBLAS, install the optional `cuda` extra
(`pip install ".[asr,cuda]"`) before a long run; the actual runtime path is recorded in checkpoints.
Preserve discrepancies such as `42`, `forty-two`, and `forty two`; do not normalize away a
high-impact exact-wording question. A normal reconstruction run is offline and must never fetch
weights.

## Speakers and audio events

Use neutral speaker labels unless a name is supplied, explicitly spoken, unambiguously shown, or manually mapped with evidence. Preserve reliable captioned music, laughter, applause, alarms, and meaningful silence. Never invent an analyzed-empty audio result.

Any optional neural diarization labels are backend-local clustering labels, not identities. Map them in first-appearance order to `Speaker 1`, `Speaker 2`, and so on. If one selected transcript segment overlaps more than one diarization turn and exact splitting is not evidence-safe, keep the boundary uncertain instead of guessing.
