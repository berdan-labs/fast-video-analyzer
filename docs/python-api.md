# Python API

This is the supported synchronous library facade for one-input workflows. It
is deliberately smaller than the CLI and implementation package. The public
module is `video_script_reconstructor.api`; `pipeline`, `review`, and
`validate_output` imports remain internal.

## Guarantees

- Calls are synchronous and local. `plan()` performs no full media processing;
  `run()` defaults to `offline=True` and never downloads a model implicitly.
- Inputs and output locations accept `str` or `pathlib.Path`. Returned paths are
  `Path` objects.
- Results are frozen dataclasses. Lists are exposed as tuples and JSON-shaped
  mappings are read-only snapshots.
- A normal run that needs human or host-agent work returns `status="review_required"`
  with exit code `3`. It is not silently promoted to a successful final result.
- Generated projects remain outside the source repository unless the caller
  explicitly chooses an output location there; the API does not copy source
  media into a review result.

## Functions

```python
from pathlib import Path
from video_script_reconstructor.api import (
    list_review_items,
    plan,
    run,
    show_review_item,
    validate,
)

preview = plan(Path("recording.mp4"), subtitles=[Path("recording.srt")])
result = run(
    Path("recording.mp4"),
    output_root=Path("outputs"),
    subtitles=[Path("recording.srt")],
    offline=True,
)
report = validate(result.project_dir)
pending = list_review_items(result.project_dir)
detail = show_review_item(result.project_dir, pending[0].review_id) if pending else None
```

### `plan(input_value, *, ...) -> Plan`

Inspects the input type, available media probe, transcript sources, expected
stages, prerequisites, storage estimate, and output path. It does not download
models, call an external service, or process the full input.

### `run(input_value, *, ...) -> RunResult`

Runs one input with the same core options as the canonical CLI: `output_root`,
`subtitles`, `transcript`, `preset`, `config_path`, `subtitle_mode`,
`language`, `fidelity_mode`, `vision_mode`, ASR chunk/overlap bounds,
`semantic_max_packets`, `resume`, `offline`, and the explicit
`allow_remote_download`/`allow_external_ai` switches. Provider injection,
progress callbacks, batch orchestration, and legacy semantic continuation are
intentionally not part of this first stable facade; use the CLI for those
owner-facing operations.

The returned `RunResult` contains `project_dir`, `markdown_path`, `status`,
`exit_code`, and an optional `ValidationResult`. Status and exit-code meanings
are defined in the [public contracts](public-contracts.md#exit-codes-and-result-status).

### `validate(project_dir, *, verify_metadata=True) -> ValidationResult`

Performs the independent output-contract, link, hash, metadata, chronology,
and state checks. A failed report is returned with `valid=False`; it is not
converted into a false success.

### `list_review_items(project_dir) -> tuple[ReviewItem, ...]`

Returns the immutable review queue snapshot. `show_review_item(project_dir,
review_id)` returns the same typed item with stored image paths, source IDs,
and competing evidence populated.

This first facade only reads review state. Applying corrections and finalizing
a project remain CLI operations until their mutation, authorization, and
rollback contract is separately versioned.

## Exceptions and compatibility

Expected missing paths are normalized to `InputError`. Other package errors
(`SecurityError`, `BlockedError`, `ValidationFailure`, `ReviewRequired`, and
`StaleRevisionError`) retain their normal meanings and are not swallowed.
Review-required and blocked outcomes from a completed pipeline run are returned
in `RunResult`; an exception is reserved for an operation that cannot produce
a trustworthy result.

The facade is versioned with the distribution. Additive fields may be added in
a minor release; existing fields keep their meaning. Removing or changing a
field, status, exception rule, or default requires a release note and a
migration path. Code should not import implementation modules or parse
`.state` files directly.
