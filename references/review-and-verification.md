# Review and verification

## Contents

- Review queue
- Corrections
- Status gates

## Review queue

List every unresolved item with severity, time, type, exact uncertainty, block/image links, competing evidence, required action, and whether it blocks full verification. Do not hide accepted uncertainty or semantic-pending evidence.

## Corrections

Reject unknown IDs. Preserve raw evidence. Append reviewer, time, rationale, old/new value, and decision. Rerender from canonical state and rerun affected audits after each correction.

## Status gates

Automatic checks may produce `automatically_checked`, never `fully_verified`. Use `review_required` for consequential uncertainty and `blocked` for unsafe continuation. Full verification requires passing all audits, zero unsupported claims, no blocking/high-severity unresolved item, and explicit attributable human sign-off.

