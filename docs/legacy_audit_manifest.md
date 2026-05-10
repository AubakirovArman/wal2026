# Legacy Experiment Manifest

Date: 2026-05-10

This manifest classifies every `experiments/*.py` script into a runner type and a modern review status.

## Summary

- Total scripts: `820`
- With historical artifacts: `321`
- With `wal.results.v1` artifacts: `76`
- Current public claim allowed after audit: `61`

## Review Status Counts

- `blocked_by_policy`: `2`
- `blocked_needs_controlled_model_runner`: `390`
- `blocked_needs_dry_run`: `38`
- `blocked_needs_model_small_runner`: `7`
- `blocked_needs_slow_runner`: `3`
- `blocked_needs_subprocess_review`: `5`
- `doc_or_meta_only`: `86`
- `still_valid`: `61`
- `still_valid_needs_schema_v1`: `228`

## Runner Type Counts

- `blocked_review`: `2`
- `docs_public_claims`: `86`
- `gpu_or_model_controlled`: `390`
- `model_small`: `7`
- `mutation_dry_run`: `38`
- `safe_core`: `7`
- `safe_core_with_artifact`: `282`
- `slow_safe`: `3`
- `subprocess_review`: `5`

## Policy

- `still_valid` means a script passed the modern safe sweep and has a schema-v1 result artifact.
- `still_valid_needs_schema_v1` means the script still runs but is not allowed as a current public claim until it emits `wal.results.v1`.
- Blocked statuses are not failures; they require controlled runners such as GPU/model, slow, mutation dry-run, or subprocess review.
