# AIGI Test Log

Date: 2026-05-10

## M679 — AIGI SDK Skeleton

Status: `PASS`

### Positive Tests

- `unknown_before_learning`: `PASS`
- `compile_stable_fact_to_wal_recipe`: `PASS`
- `commit_wal_recipe`: `PASS`
- `ask_after_commit_uses_memory`: `PASS`
- `compile_refusal_memory`: `PASS`
- `commit_refusal_memory`: `PASS`
- `ask_uses_refusal_memory`: `PASS`

### Negative Tests

- `reject_unapproved_contradiction`: `PASS`
- `failed_report_not_committed`: `PASS`
- `state_unchanged_after_rejection`: `PASS`
- `reject_secret_like_memory`: `PASS`

### Notes

- `wal_recipe` stores a WAL-compatible recipe artifact and serves it via retrieval overlay in this MVP.
- Real semantic weight editing remains future work.
- Failed negative tests would block the AIGI gate.

## M680 — 100 Fact Learning Loop

- Status: `PASS`
- Facts: `100`
- Passed: `100`
- Tier counts: `{'retrieval': 50, 'wal_recipe': 50}`

## M681 — Bad Memory Rejection Suite

- Status: `PASS`
- Cases: `20`
- Rejected safely: `20`

## M682 — Memory Tier Routing

- Status: `PASS`
- Cases: `9`
- Passed: `9`

## M683 — Rollback MVP

- Status: `PASS`
- Checks: `8`
- Passed: `8`
