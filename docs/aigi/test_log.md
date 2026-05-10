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

## M684 — Behavioral Contracts

- Status: `PASS`
- Checks: `4`
- Passed: `4`

## M685 — Experience To Memory

- Status: `PASS`
- Cases: `8`
- Passed: `8`

## M686 — Verified Feedback Loop

- Status: `PASS`
- Episodes: `25`
- Passed: `25`

## M687 — Contract-Gated Rollback

- Status: `PASS`
- Checks: `5`
- Passed: `5`

## M689 — Memory Change Budget

- Status: `PASS`
- Checks: `7`
- Passed: `7`

## M690 — Risk Ledger

- Status: `PASS`
- Checks: `8`
- Passed: `8`

## M691 — Contract Regression Suite

- Status: `PASS`
- Protected contracts: `10`
- Checks: `6`
- Passed: `6`

## M692 — Commit Decision Report

- Status: `PASS`
- Checks: `7`
- Passed: `7`
