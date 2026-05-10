# M682 — AIGI Memory Tier Routing

Date: 2026-05-10
Status: PASS
Result: `experiments/m682_aigi_memory_tier_routing_results.json`
Doc: `docs/aigi/test_log.md`

## Purpose

Verify deterministic routing from memory kind to tier.

## Outcome

Stable facts route to `wal_recipe` when enabled, volatile/hard facts route to retrieval, unsafe memories route to refusal, procedures route to tool, and zero confidence routes to reject.
