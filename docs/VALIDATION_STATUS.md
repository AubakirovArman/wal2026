# Validation Status

Date: 2026-05-10

This file is the short public validation ledger. It separates real local checks, controlled runtime/artifact gates, simulations, and blocked work.

## Current Gates

| Area | Gate | Status | Notes |
|------|------|--------|-------|
| Core package | `pytest -q tests` | PASS | 30 maintained tests pass locally. |
| Result schema | `python -m wal validate-results experiments --fail-on-invalid` | PASS | 481/481 result files valid. |
| Inventory | M624 | PASS | 813 experiment scripts, 0 parse failures. |
| Safe sweep | M625 | PASS | 285 safe scripts pass, 528 scripts blocked by policy. |
| Public claims | M630/M671 | PASS | Release-facing docs keep pre-alpha wording. |
| Docs smoke | M631 | PASS | 65/65 fast reviewer commands pass. |
| Small models | M632/M633/M635 | PASS | SmolLM2-360M, Qwen2.5-0.5B, TinyLlama-1.1B controlled runtime/artifact workflows pass. |
| Cross-model aggregate | M636-M638 | PASS | 3 unique local model paths; runtime/artifact protocol only. |
| Gemma small | M634 | BLOCKED | No local Gemma-small snapshot yet. |
| Hard-facts hybrid backend | M645 | SIMULATED | No real hybrid backend execution yet. |
| 24h soak | M666 | BLOCKED | Requires a real long-duration runner. |
| Memory long run | M667 | SIMULATED | Short memory sentinel only. |
| Legacy manifest | M677 | PASS | 813 scripts classified by runner type and review status. |
| Legacy audit M1-M50 | M678 | PASS | 143 scripts audited; 133 controlled-model, 3 slow-runner, 7 still-valid-needs-schema-v1, 0 current public claims. |
| AIGI verified feedback memory loop | M679-M687 | PASS | M680: 100/100 facts; M681: 20/20 rejected bad memories; M682: 9/9 routing; M683: 8/8 rollback; M684: 4/4 contracts; M685: 8/8 extraction; M686: 25/25 feedback; M687: 5/5 contract rollback. |
| Single-file context | M688 | PASS | `WAL_AIGI_FULL_CONTEXT.md` collects current architecture, metrics, gates, limitations, commands, and next steps. |

## Non-Claims

- These gates do not prove production readiness.
- The small-model gates do not perform semantic weight-edit training.
- Deployment gates are local prototypes/simulations unless explicitly marked otherwise.
- Historical generated `A+` or `certified` artifacts are audit history, not current release claims.
- Legacy audit entries without `wal.results.v1` artifacts are not current public claims.
- AIGI M679-M687 do not prove autonomous AGI or real semantic weight editing; `wal_recipe` is still a recipe artifact plus retrieval overlay.
