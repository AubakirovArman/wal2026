# WAL Roadmap v5: M266–M290 — Execution Status

**Date:** 2026-04-20

## Completed Experiments

| Exp | Name | Status | Key Result |
|-----|------|--------|------------|
| M266 | CLI Smoke Test | ✅ | All 8 commands work end-to-end |
| M267 | Recipe DAG | ✅ | Branch/fork/merge work |
| M270 | CI Report Schema | ✅ | Unified JSON, NLS/CRS metrics |
| M271 | Semantic Diff Dashboard | ✅ | Human-readable project overview |
| M272 | Rollback Chain Test | ✅ | 5.94s rollback, 6.73s rebuild |
| M276 | Layer 16 Scale Test | ✅ | 96.7% survival on 50 facts |
| M277 | Layer Aperture Sweep | ✅ | All layers 3/3 on small set |
| M278 | Module Aperture Search | ✅ | All modules 3/3 on small set |
| M279 | Long Chain Rehearsal | ✅ | 96.7% across 10 batches |
| M280 | Batch Size Frontier | ✅ | 100% survival for 1-20 batch sizes |
| M281 | Negative-Aware Training | ✅ | Negative: 50% → 100% |
| M282 | Context Robustness Training | ✅ | 4/4 context variations |
| M283 | Paraphrase Augmentation | ⚠️ | HURTS survival (3/3 → 0/3) |
| M287 | Retrieval Contamination | ⚠️ | 7/7 pass, 1 contaminated |
| M288 | Hybrid Arbitration | ✅ | 4/4 pass, weights-first |
| M290 | Memory Tier Auto-Router | ⚠️ | 5/6 correct (83%) |

## Remaining Experiments

| Exp | Name | Priority |
|-----|------|----------|
| M273 | Multi-seed stability | Medium |
| M275 | Recipe signing | Medium |
| M284 | Old-answer lure training | Medium |
| M286 | Retrieval matcher v2 | Low |

## Wild Research Ideas (Backlog)

Recipe DNA, Model organism evolution, Edit immune system, Knowledge half-life,
Semantic GC, Branch marketplace, Edit package manager, Model hotfix system,
Canary edits, Shadow branch deploy, Auto-generated unit tests, Edit fuzzing,
Weight blame, Semantic bisect, Behavioral checksum, Auto release notes,
Diff-to-English, Edit conflict predictor, Layer aperture compiler,
Neural recipe optimizer, Hard-fact refusal tier, Memory provenance,
Live model patching, Model time travel.

## Production Stack v13

```text
Base:         Hadamard-WAL K=256, seed=42
Edit:         LoRA rank-4, layer 16 ONLY
Training:     FP32 adapters + gradient clipping + rehearsal + negative-aware
Determinism:  Fixed seed encode + recipe replay bit-exact
Caching:      Build cache idempotent, rollback 2.7× faster
Versioning:   Recipe DAG with branch/fork/merge + checkpoint stable
CI:           Unified schema with NLS/CRS metrics
Diff:         Recipe diff + semantic tensor diff
Dashboard:    Semantic diff dashboard
Scale:        Layer 16 handles 50 facts (96.7% survival)
Chain:        10 batches with rehearsal (96.7% survival)
Batch:        All sizes 1-20 achieve 100% survival
Negative:     Negative-aware training improves robustness
Context:      Context wrapping improves robustness
Arbitration:  Weights-first handles conflicts
Router:       Confidence-based auto-router (83% accuracy)
```
