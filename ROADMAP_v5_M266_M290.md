# WAL Roadmap v5: M266–M290

**Date:** 2026-04-20
**Based on:** M251-M265 results and strategic pivot

## Core Pivot

WAL is no longer "weights encoded as atom × coeff". WAL is a **WeightOps platform**:

```text
WAL = recipe + deterministic build + CI + registry + rollback
Model version = Base hash + Recipe DAG + Build seed + CI report
```

## New Rules (Mandatory)

1. **All adapters — FP32**
2. **All builds — seeded**
3. **All edits — via recipes**
4. **All diffs — recipe diff + semantic tensor diff, NOT checkpoint byte diff**
5. **All sequential edits — with rehearsal**
6. **All CI — exact + paraphrase + negative + context, negative weight high**
7. **Layer 16 — default first edit aperture**
8. **Hard facts — route/probe first, don't blindly weight-edit**

## Completed Phases

### Phase 1 — Stabilize Core Stack (M251-M255) ✅

| Exp | Result |
|-----|--------|
| M251 | ✅ 25/25 survival all modes |
| M252 | ✅ exact 100%, paraphrase 100%, negative 50% → FAIL |
| M253 | ✅ bit-exact across 3 runs |
| M254 | ✅ bit-exact recipe replay |
| M255 | ✅ deterministic AND diverse (5/5 unique hashes) |

### Phase 2 — Memory Tier Compiler (M256-M260) ✅

| Exp | Result |
|-----|--------|
| M256 | ⚠️ easy/hard boundary blurred |
| M258 | ⚠️ routing 3/6, backend success 6/6 |
| M259 | ✅ build cache idempotent |
| M260 | ✅ recipe diff works |

### Phase 3 — Better Editing Backends (M261-M265) ✅

| Exp | Result |
|-----|--------|
| M261 | ✅ rehearsal prevents forgetting |
| M262 | ✅ 2.7× faster rollback |
| M263 | ✅ edit localized to layer 16 (4/226 tensors) |
| M264 | ✅ checkpoint stable |
| M265 | ⚠️ negative test most discriminative |

---

## Phase A — Build System MVP (M266-M275)

Goal: public demo `wal init → edit add → build → test → tag → rollback`

| Exp | Purpose | Priority |
|-----|---------|----------|
| **M266** | Full WAL CLI smoke test | 🔥 CRITICAL |
| M267 | Recipe DAG build (branch/fork/merge) | High |
| M268 | Recipe-level branch merge | High |
| M269 | Build cache invalidation rules | Medium |
| **M270** | CI report schema (unified JSON) | 🔥 CRITICAL |
| M271 | Semantic diff dashboard | Medium |
| M272 | Rollback chain test (v0→v1→v2→v3→rollback) | High |
| M273 | Multi-seed stability test | Medium |
| M274 | Cross-hardware determinism | Low |
| M275 | Recipe signing / tamper detection | Medium |

## Phase B — Reliable Edit Compiler (M276-M285)

Goal: understand and optimize edit backend

| Exp | Purpose | Priority |
|-----|---------|----------|
| **M276** | Layer 16 scale test: 50/100 facts | 🔥 CRITICAL |
| M277 | Layer aperture sweep 0-31 | High |
| M278 | Module aperture search (q/k/v/o/gate/up/down) | High |
| M279 | Long chain: 20 sequential batches with rehearsal | Medium |
| M280 | Batch size frontier: 1/3/5/10/20 | Medium |
| **M281** | Negative-test-aware training | 🔥 CRITICAL |
| M282 | Context robustness training | High |
| M283 | Paraphrase augmentation (5 per fact) | Medium |
| M284 | Old-answer lure training | Medium |
| M285 | Fact difficulty probe v2 | Low |

## Phase C — Memory Router (M286-M290)

Goal: weights vs retrieval decision

| Exp | Purpose | Priority |
|-----|---------|----------|
| M286 | Retrieval matcher v2 (exact + fuzzy + embedding) | High |
| M287 | Retrieval contamination stress test | Medium |
| M288 | Hybrid answer arbitration | Medium |
| M289 | Retrieval confidence threshold | Medium |
| M290 | Memory tier auto-router | High |

## Wild Research Ideas (Backlog)

1. Recipe DNA / model organism evolution
2. Edit immune system (CI as immunity)
3. Model memory allocator (RAM/SSD/cache = weights/retrieval/prompt/tool)
4. Knowledge half-life + semantic GC
5. Branch marketplace + edit package manager
6. Model hotfix system + canary edits
7. Shadow branch deploy
8. Auto-generated unit tests + edit fuzzing
9. Weight blame + semantic bisect
10. Behavioral checksum + auto release notes
11. Diff-to-English
12. Edit conflict predictor
13. Layer aperture compiler
14. Neural recipe optimizer
15. Hard-fact refusal tier
16. Memory provenance
17. Live model patching
18. Model time travel

## Production Stack v12

```text
Base:         Hadamard-WAL K=256, seed=42
Edit:         LoRA rank-4, layer 16 ONLY
Training:     FP32 adapters + gradient clipping + rehearsal
Determinism:  Fixed seed encode + recipe replay bit-exact
Caching:      Build cache idempotent, rollback 2.7× faster
Versioning:   Checkpoint stable, diff localized to layer 16
CI:           Negative test most discriminative (M265)
Diff:         Recipe diff, NOT checkpoint byte diff
Registry:     Recipe DAG with branch/fork/merge
```

## MVP Demo Script

```bash
wal init
wal edit add facts.json
wal build
wal test
wal tag v1
wal rollback v0
wal diff v0 v1
```

What to show:
1. Build bit-exact
2. Recipe replay bit-exact
3. Edit localized to layer 16
4. CI catches bad edits
5. Rollback works
6. Registry can publish branch
