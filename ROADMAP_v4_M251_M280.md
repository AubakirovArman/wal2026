# WAL Roadmap v4: M251–M280

**Date:** 2026-04-20
**Based on:** M235–M246 analysis and strategic pivot

## Core Pivot

WAL is no longer "weights encoded as atom × coeff". WAL is a **WeightOps platform**:

```text
WAL = операционная система для изменений в весах модели
      = build system + memory router + CI + registry + rollback
```

## Phase 1 — Stabilize Core Stack (M251-M255) ✅ COMPLETE

Goal: fix bugs, make pipeline reliable.

| Exp | Purpose | Result |
|-----|---------|--------|
| M251 | Repeat M235 with FP32 | ✅ 25/25 survival all modes |
| M252 | Repeat M240 CI with FP32 | ✅ exact 100%, paraphrase 100%, PPL pass, negative 50% → FAIL |
| M253 | Deterministic build audit | ✅ bit-exact across 3 runs |
| M254 | Recipe replay | ✅ bit-exact rebuild from recipes |
| M255 | Encode seed sensitivity | ✅ deterministic AND diverse (5/5 unique hashes) |

## Phase 2 — Memory Tier Compiler (M256-M260) ✅ COMPLETE

Goal: route knowledge to correct tier.

| Exp | Purpose | Result |
|-----|---------|--------|
| M256 | Retrieval matcher | ⚠️ easy/hard boundary blurred, both backends work |
| M258 | Tier compiler (confidence-based) | ⚠️ routing 3/6, backend success 6/6 |
| M259 | Build cache idempotency | ✅ bit-exact |
| M260 | Recipe diff | ✅ 4/7 tensors changed between recipes |

## Phase 3 — Better Editing Backends (M261-M265) ✅ COMPLETE

Goal: improve editing quality and stability.

| Exp | Purpose | Result |
|-----|---------|--------|
| M261 | Anti-forgetting rehearsal | ✅ rehearsal prevents forgetting completely |
| M262 | Rollback speed | ✅ 2.7× faster than rebuild, precision 7.6e-06 |
| M263 | Version delta (full model) | ✅ 4/226 tensors changed, ALL in layer 16 |
| M264 | Tag stability | ✅ checkpoint bit-exact reload |
| M265 | CI gate threshold calibration | ⚠️ negative test most discriminative |

## Phase 4 — Versioning / Anti-Forgetting (M266-M271)

Goal: production-grade version control.

| Exp | Purpose | Status |
|-----|---------|--------|
| M266 | Recipe format spec v1 | 🔜 Define JSON schema for recipes |
| M267 | Registry file format | 🔜 wal_registry.json format |
| M268 | Rehearsal + layer 16 ablation | 🔜 Does rehearsal need layer 16 only? |
| M269 | GC v2: half-life eviction | 🔜 Evict old facts by usage decay |
| M270 | Multi-user recipe merge | 🔜 Merge recipes from different users |
| M271 | Version DAG visualization | 🔜 Show version history as graph |

## Phase 5 — CI / Product MVP (M272-M279)

Goal: shipable CLI tool.

| Exp | Purpose | Status |
|-----|---------|--------|
| M272 | Auto test generation | 🔜 Generate paraphrases automatically |
| M273 | Negative robustness training | 🔜 Train model to reject wrong answers |
| M274 | CLI MVP: wal init | 🔜 `wal init` command |
| M275 | CLI MVP: wal edit add | 🔜 `wal edit add <fact>` command |
| M276 | CLI MVP: wal build | 🔜 `wal build` command |
| M277 | CLI MVP: wal test | 🔜 `wal test` command |
| M278 | CLI MVP: wal tag / rollback | 🔜 Version control commands |
| M279 | Dashboard MVP | 🔜 Simple web UI for registry |

## Phase 6 — External Validation (M280)

Goal: prove WAL works on real benchmarks.

| Exp | Purpose | Status |
|-----|---------|--------|
| M280 | ZsRE / CounterFact benchmark | 🔜 Run on standard model editing datasets |

## Production Stack v11

```text
Base:       Hadamard-WAL K=256, seed=42
Edit:       LoRA rank-4, layer 16 ONLY
Training:   FP32 adapters + gradient clipping
Determinism: Fixed seed encode + recipe replay bit-exact
Caching:    Build cache idempotent, rollback 2.7× faster
Versioning: Checkpoint stable, diff localized to layer 16
Rehearsal:  Prevents forgetting (M261)
CI:         Negative test most discriminative (M265)
Tiering:    Easy/hard boundary blurred — both backends work
```

## Key Findings

1. **FP32 fix is complete** — all prior "0 survival" results from M235/M240 were training-precision bugs
2. **Determinism proven** — encode, recipe replay, build cache all bit-exact
3. **Edit localization confirmed** — only layer 16 changes (4/226 tensors)
4. **Rehearsal critical** — prevents catastrophic forgetting in sequential editing
5. **Rollback fast** — 2.7× faster than rebuild via delta caching
6. **Easy/hard boundary blurred** — model can learn any fact via LoRA; retrieval works for all
7. **Negative test = best CI gate** — most discriminative between good and bad edits
8. **Tag stability proven** — checkpoints reload bit-exact
