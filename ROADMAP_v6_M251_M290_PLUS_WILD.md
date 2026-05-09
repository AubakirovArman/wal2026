# WAL Roadmap v6: M251-M290+ + Wild Ideas — Complete

**Date:** 2026-05-03

## Completed Experiments (M251-M290)

| Exp | Name | Status | Key Result |
|-----|------|--------|------------|
| M251 | Batch Rehearsal FP32 | ✅ | 25/25 survival all modes |
| M252 | WAL CI FP32 | ⚠️ | Exact/para 100%, negative 50% |
| M253 | Deterministic Build | ✅ | Bit-exact across 3 runs |
| M254 | Recipe Replay | ✅ | Bit-exact rebuild |
| M255 | Seed Sensitivity | ✅ | Self-consistent, 5 unique hashes |
| M256 | Retrieval Matcher | ⚠️ | Easy/hard boundary blurred |
| M258 | Tier Compiler | ⚠️ | Routing 3/6, backend 6/6 |
| M259 | Build Cache | ✅ | Bit-exact idempotent |
| M260 | Recipe Diff | ✅ | 4/7 tensors changed, layer 16 |
| M261 | Anti-Forgetting | ✅ | Rehearsal prevents forgetting |
| M262 | Rollback Speed | ✅ | 2.7× faster than rebuild |
| M263 | Version Delta | ✅ | 4/226 tensors changed, layer 16 |
| M264 | Tag Stability | ✅ | Checkpoint bit-exact reload |
| M265 | CI Thresholds | ⚠️ | Negative test = best gate |
| M266 | CLI Smoke Test | ✅ | 8/8 commands work |
| M267 | Recipe DAG | ✅ | Branch/fork/merge |
| M269 | Cache Invalidation | ✅ | Seed/K/layer changes invalidate |
| M270 | CI Report Schema | ✅ | Unified JSON |
| M271 | Diff Dashboard | ✅ | Human-readable overview |
| M272 | Rollback Chain | ✅ | v0→v1→v2→v3, rollback to v1 |
| M273 | Multi-Seed Stability | ✅ | Behavior stable across 5 seeds |
| M275 | Recipe Signing | ✅ | HMAC-SHA256 tamper detection |
| M276 | Layer 16 Scale 50 | ✅ | 96.7% survival |
| M277 | Layer Sweep 10-20 | ✅ | All layers 3/3 |
| M278 | Module Aperture | ✅ | All modules 3/3 |
| M279 | Long Chain 10 Batches | ✅ | 96.7% survival |
| M280 | Batch Frontier 1-20 | ✅ | 100% all batch sizes |
| M281 | Negative-Aware Training | ✅ | Negative 50% → 100% |
| M282 | Context Robustness | ✅ | 4/4 variations |
| M283 | Paraphrase Augmentation | ❌ | WORSENS paraphrase 3/3 → 0/3 |
| M284 | Old-Answer Lure | ✅ | Lure resistance 2/3 → 3/3 |
| M286 | Retrieval Matcher v2 | ⚠️ | 3/8 matched |
| M287 | Contamination Stress | ⚠️ | 7/7 pass, 1 contaminated |
| M288 | Hybrid Arbitration | ✅ | Weights-first 4/4 |
| M289 | Retrieval Confidence | ✅ | Optimal threshold 0.6 |
| M290 | Memory Auto-Router | ⚠️ | 5/6 correct (83%) |

## Completed Wild Ideas (25 total)

| # | Name | Status |
|---|------|--------|
| 1 | Recipe DNA | ✅ |
| 2 | Evolution | ✅ |
| 3 | Immune System | ✅ |
| 4 | Semantic GC | ✅ |
| 5 | Package Manager | ✅ |
| 6 | Hotfix Pipeline | ✅ |
| 7 | Canary Edits | ✅ |
| 8 | Shadow Deploy | ✅ |
| 9 | Auto-Generated Tests | ✅ |
| 10 | Edit Fuzzing | ✅ |
| 11 | Weight Blame | ✅ |
| 12 | Semantic Bisect | ✅ |
| 13 | Behavioral Checksum | ✅ |
| 14 | Release Notes | ✅ |
| 15 | Diff-to-English | ✅ |
| 16 | Conflict Predictor | ✅ |
| 17 | Layer Compiler | ✅ |
| 18 | Neural Optimizer | ✅ |
| 19 | Refusal Tier | ✅ |
| 20 | Memory Provenance | ✅ |
| 21 | Live Patching | ✅ |
| 22 | Time Travel | ✅ |

## Production Stack v15

```text
Base:         Hadamard-WAL K=256, seed=42
Edit:         LoRA rank-4, layer 16 ONLY
Training:     FP32 adapters + gradient clipping + rehearsal + negative-aware + lure-aware
Determinism:  Fixed seed encode + recipe replay bit-exact + multi-seed stable
Caching:      Build cache idempotent, rollback 2.7× faster
Versioning:   Recipe DAG with branch/fork/merge + signing + checkpoint stable
CI:           Unified schema with NLS/CRS metrics + auto-generated tests
Diff:         Recipe diff + semantic tensor diff + diff-to-English
Dashboard:    Semantic diff dashboard
Scale:        Layer 16 handles 50 facts (96.7% survival)
Chain:        10 batches with rehearsal (96.7% survival)
Batch:        All sizes 1-20 achieve 100% survival
Negative:     Negative-aware + lure-aware training improves robustness
Context:      Context wrapping improves robustness
Arbitration:  Weights-first handles conflicts
Router:       Confidence-based auto-router (83% accuracy) + threshold 0.6
Signing:      HMAC-SHA256 tamper detection
Safety:       Immune system + quarantine + refusal tier
Deployment:   Canary + shadow + hotfix pipeline
Debug:        Weight blame + semantic bisect + behavioral checksum
Optimization: Neural recipe optimizer + layer compiler
Audit:        Memory provenance + release notes
```

## Next Steps

1. **Public CLI Demo**: `wal init → edit → build → test → tag → rollback`
2. **Full Integration Test**: End-to-end system validation
3. **Performance Benchmarks**: Throughput and latency metrics
4. **Documentation**: User guide and API reference
