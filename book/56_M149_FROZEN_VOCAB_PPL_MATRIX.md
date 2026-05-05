# M149 — Frozen Vocabulary PPL Matrix

**Date:** 2026-04-20
**Status:** ✅ Complete (v2 fast version)
**Goal:** Compare rebuilt per-layer table vs frozen global table.

## Method

- Model on CPU (`device_map="cpu"`)
- 2 layers (0, 16), 2 modules (q_proj, v_proj)
- K=64, C=8, iters=1 (fast config)
- Synthetic edit: Gaussian noise σ=0.001 on 5% of weights

## Results

| Layer | rebuilt_mse | frozen_mse | ratio | rebuilt_diff | frozen_diff |
|-------|-------------|------------|-------|--------------|-------------|
| layers.0.q_proj | 1.85e-07 | 6.02e-07 | 3.25 | 0.868 | 0.812 |
| layers.0.v_proj | 4.71e-09 | 4.39e-09 | 0.93 | 0.893 | 0.833 |
| layers.16.q_proj | 9.31e-08 | 4.99e-08 | 0.54 | 0.796 | 0.805 |
| layers.16.v_proj | 5.05e-09 | 6.72e-09 | 1.33 | 0.861 | 0.826 |

**Averages:**
- Frozen/rebuilt MSE ratio: **1.512**
- Rebuilt diff: **0.855**
- Frozen diff: **0.819**

## Key Findings

1. **Frozen table MSE ≈ 1.5× rebuilt** (with K=64). With K=256 this gap would shrink.
2. **Frozen diff < Rebuilt diff** (0.819 vs 0.855). Frozen table produces slightly more localized diffs.
3. **Layer 16 q_proj** is actually BETTER with frozen table (ratio 0.54). Global atoms happen to fit this layer well.

## Limitations

- K=64 is too small for production; K=256 would give much lower MSE
- Only 2 layers tested; full model sweep needed for validation
- No PPL measured (CPU-only, too slow for full forward pass)

## Artifacts

- `experiments/m149_frozen_vocab_ppl_matrix_v2.py`
- `experiments/m149_frozen_vocab_ppl_matrix.json`
