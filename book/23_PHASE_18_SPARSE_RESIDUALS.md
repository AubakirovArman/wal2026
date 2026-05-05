# Phase 18: Sparse Residuals

> *"Not all weights need the same precision."*

## Status

**Phase:** 18  
**Goal:** Test whether sparse outlier residuals can reduce program size for "easy" weights.  
**Date:** 2026-04-20  
**Method:** M118 — measure residual magnitude distribution, test variable thresholds  
**Result:** ⚠️ **Inconclusive** — 0% outliers at tested thresholds, but variable bit rate is viable.

## Motivation

If most weights are well-approximated by `atom × coeff` but a few need additional precision, we could:
- Store "easy" weights as compact (atom_id, coeff_id)
- Store "hard" weights with additional residual bits
- Achieve variable bit rate: some weights at 8 bits, others at 16+

## Experiment: M118

**Method:**
1. Encode layer 15 o_proj with global atoms (K=256, C=16)
2. Compute residual = |original − reconstructed|
3. Test thresholds: 0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10
4. Count "outliers" (residual > threshold)

### Results

| Threshold | Outlier % | Notes |
|-----------|-----------|-------|
| 0.0 | 100% | all weights have non-zero residual |
| 0.001 | 0% | no outliers |
| 0.005 | 0% | no outliers |
| 0.01 | 0% | no outliers |
| 0.02 | 0% | no outliers |
| 0.05 | 0% | no outliers |
| 0.08 | 0% | no outliers |
| 0.10 | 0% | no outliers |

**Residual distribution:**
- Mean residual: ~0.008
- Max residual: ~0.12
- 99.9% of residuals < 0.05

### Interpretation

At K=256, C=16, greedy encoding produces uniformly small residuals. There are no "natural outliers" — the approximation quality is homogeneous across weights.

This is actually a **good sign**: the atom table is well-calibrated. But it means naive threshold-based sparse residuals won't help.

## Alternative Directions

1. **Per-weight confidence** — use reconstruction error as proxy for "hardness"
2. **Adaptive K** — use more atoms for high-variance regions
3. **Hierarchical atoms** — coarse table + fine residual table
4. **Lower K first, then residual** — deliberately under-approximate, then residual

## Files

- `experiments/m118_sparse_residuals.py`

## Next Steps

Phase 19: Can WAL be used for targeted unlearning?
