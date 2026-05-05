# Phase 24: Cross-Layer Atom Correlation

> *"All 256 atoms are used everywhere."*

## Status

**Phase:** 24  
**Goal:** Measure atom overlap between different layers and layer types.  
**Date:** 2026-04-20  
**Method:** M124 — encode 7 sample layers with global atoms, compute Jaccard similarity  
**Result:** ✅ **PASS** — 100% cross-layer overlap. All atoms reused everywhere.

## Experiment: M124

**Layers sampled:** 0, 5, 10, 15, 20, 25, 30  
**Params per layer:** self_attn.o_proj, mlp.down_proj  
**Metric:** Jaccard similarity = |A ∩ B| / |A ∪ B|

### Results

| Comparison | Jaccard |
|------------|---------|
| Any layer vs any layer | **100.0%** |
| Attention vs MLP | **100.0%** |
| Shallow vs deep | **100.0%** |

**Unused atoms:** 0/256 (0.0%)

### Top Atom Frequencies (across all sampled layers)

| Rank | Atom ID | Frequency |
|------|---------|-----------|
| 1 | 240 | 0.82% |
| 2 | 37 | 0.81% |
| 3 | 45 | 0.81% |
| 4 | 122 | 0.80% |
| 5 | 199 | 0.80% |

Even the "most used" atom accounts for <1% of weights.

## Interpretation

**Every atom is used in every layer type.** There is no specialization:
- No "attention-only" atoms
- No "MLP-only" atoms
- No "early-layer" or "late-layer" atoms

This confirms the finding from Phase 21 (Program Heatmap): atoms are **basis directions**, not **semantic units**.

## Implications

### Global atoms are optimal
- One table truly serves all layers
- No need for per-layer or per-type atom subsets
- 225× storage savings is "free"

### For model soups
- Since all atoms are shared, cross-model transfer is about **program alignment**, not atom alignment
- Programs from different models use the same vocabulary
- But program interpolation is still invalid (see Phase 17)

### For future work
- Cross-model atom libraries: one table per model family is sufficient
- No benefit from learning "specialized" atoms per layer type
- Hierarchical atoms should be global at each level

## Files

- `experiments/m124_cross_layer_correlation.py`

## Next Steps

Phase 25: Final summary and roadmap update.
