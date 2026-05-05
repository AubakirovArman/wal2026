# Phase 21: Program Heatmap

> *"Atoms are basis directions, not semantic neurons."*

## Status

**Phase:** 21  
**Goal:** Analyze atom usage patterns across all 225 layers to understand program structure.  
**Date:** 2026-04-20  
**Method:** M121 — encode full model with global atoms, compute per-layer statistics  
**Result:** ✅ **PASS** — High entropy (0.966/1.0), no semantic localization.

## Experiment: M121

**Metrics computed per layer:**
- **Entropy**: Shannon entropy of atom ID distribution (normalized to [0,1])
- **Top-3 dominance**: % of weights covered by the 3 most frequent atoms
- **Atom frequency**: how often each atom appears

### Results

| Statistic | Value |
|-----------|-------|
| **Average entropy** | **0.966 / 1.0** |
| **Average top-3 dominance** | **3.0%** |
| **Attention entropy** | 0.965 |
| **MLP entropy** | 0.968 |
| **Most specialized layer** | layer 0 q_proj (entropy 0.925, top-3 5.9%) |
| **Most diverse layer** | layer 5 mlp.up_proj (entropy 0.979, top-3 2.8%) |

### Interpretation

**High entropy = uniform atom usage**
- 256 atoms are used almost equally across all layers
- No atom dominates (top atom ~2% frequency)
- No layer-type specialization (Attention vs MLP almost identical)

**What this means:**
- Atoms are **basis directions** in weight space, not **semantic units**
- There is no "Paris atom" or "attention atom"
- Atoms form a shared linear basis that efficiently spans weight manifolds
- This is similar to how PCA components work — they're structural, not semantic

### Comparison to Semantic Hypothesis

| Hypothesis | Expected | Actual |
|------------|----------|--------|
| Semantic atoms | Some atoms dominate specific layers | All atoms used uniformly |
| Layer specialization | Attention ≠ MLP atom usage | Attention ≈ MLP |
| Knowledge localization | Few atoms = few concepts | 256 atoms ≈ 256 basis directions |

## Implications

### For editing
- Don't expect to find "the atom for X" — atoms are not disentangled
- Editing must work on **program distributions**, not individual atoms
- Global atom table is justified: atoms are generic basis vectors

### For compression
- Uniform usage means no simple pruning (all atoms are needed)
- Variable K per layer won't help much
- Hierarchical atoms might help: coarse basis + fine residuals

## Files

- `experiments/m121_program_heatmap.py`

## Next Steps

Phase 22: Can genetic algorithms find better programs than greedy encode?
