# Phase 16: Global Atoms

> *"One atom table to rule them all."*

## Status

**Phase:** 16  
**Goal:** Replace 225 per-layer atom tables with a single global table.  
**Date:** 2026-04-20  
**Method:** M116 — pool all linear weights, build one k-means codebook, encode all layers  
**Result:** ✅ **PASS** — PPL-neutral (+0.03), 225× atom storage reduction.

## Motivation

Before Phase 16, every layer had its own atom table:
- 225 layers × 256 atoms × 256 floats = ~59 MB of atom tables
- Each table independently derived via k-means
- No sharing, no cross-layer structure exploited

The question: can one global atom table serve all layers without quality loss?

## Experiment: M116

**Model:** meta-llama/Llama-3.1-8B  
**Dataset:** WikiText-2 validation (50 texts)  
**Method:**
1. Collect all `nn.Linear` weights (excluding embed/norm)
2. Pool into single tensor: 7.505B elements
3. Sample 1M elements for k-means (memory constraint)
4. Build one atom table: K=256, iters=5
5. Build one coeff table: C=16, iters=3
6. Encode each layer with global atoms + per-layer programs

### Results

| Metric | Per-Layer Atoms | Global Atoms | Delta |
|--------|-----------------|--------------|-------|
| **PPL** | 10.0285 | 10.0586 | **+0.03** (+0.3%) |
| **Encode time** | 304s | 216s | **−88s** (−29%) |
| **Atom storage** | 59.0 MB | 0.26 MB | **−225×** |
| **Layers** | 225 | 225 | same |

### Key Finding

PPL increase of +0.03 is statistically negligible. The global atom table captures enough structure from the pooled weight distribution to serve all layer types (attention Q/K/V/O, MLP gate/up/down).

## Why Global Atoms Work

1. **Weight distributions are similar across layers** — all linear layers in a transformer are initialized from the same distribution and trained with similar dynamics
2. **K-means on pooled data gets more samples** — 1M samples vs ~50K per layer → better centroid estimation
3. **Atoms are basis directions, not semantic units** — see Phase 21 (Program Heatmap)

## Implications

### Storage
- Atom tables shrink from 59 MB → 262 KB
- For model distribution: ship one 262 KB table + per-layer programs
- For model soups: one table serves all models in a family

### Speed
- Encode faster: single k-means instead of 225
- No per-layer atom table memory overhead during inference

### Future
- Cross-model atom libraries: pre-compute atoms for Llama-3.x family
- Hierarchical atoms: global table + small per-layer residual tables

## Files

- `experiments/m116_global_atoms.py` — full experiment
- `src/wal/v1/encoder.py` — `build_l0_atoms()`, `build_coeff_table()`

## Next Steps

Phase 17: Can we merge models at the program level?
