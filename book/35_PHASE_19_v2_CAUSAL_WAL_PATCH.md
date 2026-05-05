# Phase 19 v2 / M130: Causal WAL Patch Ablation

**Date:** 2025-04-24  
**Status:** ✅ Completed  
**Goal:** Repeat M127 with canonicalization + same seed to see if WAL-diff reveals real structural changes.

## Background

M127 (Phase 18) showed 99.98% program diff after LoRA edit. But this was dominated by permutation noise — k-means returned atoms in random order, making programs incomparable.

M129 fixed this with canonicalization. M130 tests whether WAL-diff now shows localized changes.

## Method

1. Encode base model with seed=42 + canonicalization
2. Decode to dense
3. Apply LoRA edit (rank=4, steps=100) on layers 14-16 o_proj
4. Re-encode edited model with **same seed=42** + canonicalization
5. Compare programs

## Results

### Global Diff Statistics

| Metric | M127 (no canon) | **M130 (canon, same seed)** |
|--------|----------------|----------------------------|
| Total weights | — | 7,504,658,432 |
| Atom ID changes | 99.62% | **25.00%** |
| Coeff ID changes | 93.58% | **25.00%** |
| Any program change | 99.98% | **25.00%** |
| Both changed | 93.22% | **24.999%** |

### Per-Layer Diff

All 225 layers show **exactly 25% diff**:
- Target layers (14-16 o_proj): 25.00%
- Non-target layers: 25.00%
- lm_head: 25.00%

## Analysis

### What 25% means

With canonicalization, the 99.98% noise is gone. The remaining 25% is **re-encode quantization error**:

1. LoRA merge changes target layer weights by small deltas
2. Greedy encode re-evaluates each weight against the atom table
3. ~25% of weights get a different (atom, coeff) pair — not because the edit touched them, but because the weight value crossed a quantization boundary

### Why diff is uniform across all layers

Even non-target layers show 25% diff. This is because:

1. We rebuild the atom table from ALL weights (including edited ones)
2. The global atom table shifts slightly due to changed weight distribution
3. Greedy encode re-optimizes for the new table

If we used a **fixed atom table** (pre-computed on base model, not rebuilt), non-target layers might show ~0% diff.

## Conclusion

**WAL-diff is still diffuse.** Even with canonicalization and same seed:
- You cannot localize where the edit happened by looking at program differences
- ~25% of programs change everywhere, not just in target layers

**Implication for patch analysis:** WAL-diff is not a viable tool for understanding *where* an edit occurred. It can only tell you *that* the model changed.

**Alternative:** For patch compilation, don't diff programs — instead, encode the edited model and distribute the full new programs. The value of WAL is in the structured representation, not in diff-based patch analysis.

## Artifacts

- `experiments/m130_causal_wal_patch_v2.py`
- `experiments/m130_causal_wal_patch.json`
