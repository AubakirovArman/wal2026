# Phase 17: Program Soup

> *"Averaging programs is not averaging weights."*

## Status

**Phase:** 17  
**Goal:** Test whether discrete WAL programs can be interpolated (model soup at program level).  
**Date:** 2026-04-20  
**Method:** M117 — encode base + edited model with global atoms, average programs, decode  
**Result:** ❌ **FAIL** — Discrete program interpolation destroys the model.

## Motivation

Model soups (averaging weights of multiple fine-tunes) are a powerful technique. If WAL programs could be interpolated directly:
- Model merging would be O(programs) instead of O(weights)
- Distributed editing: each user edits programs, merge by averaging IDs
- No re-encoding needed

## Experiment: M117

**Setup:**
1. Encode base model (Llama 3.1 8B) with global atoms
2. Encode edited model (same base + contrafactual LoRA edits) with same global atoms
3. Average programs: `prog_soup = (prog_base + prog_edited) / 2` (uint8 → int16)
4. Decode soup programs

### Results

| Metric | Base | Edited | Soup (α=0.5) |
|--------|------|--------|--------------|
| **PPL** | 10.03 | 12.95 | **6.4×10¹³** |
| **Contrafactuals** | 0/10 | 10/10 | — |

**Program similarity:**
- Atom ID match: 25%
- Coeff ID match: 6.6%

### Why It Fails

Averaging uint8 atom IDs is meaningless:
- Atom ID 47 + Atom ID 129 = Atom ID 88 — but atom 88 has no relation to 47 or 129
- The atom table is unordered — IDs are arbitrary k-means cluster labels
- Same for coefficients: coeff ID 3 + coeff ID 12 ≠ meaningful interpolation

## Correct Way to Merge WAL Models

```
Model A (WAL) → decode → dense weights A
Model B (WAL) → decode → dense weights B
Dense soup = (A + B) / 2  ← merge in WEIGHT space
Dense soup → re-encode → WAL soup
```

**Cross-model operations must happen in continuous weight space.**

## Lesson

WAL programs are a **storage format**, not a **differentiable manifold**. The discrete program space has no meaningful interpolation operator.

## Files

- `experiments/m117_program_soup.py`

## Next Steps

Phase 18: Can we use sparse residuals for variable bit rate?
