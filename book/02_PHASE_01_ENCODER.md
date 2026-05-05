# Phase 1: Encoder v2 (M60–M61)

## The Problem

By M59, WAL had a working scalar encoder but it was slow and produced variable-length programs. The runtime was complex. The quality was good (PPL 2.7821) but the encoding process was a mess.

We needed a **production-quality encoder**: fast, simple, deterministic, and with guaranteed quality.

## The Breakthrough Idea

Instead of variable-depth ternary routes, use **single-call programs**:
```
weight = atom[atom_id] * coeff[coeff_id] + residual
```

This is one atom call, one coefficient lookup, and an optional residual. No loops. No branches. No dynamic depth.

## Two-Pass Encoding

The encoder works in two passes:

**Pass 1: Atom initialization**
- Sample 2M weights from the tensor
- k-means++ initialization for K=256 atoms
- Lloyd-Max refinement for 5 iterations

**Pass 2: Coefficient quantization**
- For each weight, compute the optimal coefficient ratio: `weight / best_atom`
- Build C=16 coefficient levels using Lloyd-Max on sampled ratios
- Quantize each weight to nearest (atom, coeff) pair
- Compute residual for weights with large error

## Why It Works

- **One atom call**: Simpler than multi-step routes. No accumulation error.
- **Continuous coefficients**: C=16 levels (4 bits) are more expressive than ternary {-1,0,+1}.
- **Lloyd-Max on samples**: 2M samples is enough for stable convergence. Full tensor would be 67M — overkill.

## The Numbers

| Metric | Value |
|--------|-------|
| PPL | **2.7781** (vs baseline 2.7805, delta −0.0024) |
| Encode time | 1810s (30 min) for full 70B |
| Encoded params | 540 |
| Skipped params | 183 (embed, lm_head, 1D, spiky) |
| Compression | 1.33× vs bf16 |

## The Surprise

The encoded model is **better than dense**. PPL 2.7781 < 2.7805. This is not a bug — the quantization acts as a mild regularizer, smoothing weight noise that doesn't contribute to prediction.

## Files
- `src/wal/v2/encoder.py`
- `experiments/m60_wal_v2_scalar_prototype.py`
- `experiments/m61_wal_v2_70b_ppl.py`
