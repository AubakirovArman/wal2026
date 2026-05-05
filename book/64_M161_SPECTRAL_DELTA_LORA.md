# M161 — Spectral Delta of LoRA

**Date:** 2026-04-20
**Status:** ✅ Complete
**Goal:** Analyze spectral sparsity of LoRA deltas via DCT.

## Method

- Synthetic LoRA deltas: `delta = A @ B * scale`
- 2D DCT decomposition
- Measure energy concentration in top-10% coefficients

## Results

| Rank | Scale | LL | LH | HL | HH | Top-10% Energy |
|------|-------|-----|-----|-----|-----|----------------|
| 1 | 1.0 | 0.248 | 0.248 | 0.252 | 0.252 | **0.387** |
| 4 | 1.0 | 0.248 | 0.248 | 0.252 | 0.252 | 0.295 |
| 8 | 1.0 | 0.248 | 0.250 | 0.250 | 0.252 | 0.279 |
| 4 | 3.0 | 0.251 | 0.247 | 0.253 | 0.249 | 0.295 |

## Key Finding

**Lower rank = more spectrally sparse delta.**

- rank=1: 38.7% of energy in top-10% DCT coefficients
- rank=4: 29.5%
- rank=8: 27.9%

## Why This Matters

This explains M138's rank=1 collapse:
1. rank=1 deltas are spectrally sparse (concentrated in few frequencies)
2. WAL quantization is most sensitive to sparse structure
3. Re-encode destroys the sparse pattern → catastrophic loss
4. Higher rank deltas are more diffuse → survive re-encode better

## Scale Invariance

Changing scale (1.0 vs 3.0) does NOT change spectral distribution — only amplitude. This confirms spectral signature is intrinsic to rank, not magnitude.

## Artifacts

- `experiments/m161_spectral_delta_lora.py`
- `experiments/m161_spectral_delta_lora.json`
