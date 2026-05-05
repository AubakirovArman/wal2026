# M251 — Batch Rehearsal FP32 Fix (M235 Repaired)

**Date:** 2026-04-20
**File:** `experiments/m251_batch_rehearsal_fp32.py`

## Purpose

Repeat M235 batch+rehearsal experiment after M241 FP32 fix.
Test batch sizes 5 with none/random/low_survival rehearsal using FP32 adapters + gradient clipping on layer 16 only.

## Setup

```
Model: meta-llama/Llama-3.1-8B
Batch size: 5
Rehearsal modes: none, random, low_survival
Training: FP32 adapters + gradient clipping
Layer: 16 only
Seed: 42
```

## Results

| Mode | Batch 1 | Batch 2 | Batch 3 | Batch 4 | Batch 5 | Cumulative |
|------|---------|---------|---------|---------|---------|------------|
| none | 5/5 (Δ+0.03) | 5/5 (Δ+0.02) | 5/5 (Δ+0.39) | 5/5 (Δ+0.33) | 5/5 (Δ+0.20) | 25/25 |
| random | 5/5 (Δ+0.03) | 5/5 (Δ-0.06) | 5/5 (Δ-0.01) | 5/5 (Δ-0.04) | 5/5 (Δ+0.23) | 25/25 |
| low_survival | 5/5 (Δ+0.06) | 5/5 | 5/5 | 5/5 | 5/5 | 25/25 |

## Conclusion

✅ **FP32 fix restores batch editing completely.** All rehearsal modes achieve 100% cumulative survival. M235's 0/25 was a training-precision bug, not a conceptual failure.
