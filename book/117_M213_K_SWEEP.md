# M213: K Sweep for Compiled Edits

**Status:** ⚠️ Partial (K=1024 OOM)
**Date:** 2026-04-30
**Model:** Llama-3.1-8B, steps=400 strong edit

## Question

What is the optimal K for compiled WAL edits? Test K=[128, 256, 512, 1024].

## Results

| K | Encoded Δ | LoRA Δ | Re-enc Δ | Survival | Enc Time | Re-enc Time |
|---|-----------|--------|----------|----------|----------|-------------|
| 128 | +0.018 | +0.579 | +0.587 | 2/10 | 77s | 76s |
| 256 | +0.005 | +1.472 | +1.541 | **3/10** | 192s | 148s |
| 512 | +0.004 | +2.655 | +2.700 | 0/10 | 468s | 467s |
| 1024 | — | — | — | OOM | — | — |

## Key Finding

**K=256 is the sweet spot** for compiled strong edits:
- Best survival (3/10)
- Acceptable encode time (192s)
- But high PPL penalty (+1.54) for steps=400

**K=128:** Fast encode (~1.3 min), lower survival (2/10)
**K=512:** Catastrophic PPL (+2.7), very slow encode (~8 min), zero survival

## Practical Implication

For compiled strong edits (steps=400):
- K=128 — fast mode, lower quality
- K=256 — **balanced mode** (recommended)
- K=512+ — ineffective
