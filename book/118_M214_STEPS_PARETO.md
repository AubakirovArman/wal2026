# M214: Steps Pareto Frontier

**Status:** ✅ Complete
**Date:** 2026-04-30
**Model:** Llama-3.1-8B, K=256, 10 facts

## Question

What is the optimal training steps for factual editing? Build Pareto frontier.

## Results

| Steps | LoRA Δ | Re-enc Δ | Survival | Time |
|-------|--------|----------|----------|------|
| 50 | +0.017 | +0.021 | 0/10 | 11s |
| 100 | +0.302 | +0.305 | 2/10 | 15s |
| 200 | +0.463 | +0.472 | **3/10** | 27s |
| 300 | +1.542 | +1.530 | 3/10 | 41s |
| 400 | +0.743 | +0.747 | 0/10 | 49s |
| 600 | +6.401 | +6.577 | 2/10 | 74s |
| 800 | +1.958 | +1.994 | 2/10 | 101s |

## Pareto-Optimal Points

- steps=50: survival=0, ΔPPL=+0.021 ✅ (minimal penalty)
- steps=100: survival=2/10, ΔPPL=+0.305 ✅
- steps=200: survival=3/10, ΔPPL=+0.472 ✅ (best balance)

## Key Finding

**steps=200 is the sweet spot** for 10 facts on K=256.

More steps ≠ better survival! steps=300-800 show worse or equal survival with higher PPL damage. steps=600 shows catastrophic overfitting (PPL +6.6!).

## Practical Implication

For production: **steps=100-200** is optimal. Don't waste time on 400+ steps for standard facts — overfitting without survival gain.
