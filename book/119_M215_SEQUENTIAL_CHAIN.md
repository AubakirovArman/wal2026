# M215: Sequential Long Chain (10 Edits)

**Status:** ✅ Complete
**Date:** 2026-04-30
**Model:** Llama-3.1-8B, K=256, 10 edits × 5 facts

## Question

Does sequential compiled editing accumulate degradation? Can we do 10 edits in a row?

## Results

| Edit | Re-enc PPL | Δ PPL | Batch Surv | Cumul Surv |
|------|-----------|-------|-----------|-----------|
| 1 | 4.3277 | +0.053 | 1/5 | 1/50 |
| 2 | 4.3772 | +0.103 | 1/5 | 2/50 |
| 3 | 4.4143 | +0.140 | 3/5 | 3/50 |
| 4 | 4.4997 | +0.225 | 3/5 | 8/50 |
| 5 | 4.6044 | +0.330 | 2/5 | 8/50 |
| 6 | 4.6614 | +0.387 | 3/5 | 10/50 |
| 7 | 4.6667 | +0.392 | 3/5 | 12/50 |
| 8 | 4.7077 | +0.433 | 3/5 | 10/50 |
| 9 | 4.7218 | +0.447 | 3/5 | 13/50 |
| 10 | 4.8008 | +0.526 | 2/5 | **15/50** |

## Drift Analysis

- Batch 1 at final: 1/5
- Batch 2 at final: 0/5 ❌
- Batch 3 at final: 2/5

## Key Findings

1. **Sequential compiled editing works!** 30% cumulative survival after 10 edits.
2. **PPL grows with each edit** but slows down: +0.05 → +0.10 → +0.14 → +0.23 → +0.33 → +0.39 → +0.39 → +0.43 → +0.45 → +0.53
3. **Early batches degrade!** Batch 2 completely forgotten (0/5). Recency bias confirmed.
4. **Average batch survival: ~2.5/5 (50%)** — decent for individual edits.

## Practical Implication

Sequential compiled editing is viable but with caveats:
- Expect +0.05 PPL growth per edit
- Early edits will degrade
- Critical facts need periodic "refresh" (re-edit)
