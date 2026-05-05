# M280 — Batch Size Frontier

**Date:** 2026-04-20
**File:** `experiments/m280_batch_size_frontier.py`

## Purpose

Find optimal batch size: 1/3/5/10/20 facts per batch.

## Results

- Batch 1: 1/1 = 100%, 10.5s
- Batch 3: 3/3 = 100%, 9.3s
- Batch 5: 5/5 = 100%, 9.1s
- Batch 10: 10/10 = 100%, 9.2s
- Batch 20: 20/20 = 100%, 9.4s

## Conclusion

🎯 **All batch sizes achieve 100% survival.** No degradation up to 20 facts.
