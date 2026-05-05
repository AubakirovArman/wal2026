# M290 — Memory Tier Auto-Router

**Date:** 2026-04-20
**File:** `experiments/m290_memory_tier_auto_router.py`

## Purpose

Automatically route facts to weights/retrieval/hybrid based on confidence probe.

## Results

- Correct routes: 5/6
- weights: 2, retrieval: 2, hybrid: 2
- 1 incorrect: UK currency routed to retrieval but model knows it

## Conclusion

⚠️ **Auto-router works but needs threshold tuning.** 83% accuracy.
