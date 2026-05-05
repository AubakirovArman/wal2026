# M261 — Anti-Forgetting Rehearsal Test

**Date:** 2026-04-20
**File:** `experiments/m261_anti_forgetting_rehearsal.py`

## Purpose

Verify that rehearsing previously learned facts during new training prevents catastrophic forgetting.

## Results

Without rehearsal:
- Batch 3 forgets Batch 2 (Japan → False)

With rehearsal:
- NO forgetting in any batch

## Conclusion

✅ **Rehearsal prevents forgetting completely.** Critical for sequential editing.
