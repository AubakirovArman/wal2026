# M272 — Rollback Chain Test

**Date:** 2026-04-20
**File:** `experiments/m272_rollback_chain_test.py`

## Purpose

Test v0→v1→v2→v3 chain, rollback to v1, rebuild v3 from deltas.

## Results

- Rollback to v1: 5.94s, accuracy 7.6e-06
- Rebuild v3: 6.73s, accuracy 1.5e-05
- v3 survival: 3/3 facts
- Rebuild survival: identical

## Conclusion

✅ **Rollback chain works.** Delta-based rebuild is fast and accurate.
