# M262 — Rollback Speed Test

**Date:** 2026-04-20
**File:** `experiments/m262_rollback_speed.py`

## Purpose

Measure rollback speed vs full rebuild by applying pre-computed deltas.

## Results

- Build V1 time: 11.5s
- Rollback time: 4.3s
- Speedup: 2.7×
- Precision: max_diff = 7.6e-06 (bf16 rounding, negligible)

## Conclusion

✅ **Rollback is 2.7× faster than rebuild.** Precision loss is negligible.
