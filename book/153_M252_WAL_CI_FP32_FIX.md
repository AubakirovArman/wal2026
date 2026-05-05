# M252 — WAL CI Pipeline FP32 Fix (M240 Repaired)

**Date:** 2026-04-20
**File:** `experiments/m252_wal_ci_fp32.py`

## Purpose

Repeat M240 CI pipeline with FP32 fix and layer-16-only strategy.

## Results

| Gate | Result | Weight |
|------|--------|--------|
| exact_match | 3/3 ✅ | 0.3 |
| paraphrase | 3/3 ✅ | 0.3 |
| negative | 1/2 ❌ | 0.2 |
| PPL | 1.82 ✅ | 0.2 |
| no_nan | True ✅ | — |

**Overall: FAIL** (negative test 50%)

## Conclusion

CI pipeline works but negative robustness remains weak (same as M234: 60%). Needs context robustness training (proposed M273).
