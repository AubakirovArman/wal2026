# M287 — Retrieval Contamination Stress Test

**Date:** 2026-04-20
**File:** `experiments/m287_retrieval_contamination_stress.py`

## Purpose

Test retrieval robustness against wrong/conflicting/irrelevant context.

## Results

- Pass: 7/7
- Contaminated: 1 (wrong context → London instead of Paris)

## Conclusion

⚠️ **Retrieval is vulnerable to wrong context.** 1/7 cases contaminated.
