# M281 — Negative-Test-Aware Training

**Date:** 2026-04-20
**File:** `experiments/m281_negative_aware_training.py`

## Purpose

Add negative prompts (with correct answers) to training set.

## Results

- Baseline: exact 3/3, negative 1/2, PPL 82.80
- Negative-aware: exact 3/3, negative 2/2, PPL 42.69

## Conclusion

🎯 **Negative-aware training improves robustness.** Negative: 50% → 100%.
