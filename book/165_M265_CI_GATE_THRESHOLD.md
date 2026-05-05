# M265 — CI Gate Threshold Calibration

**Date:** 2026-04-20
**File:** `experiments/m265_ci_gate_threshold.py`

## Purpose

Determine optimal thresholds for CI gates by measuring score distributions.

## Results

- EXACT: cannot distinguish good/bad (all 1.0)
- NEGATIVE: best discriminator (good=1.0, bad=0-1.0)
- PPL: overlapping ranges (good=398-943, bad=482-1085)

## Conclusion

⚠️ **Negative test is the most discriminative CI gate.** Exact and PPL have limited discriminative power.
