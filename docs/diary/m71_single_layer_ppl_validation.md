# M71 SINGLE LAYER PPL VALIDATION

## Date
2026 (exact date from git log or experiment run)

## Goal
M71: Single-layer PPL validation of M65-M69 findings.

## Configuration
K=128, K_ATOMS=256, KMEANS_ITERS=5, iters=5, num_steps=0

## Method / What was tested
Replaces ONLY layer 40 o_proj with each method, measures PPL.
This validates whether single-layer output_relMSE correlates with full PPL.

Methods tested:
  A. M65 T=8 (worst vector quantization)
  B. M66 T=8,M=8 (best PQ)
  C. M67 two-tier T=8,M=4+4 (two-tier PQ)
  D. M69 K=128 (position-specific, SUSPECT)
  E. M69 K=256 (position-specific, OK by relMSE)
  F. WAL v2 (baseline)

## Result
PPL evaluation.
Likely negative result Has PASS/FAIL asserts

## Artifacts
- `experiments/m71_single_layer_ppl_validation.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Single-layer PPL validation. All methods PASS single-layer, but this does NOT predict full PPL.

**Notes:** M69 K=128: single-layer +0.0002, full +4.90. Difference up to 24,500×.


## Extracted Metrics (from source)

- PPL: .4
- PPL: .4
