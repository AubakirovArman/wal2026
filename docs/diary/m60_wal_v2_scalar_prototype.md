# M60 WAL V2 SCALAR PROTOTYPE

## Date
2026 (exact date from git log or experiment run)

## Goal
M60: WAL v2 scalar prototype — single layer validation.

## Configuration
K_ATOMS=256, C_COEFFS=16, KMEANS_ITERS=10, LLOYD_MAX_ITERS=10, batch=1_048_576

## Method / What was tested
Compare WAL v2 (single-call + continuous coefficients) vs WAL-0 baseline
on layer 40 o_proj of Llama 3.3 70B.

## Result
PPL evaluation.

## Artifacts
- `experiments/m60_wal_v2_scalar_prototype.py`
- `experiments/m60_wal_v2_scalar_prototype.log`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** WAL v2 scalar prototype — single-call programs with continuous coefficients

**Notes:** Predecessor to M61. Proved that atom_id + coeff_id (12 bits/weight) can match dense quality.


## Extracted Metrics (from source)

- Time: .1
- relMSE: 0.00000454
- relMSE: 0.00001574
- relMSE: 0.00000454
- relMSE: 0.00001574
