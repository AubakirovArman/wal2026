# M75 WAL V1 70B PPL

## Date
2026 (exact date from git log or experiment run)

## Goal
M75: WAL v1 full 70B PPL + round-trip verification.

## Configuration
K_ATOMS=256, C_COEFFS=16, KMEANS_ITERS=5, batch=1_048_576, max_l1=64, num_steps=0

## Method / What was tested
WAL v1 uses same encode as v2 (12 bits/weight) but adds hierarchical atom definitions.
Expected PPL: same as v2 (~2.778).

## Result
PPL evaluation.
Likely negative result Has PASS/FAIL asserts

## Artifacts
- `experiments/m75_wal_v1_70b_ppl.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** PPL 2.7809 vs baseline 2.7805 — delta +0.0004 PASS. 35,840 L1 atoms. 1866s encode.

**Notes:** WAL v1 hierarchical atoms proven on full 70B. Semantic layer works without quality loss.


## Extracted Metrics (from source)

- PPL: .4
- PPL: .4
- PPL: .4
- PPL: .4
- Time: .0
