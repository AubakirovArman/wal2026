# M74 WAL V1 TWO TERM PROTOTYPE

## Date
2026 (exact date from git log or experiment run)

## Goal
M74: WAL-1 Two-Term Subroutine Prototype (GPU-batched).

## Configuration
K_ATOMS=256, C_COEFFS=16, KMEANS_ITERS=5, iters=5

## Method / What was tested
All heavy ops on GPU with batching. Clean GPU memory first.

## Result
Encode test.

## Artifacts
- `experiments/m74_wal_v1_two_term_prototype.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Two-term greedy (32 bits) excellent relMSE but subroutine clustering (256 subs, 12 bits) toxic (relMSE 0.04).

**Notes:** Lesson: diversity of optimal pairs too high for template clustering.
