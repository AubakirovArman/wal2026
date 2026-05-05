# M67 PQ SYSTEMATIC

## Date
2026 (exact date from git log or experiment run)

## Goal
M67: Systematic PQ sweep + two-tier residual encoding.

## Configuration
K=16, C=4, K_ATOMS=256, KMEANS_ITERS=5, iters=5

## Method / What was tested
Goal: find best compression/quality tradeoff for PQ-based WAL-1.

Tests:
1. Various (T, M) combinations
2. Two-tier: PQ coarse + PQ residual
3. Two-tier: PQ coarse + WAL v2 residual

## Result
Encode test.

## Artifacts
- `experiments/m67_pq_systematic.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Two-tier PQ systematic test. 8 bits = DEGRADE (3.1137), 12 bits = PASS (2.7824, 2.7819).

**Notes:** Full PPL tested. Worse than WAL v2.
