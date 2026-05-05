# M59 WAL GLOBAL CODEBOOK FAST

## Date
2026 (exact date from git log or experiment run)

## Goal
M59: Fast global codebook analysis using per-layer atoms (no global k-means bottleneck).

## Configuration
K_ATOMS=128, KMEANS_ITERS=5

## Method / What was tested
Encode is identical to M57 (per-layer atoms, PPL 2.7828 proven).
Only adds global codebook mining and compression analysis.

## Result
PPL evaluation.

## Artifacts
- `experiments/m59_wal_global_codebook_fast.py`
- `experiments/m59_wal_global_codebook_fast.log`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

## Extracted Metrics (from source)

- PPL: 2.7828
- PPL: 2.7828
