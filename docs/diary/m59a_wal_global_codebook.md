# M59A WAL GLOBAL CODEBOOK

## Date
2026 (exact date from git log or experiment run)

## Goal
M59a: WAL-0 global codebook across all layers.

## Configuration
K_ATOMS=128, KMEANS_ITERS=5

## Method / What was tested
Encode all parameters, collect all programs into a single global codebook.
Measure: global vocabulary size, entropy, coverage.
Compare to per-layer codebooks (sum of per-layer unique programs).

## Result
Encode test.

## Artifacts
- `experiments/m59a_wal_global_codebook.py`
- `experiments/m59a_wal_global_codebook.log`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.