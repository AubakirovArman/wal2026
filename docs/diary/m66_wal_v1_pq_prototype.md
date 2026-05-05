# M66 WAL V1 PQ PROTOTYPE

## Date
2026 (exact date from git log or experiment run)

## Goal
M66: WAL-1 Product Quantization Tile Prototype.

## Configuration
K_ATOMS=256, KMEANS_ITERS=5, iters=5

## Method / What was tested
Test Product Quantization per tile:
- Split tile into M subvectors
- K-means on each subvector independently  
- Store M atom_ids per tile
- Measure relMSE and output relMSE.

Then test with residual WAL v2 encoding.

## Result
Encode test.

## Artifacts
- `experiments/m66_wal_v1_pq_prototype.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Product Quantization prototype.

**Notes:** Two-tier position-specific quantization tested in M73.
