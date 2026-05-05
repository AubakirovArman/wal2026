# M65 WAL V1 TILE PROTOTYPE

## Date
2026 (exact date from git log or experiment run)

## Goal
M65: WAL-1 Tile-Based Vector Prototype.

## Configuration
K_ATOMS=256, KMEANS_ITERS=5, iters=5

## Method / What was tested
Test vector quantization per tile with single atom lookup.
Tile sizes: 8, 16, 32, 64, 128, 256.
Measure relMSE and output relMSE for each.

NOTE: Unloads model after weight extraction to free GPU memory.

## Result
Prototype.

## Artifacts
- `experiments/m65_wal_v1_tile_prototype.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Tile/vector quantization prototype. Single-layer OK but full PPL toxic.

**Notes:** Position-specific scalar quantization at 8 bits. Lesson: single-layer metrics unreliable.
