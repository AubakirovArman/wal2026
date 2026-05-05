# M69 PQ VARYING K

## Date
2026 (exact date from git log or experiment run)

## Goal
M69: Position-specific scalar quantization with varying K.

## Configuration
K=16, KMEANS_ITERS=5, iters=5

## Method / What was tested
For M=T (each weight has its own codebook), test different K values:
K=16 (4 bits/weight), 32 (5 bits), 64 (6 bits), 128 (7 bits), 256 (8 bits).

Also test: T=8,16,32,64,128 with M=T to see if tile size matters.

## Result
Unknown.

## Artifacts
- `experiments/m69_pq_varying_k.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Position-specific sweep K=16,32,64,128,256. K=16→111k FAIL, K=32→1.5k FAIL, K=64→46.6 FAIL, K=128→7.68 FAIL, K=256→3.02 DEGRADE.

**Notes:** Lesson: <12 bits/weight = catastrophic accumulation across 80 layers.
