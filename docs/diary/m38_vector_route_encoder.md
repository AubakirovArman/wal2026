# M38 VECTOR ROUTE ENCODER

## Date
2026 (exact date from git log or experiment run)

## Goal
M38: Vector Route Encoder (VRE).

## Configuration
iters=8

## Method / What was tested
Encodes BxB blocks of weights as residual sums over a shared vector codebook.
Each block's "program" is a sequence of (digit, vector_index) choices.

Key differences from scalar DRL v2:
- Operates on vectors (flattened BxB blocks) instead of scalars
- Shared codebook across all blocks creates natural reuse
- Stop-depth per block creates variable-length programs

## Result
Encode test.

## Artifacts
- `experiments/m38_vector_route_encoder.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

## Extracted Metrics (from source)

- Time: .2
