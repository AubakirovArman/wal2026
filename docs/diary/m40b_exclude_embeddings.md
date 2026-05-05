# M40B EXCLUDE EMBEDDINGS

## Date
2026 (exact date from git log or experiment run)

## Goal
M40b: Test hybrid encoder excluding embedding and lm_head.

## Configuration
K=16, batch=512, iters=12, block_size=4, threshold=0.0

## Method / What was tested
See `experiments/m40b_exclude_embeddings.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m40b_exclude_embeddings.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

## Extracted Metrics (from source)

- PPL: .4
- PPL: .4
- PPL: .4
- PPL: .4
