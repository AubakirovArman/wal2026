# M43ZO SCALAR TOPK NO LLOYDMAX

## Date
2026 (exact date from git log or experiment run)

## Goal
M43zo: Scalar DRL v2 with lmax=10, K=256 TOP-K only (no Lloyd-Max), skip spiky.

## Configuration
K=256, num_steps=0, threshold=0.0

## Method / What was tested
See `experiments/m43zo_scalar_topk_no_lloydmax.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m43zo_scalar_topk_no_lloydmax.py`
- `experiments/m43zo_scalar_topk_no_lloydmax.log`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.