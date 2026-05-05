# M43ZL SCALAR SKIP EARLY SPIKY

## Date
2026 (exact date from git log or experiment run)

## Goal
M43zl: Scalar all layers, skip q/k/v/gate/up in layers 0-3, encode rest.

## Configuration
num_steps=0, threshold=0.0

## Method / What was tested
See `experiments/m43zl_scalar_skip_early_spiky.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m43zl_scalar_skip_early_spiky.py`
- `experiments/m43zl_scalar_skip_early_spiky.log`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.