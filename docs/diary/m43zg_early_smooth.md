# M43ZG EARLY SMOOTH

## Date
2026 (exact date from git log or experiment run)

## Goal
M43zg: Encode only smooth params in early layers (0-19), skip spiky q/k/v.

## Configuration
K=128, num_steps=0, threshold=0.0

## Method / What was tested
See `experiments/m43zg_early_smooth.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m43zg_early_smooth.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.