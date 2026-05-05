# M43ZF EARLY OD

## Date
2026 (exact date from git log or experiment run)

## Goal
M43zf: Encode only o_proj and down_proj in early layers (0-19).

## Configuration
K=128, num_steps=0, threshold=0.0

## Method / What was tested
See `experiments/m43zf_early_od.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m43zf_early_od.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.