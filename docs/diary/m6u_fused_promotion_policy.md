# M6U FUSED PROMOTION POLICY

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6u_fused_promotion_policy.

## Configuration
See source code for full configuration.

## Method / What was tested
See `experiments/m6u_fused_promotion_policy.py` for implementation details.

## Result
Runtime test.

## Artifacts
- `experiments/m6u_fused_promotion_policy.py`

## Notes from dev_diary_ru.md
```
- ещё `1` слой уходит в non-finite fallback.

Это превратилось в явный fused promotion policy через `experiments/m6u_fused_promotion_policy.py`.

Дальше был проверен уже настоящий hybrid runtime:
```
