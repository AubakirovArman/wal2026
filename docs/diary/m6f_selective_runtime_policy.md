# M6F SELECTIVE RUNTIME POLICY

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6f_selective_runtime_policy.

## Configuration
See source code for full configuration.

## Method / What was tested
See `experiments/m6f_selective_runtime_policy.py` for implementation details.

## Result
Runtime test.

## Artifacts
- `experiments/m6f_selective_runtime_policy.py`

## Notes from dev_diary_ru.md
```

- После depth-sweep стало уже недостаточно просто говорить: "идея иногда работает".
- Поэтому появился `experiments/m6f_selective_runtime_policy.py`, который превращает результаты sweep в явное deployment-rule.

Текущий rule-set очень консервативный:
```
