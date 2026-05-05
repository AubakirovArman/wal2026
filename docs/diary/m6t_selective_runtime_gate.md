# M6T SELECTIVE RUNTIME GATE

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6t_selective_runtime_gate.

## Configuration
See source code for full configuration.

## Method / What was tested
See `experiments/m6t_selective_runtime_gate.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m6t_selective_runtime_gate.py`

## Notes from dev_diary_ru.md
```

- После появления `m6s` следующий логичный шаг был уже не про ещё один per-layer microbench, а про реальный короткий full-model gate с тем самым dispatch.
- Для этого появился `experiments/m6t_selective_runtime_gate.py`.

Что он делает:
```


## Extracted Metrics (from source)

- PPL: .4
- PPL: .4
