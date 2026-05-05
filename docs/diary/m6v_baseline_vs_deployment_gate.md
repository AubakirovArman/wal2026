# M6V BASELINE VS DEPLOYMENT GATE

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6v_baseline_vs_deployment_gate.

## Configuration
See source code for full configuration.

## Method / What was tested
See `experiments/m6v_baseline_vs_deployment_gate.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m6v_baseline_vs_deployment_gate.py`

## Notes from dev_diary_ru.md
```

- После шага 12p уже нельзя было продолжать сводить картину вручную из разных gate-скриптов. Нужен был один compare-run, который всегда показывает, что у исходной модели и что у нашего текущего deployable runtime по одним и тем же окнам текста.
- Для этого появился `experiments/m6v_baseline_vs_deployment_gate.py`.

Он делает ровно то, что нужно видеть постоянно:
```


## Extracted Metrics (from source)

- PPL: .4
- PPL: .4
