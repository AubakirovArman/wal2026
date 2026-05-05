# M6S SHAPE RUNTIME POLICY

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6s_shape_runtime_policy.

## Configuration
See source code for full configuration.

## Method / What was tested
See `experiments/m6s_shape_runtime_policy.py` for implementation details.

## Result
Benchmark.

## Artifacts
- `experiments/m6s_shape_runtime_policy.py`

## Notes from dev_diary_ru.md
```
- Значит, следующий practical milestone теперь не ``ещё одна слепая kernel-гипотеза'', а shape-aware runtime dispatch.

Для этого появился `experiments/m6s_shape_runtime_policy.py`.

Он агрегирует текущие benchmark frontier results (`m6l`, `m6q`, `m6r`) и строит уже deployment-style runtime policy:
```

```
## Шаг 12n. Первый full-model selective-runtime gate: скорость уже измерена, но fused rollout пока неустойчив

- После появления `m6s` следующий логичный шаг был уже не про ещё один per-layer microbench, а про реальный короткий full-model gate с тем самым dispatch.
- Для этого появился `experiments/m6t_selective_runtime_gate.py`.

```
