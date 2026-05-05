# M6 ROUTE DISTILL PILOT

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6_route_distill_pilot.

## Configuration
l_max=12, iters=20, threshold=0.0

## Method / What was tested
See `experiments/m6_route_distill_pilot.py` for implementation details.

## Result
Encode test.

## Artifacts
- `experiments/m6_route_distill_pilot.py`

## Notes from dev_diary_ru.md
```

- Был добавлен новый helper `src/route_distill.py`.
- Был добавлен pilot-скрипт `experiments/m6_route_distill_pilot.py`.
- Это не full-model distillation, а минимальный tile-local prototype: он берёт один реальный tile из уже закодированного слоя, пытается сжать число локальных route через маленький palette и подгоняет student под teacher по output-MSE на sampled activations.

```

```
- После первого pilot стало ясно, что основной дефект сидит в грубой hard-projection стадии.
- В `src/route_distill.py` был добавлен constrained 1D palette refinement после проекции в допустимые route-values.
- После этого появился новый sweep-скрипт `experiments/m6b_route_distill_sweep.py`, который уже смотрит не на один tile, а на несколько sampled high-diversity tile и на несколько palette-size.

Первый важный sweep на top-3 sampled tile слоя `layer0.q_proj` (`128 x 128`):
```

```
- на наших `128 x 128` tile это даёт около `5.03 bpw` против `11 bpw`, то есть reduction по локальному index-traffic чуть больше чем в `2x`.

После этого был сделан новый эксперимент `experiments/m6c_route_distill_layer_suite.py`.
Он прогнал top-1 sampled high-diversity tile уже не только на `q_proj`, а на всех 7 линейных семействах layer 0:

```
