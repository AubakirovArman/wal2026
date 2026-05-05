# M6C ROUTE DISTILL LAYER SUITE

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6c_route_distill_layer_suite.

## Configuration
l_max=12, iters=20, threshold=0.0

## Method / What was tested
See `experiments/m6c_route_distill_layer_suite.py` for implementation details.

## Result
Benchmark.

## Artifacts
- `experiments/m6c_route_distill_layer_suite.py`

## Notes from dev_diary_ru.md
```
- на наших `128 x 128` tile это даёт около `5.03 bpw` против `11 bpw`, то есть reduction по локальному index-traffic чуть больше чем в `2x`.

После этого был сделан новый эксперимент `experiments/m6c_route_distill_layer_suite.py`.
Он прогнал top-1 sampled high-diversity tile уже не только на `q_proj`, а на всех 7 линейных семействах layer 0:

```
