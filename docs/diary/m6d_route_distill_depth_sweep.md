# M6D ROUTE DISTILL DEPTH SWEEP

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6d_route_distill_depth_sweep.

## Configuration
l_max=12, iters=20, threshold=0.0

## Method / What was tested
See `experiments/m6d_route_distill_depth_sweep.py` for implementation details.

## Result
Benchmark.

## Artifacts
- `experiments/m6d_route_distill_depth_sweep.py`

## Notes from dev_diary_ru.md
```
Это очень важный поворот. Раньше у нас был только storage-аргумент. Теперь уже есть и execution-аргумент, но он пока даёт выигрыш только относительно global route kernel, а не относительно dense.

После этого был добавлен `experiments/m6d_route_distill_depth_sweep.py` и сделан sweep по глубине для `q_proj` и `up_proj` на слоях:

- `0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 79`
```
