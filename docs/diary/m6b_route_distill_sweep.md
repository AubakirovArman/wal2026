# M6B ROUTE DISTILL SWEEP

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6b_route_distill_sweep.

## Configuration
l_max=12, iters=20, threshold=0.0

## Method / What was tested
See `experiments/m6b_route_distill_sweep.py` for implementation details.

## Result
Encode test.

## Artifacts
- `experiments/m6b_route_distill_sweep.py`

## Notes from dev_diary_ru.md
```
- После первого pilot стало ясно, что основной дефект сидит в грубой hard-projection стадии.
- В `src/route_distill.py` был добавлен constrained 1D palette refinement после проекции в допустимые route-values.
- После этого появился новый sweep-скрипт `experiments/m6b_route_distill_sweep.py`, который уже смотрит не на один tile, а на несколько sampled high-diversity tile и на несколько palette-size.

Первый важный sweep на top-3 sampled tile слоя `layer0.q_proj` (`128 x 128`):
```
