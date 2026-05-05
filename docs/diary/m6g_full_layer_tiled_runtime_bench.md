# M6G FULL LAYER TILED RUNTIME BENCH

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6g_full_layer_tiled_runtime_bench.

## Configuration
l_max=12, iters=20, threshold=0.0

## Method / What was tested
See `experiments/m6g_full_layer_tiled_runtime_bench.py` for implementation details.

## Result
Benchmark.

## Artifacts
- `experiments/m6g_full_layer_tiled_runtime_bench.py`

## Notes from dev_diary_ru.md
```
Это хороший результат в инженерном смысле: теперь у нас есть не просто идея selective deployment, а конкретный reproducible rule.

После этого был сделан `experiments/m6g_full_layer_tiled_runtime_bench.py`.
Здесь уже benchmark шёл не на одном tile, а на всём слое для policy-approved кандидатов.

```
