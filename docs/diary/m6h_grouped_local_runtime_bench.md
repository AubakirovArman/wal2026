# M6H GROUPED LOCAL RUNTIME BENCH

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6h_grouped_local_runtime_bench.

## Configuration
l_max=12, iters=20, threshold=0.0

## Method / What was tested
See `experiments/m6h_grouped_local_runtime_bench.py` for implementation details.

## Result
Benchmark.

## Artifacts
- `experiments/m6h_grouped_local_runtime_bench.py`

## Notes from dev_diary_ru.md
```

- После шага 12e следующая гипотеза была уже очень конкретной: если объединять несколько соседних column-tile в одну более широкую группу, можно ли резко сократить launch overhead без потери точности?
- Для этого появился `experiments/m6h_grouped_local_runtime_bench.py`, а `src/full_layer_tiled_runtime.py` был обобщён до grouped execution по параметру `group_cols`.

Результаты оказались очень сильными.
```
