# M6J GROUPED SHAPE FRONTIER BENCH

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6j_grouped_shape_frontier_bench.

## Configuration
l_max=12, iters=20, threshold=0.0

## Method / What was tested
See `experiments/m6j_grouped_shape_frontier_bench.py` for implementation details.

## Result
Benchmark.

## Artifacts
- `experiments/m6j_grouped_shape_frontier_bench.py`

## Notes from dev_diary_ru.md
```

- После шага 12g следующая проверяемая гипотеза была очень узкой: а что если `group_rows=512` всё ещё не потолок, и можно ещё снизить latency, не убив local path слишком большим union на группу?
- Для этого появился `experiments/m6j_grouped_shape_frontier_bench.py` со sweep по `group_rows=256/512/1024/2048` и `group_cols=4096/8192`.

Результат оказался сильнее ожиданий:
```
