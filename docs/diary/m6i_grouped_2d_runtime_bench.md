# M6I GROUPED 2D RUNTIME BENCH

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6i_grouped_2d_runtime_bench.

## Configuration
l_max=12, iters=20, threshold=0.0

## Method / What was tested
See `experiments/m6i_grouped_2d_runtime_bench.py` for implementation details.

## Result
Benchmark.

## Artifacts
- `experiments/m6i_grouped_2d_runtime_bench.py`

## Notes from dev_diary_ru.md
```

- После шага 12f остался следующий узкий вопрос: если grouping по `group_cols` уже резко уменьшает launches, даст ли grouping ещё и по строкам дополнительный выигрыш?
- Для этого появился `experiments/m6i_grouped_2d_runtime_bench.py`, который sweeps пары `(group_rows, group_cols)` на policy-approved слоях.

Итог оказался однозначным:
```
