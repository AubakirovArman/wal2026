# M6P HOTPREFIX FRONTIER BENCH

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6p_hotprefix_frontier_bench.

## Configuration
l_max=12, iters=20, threshold=0.0

## Method / What was tested
See `experiments/m6p_hotprefix_frontier_bench.py` for implementation details.

## Result
Benchmark.

## Artifacts
- `experiments/m6p_hotprefix_frontier_bench.py`

## Notes from dev_diary_ru.md
```
То есть сама идея hot-prefix в принципе выглядит осмысленной, особенно для `q_proj`.

После этого был сделан уже прямой kernel probe: `src/triton_local_palette_hotprefix_matmul.py` и benchmark `experiments/m6p_hotprefix_frontier_bench.py`.

И вот тут пришёл важный отрицательный результат.
```
