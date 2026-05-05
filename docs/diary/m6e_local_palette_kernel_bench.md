# M6E LOCAL PALETTE KERNEL BENCH

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6e_local_palette_kernel_bench.

## Configuration
l_max=12, iters=20, threshold=0.0

## Method / What was tested
See `experiments/m6e_local_palette_kernel_bench.py` for implementation details.

## Result
Benchmark.

## Artifacts
- `experiments/m6e_local_palette_kernel_bench.py`

## Notes from dev_diary_ru.md
```
- Это уже не просто indexing-proxy, а первый реальный Triton matmul, который работает с local palette indices.

Для проверки появился `experiments/m6e_local_palette_kernel_bench.py`.
Он был прогнан на layer-0 `q_proj`, `up_proj`, `o_proj`, `down_proj` для palette `16/32/64`.

```
