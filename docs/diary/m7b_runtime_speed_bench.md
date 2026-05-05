# M7B RUNTIME SPEED BENCH

## Date
2026 (exact date from git log or experiment run)

## Goal
Runtime microbench for ID-route and Block-RVQ layer variants.

## Configuration
See source code for full configuration.

## Method / What was tested
Supports:
  * id_route        - existing packed/fused/cached route benchmark
  * block_rvq       - single-layer Block-RVQ packed benchmark
  * block_rvq_bundle - q/k/v/o upper-bound bench for shared decode ideas

## Result
Benchmark.

## Artifacts
- `experiments/m7b_runtime_speed_bench.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.