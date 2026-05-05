# M8A FP8 MICROBENCH

## Date
2026 (exact date from git log or experiment run)

## Goal
M8a: FP8 vs BF16 microbench + correctness check on real Llama-70B weights.

## Configuration
iters=20

## Method / What was tested
Validates whether torch._scaled_mm fp8 rowwise path matches bf16 F.linear
quality on our route-decoded weights, and measures the speed / VRAM ratio
on H200.

## Result
Benchmark.

## Artifacts
- `experiments/m8a_fp8_microbench.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.