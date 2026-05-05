# M7A FUSED DIAG

## Date
2026 (exact date from git log or experiment run)

## Goal
Root-cause diagnostic for FusedIDRouteLinear NaN.

## Configuration
See source code for full configuration.

## Method / What was tested
Quantize a single Llama linear, compare fused vs reference on synthetic input,
and dump where they diverge in magnitude/finiteness.

## Result
Runtime test.

## Artifacts
- `experiments/m7a_fused_diag.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.