# M9C ACT SPARSITY PROBE

## Date
2026 (exact date from git log or experiment run)

## Goal
M9c: Activation sparsity probe for `down_proj` inputs (post-SwiGLU).

## Configuration
See source code for full configuration.

## Method / What was tested
Question: how much of `x` going into down_proj is near-zero?
If a meaningful fraction of K-tiles have max|x| below threshold, we can
skip decode + GEMM for those tiles -> direct speedup with zero quality loss
(at conservative thresholds).

Method:
  1. Load Llama-3.3-70B with EagerBf16Linear (baseline quality).
  2. Hook all `mlp.down_proj` modules to capture incoming `x` (the SiLU(gate)*up tensor).
  3. Run 4 WikiText-2 windows (2048 tokens each).
  4. For each captured x of shape [tokens, K], compute:
     - per-token max|x| (row-wise max).
     - tile-wise max|x| with TILE in {32, 64, 128} along K dim.
     - fraction of TILES below absolute threshold 1e-4, 1e-3, 1e-2.
     - fraction of TILES below RELATIVE threshold (vs row max).
  5. Report mean and per-layer-bucket (early/mid/late) statistics.

## Result
Runtime test.

## Artifacts
- `experiments/m9c_act_sparsity_probe.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.