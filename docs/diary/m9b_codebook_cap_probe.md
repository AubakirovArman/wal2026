# M9B CODEBOOK CAP PROBE

## Date
2026 (exact date from git log or experiment run)

## Goal
M9b: Codebook compression probe.

## Configuration
See source code for full configuration.

## Method / What was tested
Question: can we cap the route codebook to M' << ~1500 with negligible quality loss?
If M' <= 256 fits in int8 -> direct -50% VRAM on ids (the dominant runtime cost).

Method per layer:
  1. Encode normally -> ids[N,K] int32, codebook_sum[M] fp16.
  2. Compute route frequency over all weight positions.
  3. For each target M' in {64, 128, 256, 512, 1024}:
       - Keep top-M' most-frequent routes.
       - For each pruned route, remap its positions to nearest kept route in
         scalar codebook_sum value (scalar L1 distance, since the route is
         ultimately a single scalar after pre-summing).
       - Reconstruct weight with the smaller codebook -> compute rel_mse vs
         original full reconstruction AND vs original bf16 weight.

## Result
PPL evaluation.
Likely negative result

## Artifacts
- `experiments/m9b_codebook_cap_probe.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.