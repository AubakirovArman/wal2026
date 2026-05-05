# M153 — Real Transform-WAL Encoder

**Date:** 2026-04-20
**Status:** ✅ Complete (v2 fast version)
**Goal:** Full pipeline: W → Transform → WAL encode → WAL decode → inverse Transform → W_recon.

## Method

- Model on CPU (`device_map="cpu"`)
- 2 layers (0, 16), 2 modules (q_proj, v_proj)
- K=64, C=8, iters=1
- Transforms: Raw, RandOrth, Hadamard

## Results

### layers.0.self_attn.q_proj

| Transform | MSE | relMSE | Time |
|-----------|-----|--------|------|
| Raw | 2.16e-07 | 0.158 | 10.7s |
| RandOrth | 2.93e-08 | 0.891 | 12.5s |
| Hadamard | 1.08e-07 | 1.627 | 10.4s |

### layers.0.self_attn.v_proj

| Transform | MSE | relMSE | Time |
|-----------|-----|--------|------|
| Raw | 4.97e-09 | 0.085 | 2.3s |
| RandOrth | 2.17e-09 | 0.112 | 3.1s |
| Hadamard | 2.25e-09 | 0.111 | 2.4s |

### layers.16.self_attn.q_proj

| Transform | MSE | relMSE | Time |
|-----------|-----|--------|------|
| Raw | 3.02e-08 | 0.047 | 9.0s |
| RandOrth | 1.72e-08 | 0.080 | 11.9s |
| Hadamard | 1.96e-08 | 0.084 | 9.5s |

### layers.16.self_attn.v_proj

| Transform | MSE | relMSE | Time |
|-----------|-----|--------|------|
| Raw | 4.79e-08 | 0.059 | 2.4s |
| RandOrth | 5.50e-09 | 0.077 | 3.1s |
| Hadamard | 4.48e-09 | 0.067 | 2.8s |

## Key Findings

1. **RandOrth wins on v_proj**: 2.3–8.7× better MSE than Raw
2. **Hadamard competitive**: Similar to RandOrth on v_proj, slightly worse on q_proj
3. **Raw wins on q_proj layer 0**: But this may be due to K=64 being too small for transforms
4. **Transform benefit varies by layer type**: v_proj benefits more than q_proj

## Limitations

- K=64 is too small for production quality
- Only 2 layers tested
- No PPL validation (CPU-only)
- relMSE high for all transforms (K=64 limitation)

## Artifacts

- `experiments/m153_transform_wal_encoder_v2.py`
- `experiments/m153_transform_wal_encoder.json`
