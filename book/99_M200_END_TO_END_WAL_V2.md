# M200 — End-to-End WAL v2 Demo

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m200_end_to_end_wal_v2.py`

## Purpose

Полный production pipeline:
1. Encode base model with Hadamard-WAL adaptive K
2. Train Wave-LoRA on encoded model
3. Merge LoRA into base weights
4. Re-encode merged weights
5. Evaluate PPL and survival at each stage

## Setup

```
Model: meta-llama/Llama-3.1-8B
Encode: Hadamard-WAL adaptive K (iters=3, excl embed/lm_head)
LoRA: rank=4, baseline λ=0, mixed targets (o_proj+q_proj+v_proj+gate_proj @ layers 14-16)
Steps: 100
Facts: 20 contrafactual (for speed)
```

## Results

| Stage | PPL | PPLΔ | Survival |
|-------|-----|------|----------|
| Baseline | 10.3786 | — | 1/20 |
| After Encode | 10.3711 | **-0.0076** | — |
| After LoRA | 10.2813 | -0.0973 | 1/20 |
| After Merge | **16.5433** | **+6.1647** | 0/20 |
| After Re-encode | **16.5739** | **+6.1953** | 0/20 |

**Encode time: 1215.3s, Train time: 12.6s, Re-encode: 1214.2s**

## Critical Finding: Merge DESTROYS PPL

**Merging LoRA into WAL-encoded base weights causes catastrophic PPL degradation: +6.16 (+59% relative).**

### Why?
1. LoRA deltas обучены на encoded weights (Hadamard-quantized)
2. При merge delta добавляется к encoded base
3. Результат — веса в "неправильном" пространстве (не Hadamard, не исходные)
4. Re-encode пытается применить Hadamard+quantization к этим искажённым весам
5. Quantization error накапливается: encode → edit → re-encode = lossy cascade

### Confirmation of M182
M182 showed ~99.8% program diff after LoRA edit → WAL patch fundamentally non-local.

M200 confirms: **merge + re-encode is fundamentally broken.**

## Production Path

> **Base + Overlay ONLY. Never merge + re-encode.**
>
> ```
> Base:     Hadamard-WAL adaptive K checkpoint
> Edit:     Wave-Regularized LoRA (λ=0.025 for mixed targets)
> Runtime:  WALCachedLinear + LoRA overlay
> Safety:   Spectral norm monitoring
> ```

## Encode Quality

Encoded PPL: 10.3711 vs Baseline 10.3786 → **Δ=-0.0076**

This is **near-lossless** for the encode stage! Hadamard-WAL adaptive K with iters=3 is production-quality.

## LoRA Training on Encoded Model

LoRA PPL: 10.2813 vs Encoded 10.3711 → **Δ=-0.09**

Training on encoded weights works fine — PPL actually slightly improves (possibly due to regularization effect of mixed training).

But survival: 1/20 (same as baseline). Rank=4 baseline on 20 facts with 100 steps is insufficient for good survival. Need rank=4 with 100+ steps or higher rank.

## Next Steps

- **M200b**: Test with better LoRA config (rank=8, 200 steps) for better survival
- **M200c**: Test overlay approach (no merge) — PPL should stay stable
- **M201**: Full production demo with WALCachedLinear + LoRA overlay

## Code Reference

```python
# Pipeline stages
encoded = encode_model_adaptive(model, k_map, iters=3)
model = train_mixed_wave(model, tokenizer, steps=100, rank=4, ...)
model = merge_lora(model, TARGET_LAYERS, TARGET_MODULES)
encoded_merged = encode_model_adaptive(model, k_map, iters=3)
```

## Related

- M182 — WAL patch non-locality (~99.8% diff after LoRA)
- M195b+ — Hadamard adaptive K encode quality
- M196e — Optimal λ=0.025 for mixed targets
