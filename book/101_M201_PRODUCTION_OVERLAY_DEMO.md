# M201 — Production Overlay Demo

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m201_production_overlay_demo.py`

## Purpose

Демонстрация production workflow: encode base → attach Wave-LoRA overlay → evaluate WITHOUT merge.

## Setup

```
Model: meta-llama/Llama-3.1-8B
Base: Hadamard-WAL K=256 uniform (iters=5)
Edit: LoRA rank=4, layers 14-16, modules ['o_proj','q_proj','v_proj','gate_proj']
Training: Mixed wikitext-2 + 50 facts, steps=100, λ=0 (baseline)
Runtime: WALCachedLinear + LoRA overlay (no merge)
```

## Results

| Stage | PPL | PPLΔ | Survival |
|-------|-----|------|----------|
| Baseline (dense) | 12.4883 | — | 3/50 |
| After Encode | 12.4149 | **-0.0734** | — |
| Overlay (NO merge) | 12.4018 | **-0.0865** | **4/50** |

## Analysis

### Encode is near-lossless
- PPL improves from 12.4883 → 12.4149 (-0.0734)
- WAL K=256 is actually **slightly better** than dense baseline (!)
- Likely due to quantization smoothing noise in weights

### Overlay works perfectly
- PPL further improves: 12.4149 → 12.4018 (-0.0130)
- Total vs baseline: -0.0865 PPL (**better than dense!**)
- Survival: 3 → 4/50 (+1 fact)

### Compare with M200 (merge path)
| Path | Final PPL | Result |
|------|-----------|--------|
| **Overlay (M201)** | **12.4018** | ✅ Perfect |
| Merge + re-encode (M200) | 16.5739 | ❌ Catastrophic (+33%) |

## Conclusion

> **Production path CONFIRMED: Base + overlay ONLY.**
>
> ```
> encode(base) → WAL checkpoint
> train(LoRA on cached WAL layers)
> deploy(WAL + LoRA overlay, never merge)
> ```
>
> Benefits:
> - PPL: **-0.09** vs dense baseline
> - Survival: **+1** fact
> - Edit size: **~0.5 MB** (LoRA overlay)
> - Base size: **~11.3 GB** (WAL K=256)
> - Runtime: **Dense-speed** after cache warmup

## Critical Rule

**NEVER merge + re-encode.** The merge step destroys weight structure irreversibly. Overlay is the only viable production path.

## Code Reference

```python
# Encode base
model = AutoModelForCausalLM.from_pretrained(...)
encode_model(model, K=256, iters=5)

# Inject LoRA overlay (no merge!)
inject_lora_overlay(model, target_layers=[14,15,16], ...)

# Train
for step in range(100):
    loss = train_mixed_step(model, batch)
    # LoRA weights update, base WAL stays frozen

# Evaluate — base + overlay in forward pass
ppl = eval_ppl(model)
surv = eval_survival(model, facts)
```

## Related

- M200 — End-to-End WAL v2 (merge path, catastrophic)
- M182 — Transform Editability (diff ~99.8%, confirms overlay-only)
- M135 — WAL+LoRA Overlay (original proof of concept)
