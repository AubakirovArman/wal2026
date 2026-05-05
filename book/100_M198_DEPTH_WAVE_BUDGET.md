# M198 — Depth-Wave Budget: Uniform vs Adaptive K

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m198_depth_wave_budget.py`

## Purpose

Сравнение трёх стратегий назначения K (размер codebook):
1. **Uniform K=256** — фиксированный размер для всех модулей
2. **Risk Adaptive** — K по формуле риска (spectral_norm × top10_energy)
3. **Depth Adaptive** — K по глубине слоя (early=128, mid=256, late=512)

## Setup

```
Model: meta-llama/Llama-3.1-8B
Base: Hadamard-WAL with k-means quant (iters=5)
Excluded: embed_tokens, lm_head (128256×4096 too slow)
Total modules encoded: 226
```

## Results

| Strategy | PPL | ΔPPL | vs Uniform |
|----------|-----|------|------------|
| **Uniform K=256** | **12.4069** | **-0.0814** | — |
| Risk Adaptive | 12.4927 | +0.0044 | +0.0857 |
| Depth Adaptive | 12.4296 | -0.0587 | +0.0227 |

## Analysis

### Uniform wins on PPL
- Uniform K=256: лучший PPL (12.4069)
- Depth adaptive: близко, но хуже (12.4296, Δ +0.0227)
- Risk adaptive: хуже baseline (12.4927)

### Speed
- Uniform: самый быстрый (нет overhead на вычисление риска)
- Depth adaptive: быстрый (только индекс слоя)
- Risk adaptive: медленнее (нужен spectral analysis)

### Lesson
**Adaptive K doesn't improve PPL on Llama-3.1-8B.** Uniform K=256 — simplest and best. Adaptive methods add complexity without quality benefit.

## Conclusion

> Production: **Uniform K=256 for all modules.**
>
> Rationale:
> 1. Best PPL (lowest)
> 2. Fastest encoding
> 3. Simplest implementation
> 4. No per-module tuning needed

## Code Reference

```python
def assign_depth_k(layer_idx, n_layers=32):
    if layer_idx < n_layers // 3:
        return 128
    elif layer_idx < 2 * n_layers // 3:
        return 256
    else:
        return 512
```

## Related

- M195 — Adaptive K on scalar WAL
- M195b+ — Adaptive K on Hadamard-WAL with k-means
- M190 — Wave-guided budget (failed, +39% PPL)
