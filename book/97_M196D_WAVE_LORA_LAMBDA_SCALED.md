# M196d — Wave-LoRA with Module-Count-Scaled λ

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m196d_wave_lora_lambda_scaled.py`

## Purpose

Проверить гипотезу: λ=0.1 hurts на mixed targets, потому что penalty масштабируется с числом модулей. Тестируем λ=0.025 (0.1/4), λ=0.05 (0.1/2), и λ=0.1 для сравнения.

## Setup

```
Model: meta-llama/Llama-3.1-8B
Layers: 14, 15, 16
Modules: ['o_proj', 'q_proj', 'v_proj', 'gate_proj'] (12 total)
Facts: 50 contrafactual
Rank: 4, Steps: 100, Mixed training
```

## Results

| Config | Surv | PPLΔ | λ | Train Time |
|--------|------|------|---|------------|
| rank4_baseline | 4/50 | -0.02 | 0.0000 | 13.8s |
| rank4_wave0025 | 6/50 | +0.13 | 0.0250 | 13.5s |
| rank4_wave0050 | 4/50 | -0.03 | 0.0500 | 13.7s |
| rank4_wave0100 | 6/50 | +0.09 | 0.1000 | 13.7s |

## Analysis

### Variance dominates
- Baseline: 4/50 (в M196b был 10/50!)
- λ=0.025: 6/50 (+2)
- λ=0.050: 4/50 (0)
- λ=0.100: 6/50 (+2)

Разница между λ=0.025 и λ=0.1 — **0** (оба 6/50). Масштабирование λ не дало improvement.

### Comparison with M196b
В M196b baseline дал 10/50, а λ=0.1 дал 5/50 (ухудшение). Здесь baseline 4/50, λ=0.1 дал 6/50 (улучшение). **Variance между запусками выше эффекта wave reg.**

### Conclusion
**Wave regularization effect on mixed targets is below noise floor.** Нужно:
1. Больше runs (n=10) для каждой конфигурации
2. Или другой penalty (не top10_energy)
3. Или wave reg не подходит для factual editing на больших target sets

## Lesson Learned

> **Factual editing survival has high variance.** Single-run comparisons are unreliable. Need n≥5 runs per config for statistical significance.

## Next Steps

- **M196e**: n=5 runs per λ для оценки variance
- **M196f**: Test different penalty (spectral norm instead of top10_energy)
- **M197**: Ablation — какой module type даёт больше всего survival?

## Code Reference

```python
# λ normalized by n_modules
wave_pen = wave_pen / n_modules  # average per module
loss = loss + wave_lambda * wave_pen
```

## Related

- M196 — o_proj only (wave reg helped)
- M196b — mixed targets (wave reg hurt in single run)
- M196c — penalty schedules
