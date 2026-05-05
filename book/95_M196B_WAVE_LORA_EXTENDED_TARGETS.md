# M196b — Wave-Regularized LoRA: Extended Targets (50 Facts, Mixed Modules)

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m196b_wave_lora_extended.py`

## Purpose

Проверить масштабируемость Wave-Regularized LoRA:
- Больше фактов: 50 (вместо 10)
- Больше target модулей: o_proj + q_proj + v_proj + gate_proj (слои 14,15,16)
- Разные rank: 1, 2, 4
- Сравнение baseline (λ=0) vs wave-regularized (λ=0.1)

## Setup

```
Model: meta-llama/Llama-3.1-8B
Layers: 14, 15, 16
Modules: ['o_proj', 'q_proj', 'v_proj', 'gate_proj']
Facts: 50 contrafactual
Steps: 100
LR: 5e-5
Mixed training: 50/50 wikitext-2 + facts
```

## Results

### o_proj only (reference — M196)
| Config | Surv (10 facts) | PPLΔ |
|--------|----------------|------|
| rank1 baseline | 0/10 | +0.01 |
| rank1 wave λ=0.1 | 2/10 | +0.03 |

### Mixed modules (M196b — 50 facts)
| Config | Surv | PPLΔ | Train Time |
|--------|------|------|------------|
| rank1 baseline | 6/50 | +0.09 | 20.6s |
| rank1 wave λ=0.1 | 3/50 | +0.22 | 23.0s |
| rank2 baseline | 6/50 | +0.53 | 18.8s |
| rank2 wave λ=0.1 | 4/50 | +0.73 | 20.0s |
| rank4 baseline | **10/50** | +1.07 | 16.5s |
| rank4 wave λ=0.1 | 5/50 | +1.89 | 23.4s |

## Critical Finding: Wave Reg **HURTS** on Mixed Targets

| Metric | Baseline | Wave λ=0.1 | Δ |
|--------|----------|------------|---|
| rank1 survival | 6/50 | 3/50 | **-3** |
| rank2 survival | 6/50 | 4/50 | **-2** |
| rank4 survival | 10/50 | 5/50 | **-5** |
| rank1 PPL | +0.09 | +0.22 | +0.13 |
| rank4 PPL | +1.07 | +1.89 | +0.82 |

**Wave regularization λ=0.1 ухудшает survival на mixed targets.**

Это противоположно результатам M196 (o_proj only), где wave reg улучшал survival 0→2/10.

## Hypotheses for Failure

1. **Too many constrained parameters**: 4 modules × 3 layers = 12 LoRA pairs. Wave penalty применяется ко всем одновременно. Конфликтующие градиенты — penalty тянет в сторону низкой energy, factual loss — в сторону правильного ответа. На 12 модулях конфликт сильнее.

2. **λ=0.1 too high for mixed**: На o_proj only (2 модуля) λ=0.1 работал. На 12 модулях суммарный penalty в 6× больше. Нужен λ пропорциональный числу модулей (λ/mods).

3. **Baseline already good enough**: rank4 baseline даёт 10/50 survival — лучший результат во всех LoRA-экспериментах. Wave reg мешает оптимизации factual loss, добавляя нерелевантное ограничение.

## Lesson Learned

> **Wave regularization is module-count sensitive.**
> 
> λ must be calibrated per number of target modules, not fixed globally.
> 
> Formula: `λ_eff = λ_base / sqrt(n_modules)` or `λ_eff = λ_base / n_modules`

## Next Steps

- **M196d**: Test λ=0.025 (0.1/4) для mixed targets
- **M196e**: Test per-module λ scheduling (больше λ на чувствительные модули)
- **M196f**: Ablation — какой модуль даёт больше всего survival?

## Code Reference

```python
# m196b_wave_lora_extended.py
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ['o_proj', 'q_proj', 'v_proj', 'gate_proj']
configs = [
    ("rank1_baseline", 1, 100, 0.0),
    ("rank1_wave010", 1, 100, 0.1),
    ("rank2_baseline", 2, 100, 0.0),
    ("rank2_wave010", 2, 100, 0.1),
    ("rank4_baseline", 4, 100, 0.0),
    ("rank4_wave010", 4, 100, 0.1),
]
```

## Related

- M196 — o_proj only (wave reg helped)
- M196c — penalty schedules (constant λ most reliable)
- M188 — module sensitivity ranking
