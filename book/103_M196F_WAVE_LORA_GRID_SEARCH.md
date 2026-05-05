# M196f — Wave-LoRA Grid Search λ (n=20 runs)

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m196f_wave_lora_grid_search.py`

## Purpose

Статистически значимый тест оптимального λ для wave-regularized LoRA на mixed targets.

## Setup

```
Model: meta-llama/Llama-3.1-8B
Layers: 14, 15, 16
Modules: ['o_proj', 'q_proj', 'v_proj', 'gate_proj']
Facts: 50 contrafactual
Rank: 4, Steps: 100, Mixed training
Runs: 20 per λ
Grid: [0.00, 0.01, 0.015, 0.02, 0.025, 0.03]
```

## Results

| λ | Mean | Std | Min | Max |
|---|------|-----|-----|-----|
| **0.0000** | **4.30** | 1.17 | 3 | 7 |
| 0.0100 | 4.05 | 1.10 | 3 | 7 |
| 0.0150 | 4.05 | 0.83 | 3 | 5 |
| 0.0200 | 4.20 | 1.24 | 3 | 7 |
| 0.0250 | 4.05 | 0.89 | 3 | 6 |
| 0.0300 | 4.10 | 0.85 | 3 | 6 |

## Analysis

### Baseline wins
- **λ=0 (baseline) has highest mean: 4.30**
- λ=0.02 is second: 4.20
- λ=0.03 is third: 4.10
- All wave-regularized λ give lower mean survival than baseline

### Differences are NOT significant
- Max difference: 4.30 - 4.05 = **0.25**
- Min std: 0.83
- All means within 1 std of each other

### Wave regularization does NOT help
At n=20, wave regularization provides **no statistically significant improvement** in survival.

### M196e was underpowered
M196e (n=5) showed λ=0.025 as best (4.80 vs 3.40 baseline). At n=20, this effect disappeared:
- M196e baseline: 3.40 (unlucky run)
- M196f baseline: 4.30 (more representative)

## Conclusion

> **Wave regularization is NOT beneficial for factual editing on mixed targets with rank=4.**
>
> Production recommendation: **λ = 0** (no wave regularization)
>
> Rationale:
> 1. Baseline gives highest mean survival
> 2. No statistical evidence that any λ > 0 helps
> 3. Wave reg adds computational overhead without benefit

## Updated Production Stack

```python
# Base: Hadamard-WAL K=256
# Edit: LoRA rank=4, λ=0 (baseline)
# Target: layers 14-16, 4 modules
# Training: mixed wikitext-2 + facts
```

## Lesson Learned

**n=5 is insufficient for reliable conclusions.** M196e suggested λ=0.025 was best, but n=20 shows baseline is actually better. Always need n≥20 for statistical confidence in survival experiments.

## Next Steps

- **M196g**: Test if wave reg helps with higher rank (rank=8) or more steps (200)
- **M196h**: Test if wave reg helps with single-module targets (o_proj only, as in M196)
- **M196i**: Ablation — which hyperparameter matters most: rank, steps, or lr?

## Code Reference

```python
lambdas = [0.0, 0.01, 0.015, 0.02, 0.025, 0.03]
n_runs = 20
for run in range(n_runs):
    for wave_lambda in lambdas:
        model = train_mixed(..., wave_lambda=wave_lambda)
        surv = eval_survival(model, ...)
```

## Related

- M196 — o_proj only (wave reg helped at n=1)
- M196e — n=5 variance test (λ=0.025 looked best)
- M196f — n=20 grid search (baseline wins)
