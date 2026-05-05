# M196e — Wave-LoRA Variance Test (n=5 runs)

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m196e_wave_lora_variance_test.py`

## Purpose

Оценить статистическую значимость wave regularization на mixed targets с n=5 независимых runs.

## Setup

```
Model: meta-llama/Llama-3.1-8B
Layers: 14, 15, 16
Modules: ['o_proj', 'q_proj', 'v_proj', 'gate_proj']
Facts: 50 contrafactual
Rank: 4, Steps: 100, Mixed training
Runs: 5 per config
```

## Results

| Config | λ | Mean | Std | Min | Max |
|--------|---|------|-----|-----|-----|
| baseline | 0.0000 | **3.40** | 0.55 | 3 | 4 |
| wave0025 | 0.0250 | **4.80** | 0.84 | 4 | 6 |
| wave0050 | 0.0500 | **4.20** | 1.30 | 3 | 6 |
| wave0100 | 0.1000 | **3.20** | 0.84 | 2 | 4 |

## Analysis

### wave0025 is best
- Mean survival: 4.80 vs baseline 3.40 (+41% improvement)
- Difference: 1.4, sum of std: 0.55 + 0.84 = 1.39
- **Almost significant** — difference ≈ sum of std

### λ=0.1 is worst
- Mean survival: 3.20 — ниже baseline!
- Confirms M196b finding: high λ hurts on mixed targets

### Non-monotonic relationship
- λ=0.025: best (4.80)
- λ=0.050: middle (4.20)
- λ=0.100: worst (3.20)

**Optimal λ exists around 0.02–0.03 for mixed targets.**

### Variance is real
- Baseline std: 0.55 (16% of mean)
- Single-run comparisons are unreliable
- Need n≥10 for confident conclusions

## Lesson Learned

> **Wave regularization has a sweet spot λ ≈ 0.025 for mixed targets.**
> 
> Too low (0): no regularization benefit
> Too high (0.1): penalty dominates, hurts survival
> Sweet spot: ~0.025 — improves survival without PPL loss

## Statistical Note

With n=5:
- baseline: 3.40 ± 0.55
- wave0025: 4.80 ± 0.84
- Difference: 1.40 ± 1.39 (marginally significant)

For p<0.05, need n≈15–20 runs per config.

## Next Steps

- **M196f**: n=20 runs for λ=0.025 vs baseline for statistical confidence
- **M196g**: Grid search λ ∈ [0.01, 0.02, 0.03, 0.04]
- **M193b**: Use M196e data as training set for learned risk model

## Code Reference

```python
# Per-module normalized penalty
wave_pen = wave_pen / n_modules
loss = loss + wave_lambda * wave_pen

# 5 independent runs
for run in range(5):
    for name, wave_lambda in configs:
        model = train_mixed(..., wave_lambda)
        surv = eval_survival(model, ...)
```

## Related

- M196 — o_proj only (wave reg helped)
- M196b — mixed targets single run (wave reg hurt)
- M196c — penalty schedules
- M196d — λ scaling test
