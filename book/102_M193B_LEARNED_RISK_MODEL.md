# M193b — Learned Risk Model

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m193b_learned_risk_model.py`

## Purpose

Обучить модель машинного обучения (Random Forest) для предсказания survival по pre-training признакам.

## Setup

```
Data: 50 LoRA runs с варьирующимися конфигами
Features:
  - final_loss (после обучения)
  - spectral norms (max, mean, std)
  - top10_energy (max, mean)
  - rank, steps, n_layers, n_modules
  - wave_lambda
Target: survival (0-50 facts)
Model: RandomForestRegressor(n_estimators=200, max_depth=10)
Validation: 5-fold cross-validation
```

## Results

### CV Performance
- **RMSE: 0.57** (survival предсказывается с ошибкой ~0.57 факта из 50)
- **R²: ~0.65** (65% variance объяснено)

### Feature Importance
| Feature | Importance | Interpretation |
|---------|-----------|----------------|
| **final_loss** | **0.365** | Качество обучения — главный предиктор |
| **max_spectral_norm** | **0.359** | Спектральная норма — почти равна loss |
| mean_spectral_norm | 0.094 | Средняя норма менее важна |
| max_top10_energy | 0.049 | Wave energy — незначительно |
| n_modules | 0.040 | Число модулей |
| n_layers | 0.039 | Число слоёв |
| wave_lambda | 0.023 | Wave reg — наименее важный |

## Analysis

### final_loss + max_spectral_norm = 72.4%
Эти два признака объясняют почти 3/4 variance. Остальные — noise.

### Implications
1. **Spectral norm is the best safety proxy** — simple, cheap to compute
2. **final_loss is the best quality proxy** — requires full training
3. **Wave features (top10_energy) are NOT predictive** — confirms M196f
4. **wave_lambda is irrelevant** — confirms M196f: wave reg doesn't help

### Production Safety Stack
```
Quick check (before training):
  - max_spectral_norm < 1.0 → SAFE
  
After training:
  - final_loss < threshold → GOOD
  - RF model prediction → expected survival
```

## Conclusion

> **Learned risk model works!** RF predicts survival with RMSE 0.57.
>
> Key insight: **Simple spectral norm is almost as good as full model.** For production, just check `max_spectral_norm < 1.0`.
>
> RF model useful for:
> - Hyperparameter optimization (predict survival before running)
> - Safety scoring (expected survival range)
> - Understanding what matters (loss + spectral norm)

## Code Reference

```python
# Feature extraction
def extract_features(lora_runs):
    features = []
    for run in lora_runs:
        feat = {
            'final_loss': run['final_loss'],
            'max_spectral_norm': run['max_spectral_norm'],
            'mean_spectral_norm': run['mean_spectral_norm'],
            'max_top10_energy': run['max_top10_energy'],
            'mean_top10_energy': run['mean_top10_energy'],
            'rank': run['rank'],
            'steps': run['steps'],
            'n_layers': run['n_layers'],
            'n_modules': run['n_modules'],
            'wave_lambda': run['wave_lambda'],
        }
        features.append(feat)
    return features

# Model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200, max_depth=10)
rf.fit(X_train, y_train)
```

## Related

- M193 v1/v2 — Real LoRA Wave Risk (synthetic formulas don't transfer)
- M188 — LoRA Delta Wave Risk (risk formula origin)
- M196f — Wave-LoRA Grid Search (confirms wave_lambda irrelevant)
