# M205 — Risk Dataset Expansion

## Date
2026-04-30

## Question
Can expanding the risk dataset from 50 to 70 points improve the learned risk model?

## Method
- Load existing m193b_data.csv (50 points)
- Run 20 probe experiments with varied configs (steps=50 for speed)
- Collect features + survival
- Retrain RF model on expanded dataset

## Probe Configs

20 configurations varying: rank (2,4,8), modules (1,2,4,6), layers (1,2,3,5), steps (50,100,200), lr (2e-5, 5e-5, 1e-4), lambda (0, 0.05, 0.1)

## Results

- Total data points: **70** (50 + 20)
- RF CV RMSE: **0.9310**

### Feature Importances

| Feature | Importance |
|---------|-----------|
| final_loss | 0.2068 |
| mean_top10_energy | 0.1985 |
| steps | 0.1899 |
| max_top10_energy | 0.1898 |
| max_spectral_norm | 0.1120 |
| mean_spectral_norm | 0.0488 |
| n_layers | 0.0197 |
| rank | 0.0140 |
| wave_lambda | 0.0129 |
| n_modules | 0.0077 |

## Comparison with M193b

| Metric | M193b (50 pts) | M205 (70 pts) |
|--------|---------------|---------------|
| CV RMSE | 0.57 | 0.93 |

## Why RMSE Got Worse

1. **Low variance in new data**: Almost all probe runs gave survival=3/50 (steps=50 is too weak)
2. **Unbalanced dataset**: Too many points with identical survival values
3. **Missing extreme configs**: No high-survival points (steps=400 gives 20/50, but wasn't probed)

## Lessons

- Dataset size alone doesn't improve model quality
- Need **stratified sampling** across survival spectrum:
  - Weak edits (survival 3-5): already have many
  - Medium edits (survival 8-15): need more
  - Strong edits (survival 20+): need more
- Steps=50 produces too little variance — need mix: 50, 100, 200, 400

## Future Work: M205b

Run stratified probes:
- 5 configs with steps=50 (weak, survival ~3-5)
- 5 configs with steps=100 (medium, survival ~5-10)
- 5 configs with steps=200 (strong, survival ~10-15)
- 5 configs with steps=400 (very strong, survival ~15-25)

This would give balanced coverage of the survival spectrum.

## Conclusion

> Naive dataset expansion (more points with same distribution) does NOT improve model.
> 
> Need **stratified sampling** across the target variable (survival) for meaningful improvement.

## Related
- M193b — Original learned risk model (50 points, RMSE 0.57)
- M196f — Grid search that found steps=400 gives 20/50 survival
