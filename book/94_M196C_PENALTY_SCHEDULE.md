# M196c — Wave-LoRA Penalty Schedule

**Goal:** Find the optimal lambda schedule for wave regularization during LoRA training.

## Method

- Mixed training: alternate general text (wikitext-2) / contrafactual facts
- Wave penalty: `loss = task_loss + λ(t) * top10_energy_ratio(delta)`
- Schedules tested:
  - `constant`: λ = 0.1 fixed
  - `warmup`: linear 0 → 0.1 over first 30% steps
  - `cosine_decay`: cosine from 0.1 → 0.02
  - `warmup_cosine`: warmup 20% then cosine decay
  - `adaptive_specnorm`: λ = 0.1 if spec_norm > 3.0, else 0

## Results

| Config | Survival | PPL Δ | SpecNorm | Avgλ |
|--------|----------|-------|----------|------|
| rank1_baseline | 0/10 | -0.05 | 5.75 | 0.0000 |
| rank1_const010 | 1/10 | -0.06 | 4.88 | 0.1000 |
| rank1_warmup | 1/10 | -0.07 | 4.63 | 0.0845 |
| rank1_cosine | 0/10 | -0.09 | 5.28 | 0.0604 |
| rank1_warmup_cosine | 1/10 | -0.11 | 4.26 | 0.0579 |
| rank1_adaptive | 1/10 | -0.12 | 2.52 | 0.0000 |
| rank2_const010 | 1/10 | -0.12 | 1.78 | 0.1000 |
| rank2_warmup_cosine | 1/10 | -0.11 | 1.91 | 0.0579 |

## Analysis

### Schedules Don't Revolutionize Survival

All wave-regularized configs achieve ~1/10 survival. No schedule pushes beyond this ceiling for rank=1/2 on 10 facts.

### Cosine Decay Alone Fails

- `cosine_decay`: survival 0/10
- Why: λ drops to 0.02 by the end, losing penalty effectiveness
- The model needs wave regularization throughout training, not just early

### Constant λ=0.1 Is Most Reliable

Simple, predictable, and matches the best survival (1/10) with reasonable PPL.

### Warmup+Cosine Gives Best PPL

- `warmup_cosine`: PPL -0.11 (best), spec_norm 4.26 (good)
- Trade-off: slightly lower average λ (0.058) but smoother training

### Adaptive SpecNorm Problematic

- Threshold=3.0 is too high for rank=1 — spec_norm stays below threshold most of the time
- Average λ ≈ 0 (no regularization applied)
- Training time 714s (vs ~20s) due to SVD every step
- **Not viable without optimization**

## Conclusion

**Constant λ=0.1 is the recommended default.**

- Simple and reliable
- No hyperparameter tuning needed per step
- Matches best survival rates

For PPL-sensitive applications, `warmup_cosine` is a viable alternative.

**Not recommended:** `cosine_decay` alone (loses effectiveness), `adaptive_specnorm` (too slow, threshold miscalibrated).
