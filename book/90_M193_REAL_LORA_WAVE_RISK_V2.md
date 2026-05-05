# M193 v2 — Real LoRA Wave Risk Calibration (Improved)

**Goal:** Calibrate wave risk metrics on real trained LoRA with proper PPL measurement.

## Improvements over v1

1. **wikitext-2 for PPL** — realistic baseline (5.72)
2. **Mixed training** — alternate general text / contrafactual to prevent catastrophic forgetting
3. **AdamW + weight_decay** — better regularization
4. **10 configs** — rank={1,2,4,8} × steps={25,50,100}

## Results

| Config | Survival | PPL Δ | SpecNorm | WaveRisk |
|--------|----------|-------|----------|----------|
| rank1_steps50 | 0/10 | **+0.01** | 3.48 | -0.95 |
| rank1_steps100 | 0/10 | **+0.02** | 4.41 | -0.86 |
| rank2_steps50 | 1/10 | +0.01 | 2.20 | -1.98 |
| rank2_steps100 | 1/10 | +0.06 | 3.66 | -1.82 |
| rank4_steps25 | 1/10 | +0.05 | 0.95 | -2.45 |
| rank4_steps50 | 1/10 | +0.07 | 1.60 | -2.38 |
| rank4_steps100 | 1/10 | +0.14 | 2.85 | -2.15 |
| rank8_steps25 | 1/10 | +0.13 | 0.68 | -2.71 |
| rank8_steps50 | 1/10 | +0.12 | 0.88 | -2.71 |
| rank8_steps100 | 1/10 | +0.12 | 1.27 | -2.73 |

## Analysis

### Mixed Training Prevents Catastrophic Forgetting

Unlike v1 where PPL increased by 1000×, v2 keeps PPL stable (+0.01 to +0.14). Alternating between general corpus (wikitext-2) and contrafactual facts preserves model knowledge.

### Inverse Correlation: Spectral Norm vs PPL

Counter-intuitively, higher spectral norm correlates with better PPL:
- rank1 (spec_norm=3.48–4.41): PPL +0.01–0.02 (best)
- rank8 (spec_norm=0.68–1.27): PPL +0.12–0.13 (worst)

Explanation: lower rank means fewer parameters, less overfitting on contrafactual facts, better preservation of general knowledge. But...

### Rank-1 Cannot Learn 10 Facts

Survival is 0/10 for rank1 and 1/10 for all others. Rank=1 has insufficient capacity for 10 contrafactual facts.

### The PPL vs Survival Trade-off

```
Best PPL:   rank1_steps50  (+0.01, 0/10 survival)
Best balance: rank2_steps50 (+0.01, 1/10 survival)
Worst PPL:  rank4_steps100 (+0.14, 1/10 survival)
```

### WaveRiskScore Recalibration

Linear regression on 10 data points:
- spectral_norm: +0.0194
- sv_top1: -0.4997
- spectral_entropy: -0.0149
- top1/top10: large coefficients but tiny values
- Residuals: 0.0065 (excellent fit)

**Caveat:** 10 points / 5 parameters = overfitting. Needs validation on larger dataset.

## Conclusions

1. **Mixed training is essential** for LoRA+WAL workflow
2. **Low rank preserves PPL** but limits fact learning
3. **Spectral norm is informative** but relationship is non-monotonic
4. **WaveRiskScore needs more data** for reliable calibration

## Recommendations

- For **PPL preservation**: use low rank, few steps
- For **fact learning**: use rank≥2, enough steps
- For **safety scoring**: combine spectral norm with survival metrics
- Next: validate recalibrated score on held-out configs
