# M177 — Gumbel-WAL Temperature Schedule

**Goal:** Find temperature schedule that prevents program collapse.

## Method

- Tiny model (vocab=100, d=64, factorized Gumbel-WAL)
- 6 temperature schedules tested
- 50 training steps each
- Metrics: final loss, program entropy, dead atoms, loss stability

## Schedules Tested

| Schedule | Start | End | Type |
|----------|-------|-----|------|
| constant_high | 2.0 | 2.0 | flat |
| constant_low | 0.1 | 0.1 | flat |
| linear_decay | 2.0 | 0.1 | linear |
| cosine_decay | 2.0 | 0.1 | cosine |
| exp_decay | 2.0 | 0.1 | exponential |
| linear_sharp | 5.0 | 0.03 | linear |

## Results

| Schedule | Loss | Entropy | Dead | Stability |
|----------|------|---------|------|-----------|
| constant_high | 4.6021 | 3.4657 | 0 | 0.0058 |
| constant_low | 4.6115 | 3.4657 | 0 | 0.0062 |
| linear_decay | 4.6467 | 3.4657 | 0 | 0.0222 |
| **cosine_decay** | **4.6067** | **3.4657** | **0** | **0.0014** |
| exp_decay | 4.6123 | 3.4657 | 0 | 0.0113 |
| linear_sharp | 4.6010 | 3.4657 | 0 | 0.0150 |

## Analysis

**No collapse in any schedule.** All maintain full entropy and zero dead atoms.

**Cosine decay has best stability** (lowest loss variance in final steps). This makes sense: cosine provides smooth transition from exploration (high temp) to exploitation (low temp).

**Linear decay has worst stability** — sharp temperature drops cause loss fluctuations.

## Recommendation

```python
def get_temperature(step, total, start=2.0, end=0.1):
    progress = step / total
    return end + (start - end) * (1 + cos(pi * progress)) / 2
```

Use **cosine decay** for Gumbel-WAL training. Start at 2.0 for exploration, end at 0.1 for hard programs.

## Limitations

- Tested on tiny model only
- Real models may show different collapse behavior
- Entropy was constant across schedules — needs larger model to differentiate
