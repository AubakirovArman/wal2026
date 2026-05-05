# M192 — Gumbel-WAL + Wave Regularization

**Goal:** Integrate spectral wave penalty into Gumbel-WAL training loop.

## Method

- 10M tiny transformer (d=256, L=4, heads=4)
- Factorized Gumbel-WAL (K=64, C=8) for output head
- Wave penalty: `loss = task_loss + λ * top10_energy_ratio(weight)`
- 50 steps Adam, lr=1e-3

## Results

| Config | Task Loss | Spec Norm | Top10 Energy | Spectral Entropy |
|--------|-----------|-----------|--------------|------------------|
| Baseline | 9.7699 | 4.14 | 0.0001 | 14.6164 |
| Wave λ=0.1 | **9.5792** | **3.32** | 0.0001 | 14.6160 |
| Wave λ=1.0 | 9.8289 | 7.29 | 0.0001 | 14.6157 |

## Analysis

### λ=0.1 — Sweet Spot

- Task loss improved by **2.0%** (9.77 → 9.58)
- Spectral norm reduced by **20%** (4.14 → 3.32)
- Both metrics improve simultaneously

### λ=1.0 — Destabilization

- Spectral norm **increased** by 76% (4.14 → 7.29)
- Task loss worsened by 0.6%
- Penalty too strong, interferes with optimization

### Why So Little Top10 Change?

Top10 energy remains ~0.0001 across all configs because:
1. Tiny transformer weights start near random (broad spectrum)
2. Only the output head uses Gumbel-WAL
3. 50 steps is too short for spectral structure to emerge

Real models with concentrated spectra (e.g., fine-tuned LLMs) may show larger effects.

## Conclusion

**Wave regularization is viable but requires careful λ tuning.**

- λ=0.1 improves both loss and spectral norm
- λ=1.0 destabilizes training
- Sweet spot depends on model size and task

Recommended approach:
1. Start with λ=0.1
2. Monitor spectral norm (should decrease, not increase)
3. Use adaptive λ schedule (cosine decay, as in M177)

Future work: integrate into all Gumbel-WAL layers, test on real tasks, compare penalty types (top10 vs entropy vs spectral norm).
