# M191 — Wave-Regularized LoRA

**Goal:** Test whether spectral concentration penalty can reshape LoRA deltas.

## Method

- Layer 0 of Llama-3.1-8B
- Modules: q_proj, v_proj, gate_proj
- Synthetic LoRA deltas (randn init)
- Post-hoc wave regularization:
  ```python
  loss = ||delta - delta_target||² + λ * spectral_concentration(delta)
  ```
- λ = 0.5, 20 gradient steps

## Results

| Module | Δtop1 | Δtop10 | Δimpact |
|--------|-------|--------|---------|
| q_proj | -0.001 | -0.008 | +0.0053 |
| v_proj | -0.001 | -0.010 | -0.0740 |
| gate_proj | -0.000 | -0.004 | +0.0026 |

All changes are minimal (< 1% relative).

## Analysis

### Why So Little Effect?

1. **Synthetic randn deltas already have broad spectra** — there's little concentration to remove
2. **Post-hoc projection is weak** — 20 steps with small λ can't fundamentally reshape the delta
3. **No task loss guidance** — the regularizer doesn't know which directions matter for the task

### Real LoRA May Differ

Real task-trained LoRA deltas often have concentrated spectra (low-rank structure). A wave regularizer might have more impact on real deltas than on random ones.

## Conclusion

**Wave regularization is conceptually valid but requires integration into the training loop.**

Post-hoc spectral smoothing is too weak. For production:
1. Integrate `λ * spectral_concentration(delta)` into the task loss
2. Use λ = 1.0–5.0 (not 0.5)
3. Validate on real downstream task PPL, not just synthetic deltas

See M192 for Gumbel-WAL + wave regularization integration.
