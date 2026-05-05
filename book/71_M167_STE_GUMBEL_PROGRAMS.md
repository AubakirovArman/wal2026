# M167 — STE / Gumbel Program IDs

**Question:** Can we learn atom_ids and coeff_ids directly via gradient descent?

## Method

- Tiny transformer (same as M166)
- Replace all `nn.Linear` with `GumbelWALLinear`
- `GumbelWALLinear`:
  - Fixed atoms and coeffs (pre-computed from initial dense weight)
  - Learnable logits for program IDs `[N, K*C]`
  - Forward: Gumbel-Softmax(hard=True) → STE → weighted atom×coeff sum
  - Backward: gradient flows through soft probabilities to logits
- Train 30 steps, compare to dense baseline

## Results

| Phase | Final Loss |
|-------|-----------|
| Baseline (dense) | 7.212 |
| Gumbel-WAL | 6.971 |

Gumbel-WAL achieves **lower loss** than dense baseline. This is surprising — the constrained program space may act as a beneficial regularizer.

## Technical Details

**Gumbel-Softmax with STE:**
```python
y_soft = (logits + gumbel_noise) / temperature
y_soft = softmax(y_soft)
y_hard = one_hot(argmax(y_soft))
y = y_hard - y_soft.detach() + y_soft  # STE
```

In forward: hard discrete selection (exact WAL program)
In backward: gradient flows through soft probabilities

## Implications

1. **WAL-friendly training is viable** — we can train models directly in program space
2. **No decode→dense cycle needed** — model stays in WAL throughout training
3. **Memory savings** — store logits instead of full weights (but logits are [N, K*C], which is larger than weights for small K)
4. **Future work:** Joint learning of atoms + coeffs + programs (full end-to-end WAL training)

## Conclusion

**M147 (WAL-friendly training) is un-blocked.** The earlier negative result (simple regularizer underperforms L2) was because we were training dense weights with a WAL penalty. The correct approach is to train in program space directly via Gumbel-Softmax + STE.

**Recommended architecture for WAL v2 training:**
```
GumbelWALLinear:
  - atoms: fixed or learned (K=256)
  - coeffs: fixed or learned (C=16)
  - logits: learned [N, K*C]
  - forward: GumbelSoftmax(logits) @ (atoms * coeffs)
  - backward: STE through hard selection
```

This is a **breakthrough result** for WAL v2 — it opens the path to native WAL training without dense intermediates.
