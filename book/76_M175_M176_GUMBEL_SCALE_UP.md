# M175 + M176 — Gumbel-WAL Scale-Up & Factorized Logits

## M175 — Scale-Up Test

**Goal:** Does Gumbel-WAL work beyond tiny models?

### Results

| Config | Dense Loss | Gumbel Loss | Status |
|--------|-----------|-------------|--------|
| 10M (d=256, L=4) | 9.6355 | 9.8274 | ✅ Works |
| 30M (d=512, L=6) | — | OOM | ❌ Failed |

### Problem

Full logits `[N, K*C]` = 2.6B parameters for 30M model. CUDA OOM.

---

## M176 — Factorized Logits Solution

**Goal:** Reduce memory by factorizing logits.

### Approach

Instead of `[N, K*C]`, use `[N, K] + [N, C]`:

```python
atom_prog = gumbel_softmax(atom_logits)  # [N, K]
coeff_prog = gumbel_softmax(coeff_logits)  # [N, C]
selected_atoms = atom_prog @ atoms     # [N]
selected_coeffs = coeff_prog @ coeffs  # [N]
weight = selected_atoms * selected_coeffs  # [N]
```

### Results

| Config | Dense Loss | Factorized Loss | F/D | GPU Memory |
|--------|-----------|-----------------|-----|------------|
| 10M | 9.6594 | 10.3919 | 1.08 | 0.76 GB |
| 30M | 10.0573 | 10.7995 | 1.07 | 1.57 GB |

### Analysis

Factorized logits solve OOM. Both 10M and 30M models train successfully.

However, parameter count is still high:
- 10M dense → 192.6M factorized (23×)
- 30M dense → 397.9M factorized (13.6×)

The overhead comes from the output layer (vocab × d_model × K). For large vocabularies, this dominates.

### Next Steps

Further compression needed:
1. **Blockwise logits:** Share logits across weight blocks
2. **Low-rank logits:** `logits = U @ V` where U [N, r], V [r, K]
3. **Hash-based:** Use hash function to map position to shared logits
4. **Sparse logits:** Only store logits for active positions

Without further compression, Gumbel-WAL is viable for small models but impractical for large ones.
