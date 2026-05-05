# M180 — GPU High-K Transform-WAL

**Goal:** Test Transform-WAL with K=256, 512 on GPU.

## Method

- Llama-3.1-8B on GPU
- Layers 0 and 16
- Modules: q_proj, v_proj, gate_proj
- K values: 64, 256, 512
- Transforms: Raw, Hadamard, RandOrth(seed)
- DCT skipped (needs proper implementation)

## Results

### Avg MSE per Transform

| K | Raw | Hadamard | RandOrth |
|---|-----|----------|----------|
| 64 | 1.55e-08 | 3.76e-08 | 3.12e-09 |
| 256 | 8.91e-10 | 1.03e-09 | 1.69e-10 |
| 512 | 1.25e-09 | 3.14e-10 | 3.06e-11 |

### Per-Module Detail (K=512)

| Layer | Module | Raw | Hadamard | RandOrth |
|-------|--------|-----|----------|----------|
| 0 | q_proj | 2.50e-10 | 1.12e-10 | 4.01e-11 |
| 0 | v_proj | 1.31e-11 | 5.01e-12 | 1.75e-11 |
| 0 | gate_proj | 6.88e-09 | 5.30e-10 | 2.86e-11 |
| 16 | q_proj | 2.55e-10 | 6.83e-11 | 3.14e-11 |
| 16 | v_proj | 1.81e-11 | 1.27e-11 | 1.13e-11 |
| 16 | gate_proj | 7.32e-11 | 1.15e-09 | 5.46e-11 |

## Analysis

**K=512 is transformative:**
- RandOrth reaches 3e-11 MSE — near-lossless for practical purposes
- Hadamard reaches 3e-10 MSE — 4× better than Raw at K=512
- Raw at K=512: 1.25e-09 — also very good

**Hadamard anomaly at K=64:**
- K=64: Hadamard 3.76e-08 is WORSE than Raw 1.55e-08
- K=256: Hadamard catches up to Raw
- K=512: Hadamard beats Raw by 4×

This suggests Hadamard needs sufficient atom budget to realize its advantage.

**Encode time:**
- K=64: ~0.1-1.5s per module
- K=256: ~0.5-3.3s per module
- K=512: ~1.5-6.2s per module

All acceptable for offline encoding.

## Conclusion

**High-K Transform-WAL is viable.** K=512 with Hadamard or RandOrth achieves MSE low enough that PPL degradation should be minimal. The critical next test is M181 — PPL gate on full model.

**Production path:**
- Use K=256 for fast encoding (MSE ~1e-09)
- Use K=512 for quality (MSE ~3e-10)
- Hadamard = zero metadata, good quality
- RandOrth(seed) = best quality, zero metadata
