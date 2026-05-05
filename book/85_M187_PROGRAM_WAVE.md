# M187 — Program-Wave

**Goal:** Test whether WAL program frequencies inherit wave structure from weights.

## Method

- Full Llama-3.1-8B, 32 layers, 7 modules
- Global atom table: K=32, C=4
- Encode each layer/module with shared atoms/coeffs
- FFT over layer index for each atom/coeff frequency

## Results

### Top FFT Amplitudes (all < 0.5 threshold)

| Module | Top Atom | Period | Amp |
|--------|----------|--------|-----|
| v_proj | 0 | 32.0 | **0.312** |
| v_proj | 29 | 32.0 | 0.277 |
| v_proj | 9 | 32.0 | 0.272 |
| q_proj | 29 | 32.0 | 0.233 |
| k_proj | 29 | 32.0 | 0.187 |
| o_proj | 0 | 32.0 | 0.155 |
| gate_proj | 0 | 32.0 | 0.059 |
| up_proj | 0 | 16.0 | 0.040 |
| down_proj | 0 | 32.0 | 0.037 |

All reported amplitudes are below the significance threshold of 0.5. Only v_proj shows a weak signal (0.31).

## Analysis

### Weights Have Waves, Programs Don't

M186 showed strong period-16/32 waves in weight norms (FFT amplitudes >> 1.0). M187 shows that after WAL encoding, the **atom frequency distributions are nearly flat** across depth.

Why?
1. **K-means clustering is a non-linear filter** — it collapses the continuous weight space into discrete bins, losing fine-grained depth structure
2. **Global atom table averages across layers** — the shared vocabulary is designed to be layer-agnostic
3. **Coefficient quantization further smooths** — C=4 bins are too coarse to capture depth variation

### v_proj Exception

v_proj shows the strongest (but still weak) program-wave signal. This aligns with M186 where v_proj had the most dramatic depth trend (2.09× growth). The extreme norm variation in v_proj partially survives clustering.

## Conclusion

**WAL programs do NOT inherit wave structure from weights.**

Key implication: wave-guided approaches must work with **weights or continuous representations**, not discrete WAL programs. The wave features discovered in M186 are properties of the raw weight space, not the compressed program space.

This redirects wave research:
- ❌ Wave-guided atom table design
- ❌ Wave-guided program assignment
- ✅ Wave-guided LoRA regularization (M188)
- ✅ Wave-guided safety scoring (M188, M152)
