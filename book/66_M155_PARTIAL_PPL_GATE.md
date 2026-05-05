# M155 — Partial Model Transform-WAL PPL Gate

**Question:** If we apply Transform-WAL (Hadamard + K=64 atoms) to only a subset of layers, how much does PPL degrade?

## Method

- Full Llama-3.1-8B on GPU
- Precompute Transform-WAL decoded weights for ALL layers (K=64, C=8, 1 iteration)
- For each N in [0, 8, 16, 24, 31]: replace layers 0..N with decoded, keep rest original
- Measure PPL on WikiText-2 validation (128 tokens)

## Results

| N | PPL | Δ | Degradation |
|---|-----|---|-------------|
| Base | 4.3169 | — | — |
| 0 | 4.3459 | +0.0290 | +0.7% |
| 8 | 4.7870 | +0.4701 | +10.9% |
| 16 | 6.3131 | +1.9962 | +46.2% |
| 24 | 7.1527 | +2.8358 | +65.7% |
| 31 | 7.3830 | +3.0661 | +71.0% |

## Analysis

Even replacing a **single layer** (N=0) degrades PPL by +0.7%. This means K=64 atoms are too coarse for real model weights — the quantization error is perceptible in the output distribution.

For half the model (N=16), degradation is catastrophic (+46%). Full model WAL (N=31) gives PPL 7.38 vs base 4.32.

## Conclusion

**Transform-WAL with K=64 is not production-ready for full model encoding.** The atom count needs to be significantly higher (K=256 or K=1024) to achieve near-lossless reconstruction.

However, Transform-WAL remains valuable as:
1. **An analysis tool** — understanding weight structure in transform space
2. **A potential coarse compressor** — if PPL degradation is acceptable
3. **A research direction** — higher K may achieve viable quality

**Practical path:** WAL base (frozen vocab) + LoRA overlay remains the only production-viable workflow. Transform-WAL is a secondary research track, not a replacement.
