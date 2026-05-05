# M165 — Cross-Architecture Negative Control

**Question:** Can a Llama atom table work on a completely different architecture (distilbert)?

## Method

- Base: Llama-3.1-8B (decoder-only, RoPE, SwiGLU)
- Cross-arch: distilbert-base-uncased (encoder-only, absolute pos, GELU)
- Build atom table on Llama weights
- Encode BERT weights with Llama atoms
- Compare to BERT's native table

## Results

| Metric | Value |
|--------|-------|
| Llama native MSE | 2.75e-07 |
| BERT cross MSE | 4.13e-06 |
| BERT native MSE | 5.04e-07 |
| **Cross/native ratio** | **8.2×** |

## Analysis

Cross-architecture encoding is **8× worse** than native. While not as catastrophic as cross-model (368×), it is still a significant degradation.

The smaller ratio (8× vs 368×) is likely because:
1. BERT and Llama both use transformer blocks with similar weight shapes
2. Attention mechanisms are fundamentally similar (Q, K, V projections)
3. Both are trained on large text corpora

However, the 8× degradation is still unacceptable for production use.

## Conclusion

**Atom tables are architecture-specific.** Even within the transformer family, encoder vs decoder weights have sufficiently different distributions that shared atoms fail.

This confirms the negative control hypothesis: WAL atom tables capture model-specific weight structure, not universal properties of neural networks.

**Practical implication:** WAL is a model-specific format. Each model checkpoint must include its own atom table (or the table must be derived deterministically from the weights themselves).
