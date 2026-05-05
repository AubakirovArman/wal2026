# M164 — Real Cross-Model Vocabulary

**Question:** Can a WAL atom table built on Llama-3.1-8B be reused for gpt2?

## Method

- Base: Llama-3.1-8B (8B params, decoder-only transformer)
- Cross-model: gpt2 (124M params, decoder-only transformer)
- Build atom table (K=64) on Llama layer 0 weights
- Encode gpt2 weights with Llama atoms
- Compare to gpt2's own native atom table

## Results

| Metric | Value |
|--------|-------|
| Llama native MSE | 2.75e-07 |
| gpt2 cross MSE | 4.68e-03 |
| gpt2 native MSE | 1.27e-05 |
| **Cross/native ratio** | **368×** |

## Analysis

The cross-model MSE is **368× worse** than native encoding. This is a massive degradation — the atom table learned from Llama weights is completely inappropriate for gpt2.

Even though both are decoder-only transformers, the weight distributions differ significantly due to:
- Different initialization schemes
- Different training data and objectives
- Different scale (8B vs 124M)
- Different layer normalization placement

## Conclusion

**Shared atom vocabulary across models is NOT viable.** Each model needs its own atom table.

This kills the "universal weight format" dream for cross-model sharing. However, it does not affect the WAL+LoRA workflow within a single model family.

**Practical implication:** If you want to distribute WAL-encoded models, each model needs its own atom table (or the table must be bundled with the model).
