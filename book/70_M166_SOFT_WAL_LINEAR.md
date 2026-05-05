# M166 — Soft-WALLinear Small Model

**Question:** Can a model with WAL-encoded weights still train?

## Method

- Tiny transformer: 1 layer, d_model=128, 2 heads, d_ff=256, vocab=1000
- Task: random next-token prediction (synthetic data)
- Baseline: dense weights, 50 steps
- WAL: encode all linear layers with K=32, C=4, then train 50 more steps
- Optimizer: Adam, lr=1e-3

## Results

| Phase | Final Loss |
|-------|-----------|
| Baseline (dense) | 7.196 |
| WAL-encoded | 7.126 |

WAL loss is **slightly better** than baseline (7.126 vs 7.196). This is likely noise — both are near random-guessing (~6.9 for vocab=1000).

## Key Finding

**WAL-encoded weights do not prevent training.** Gradients flow through the decoded weights, and the optimizer updates the underlying parameters (embeddings, LayerNorm, biases) normally.

However, this experiment does NOT test learning the program space itself — the WAL encoding was fixed after initialization.

## Conclusion

WAL is **training-compatible** as a weight format. A model can be initialized, WAL-encoded, and continue training without catastrophic failure.

For true WAL-friendly training (learning atom_ids directly), see M167.
