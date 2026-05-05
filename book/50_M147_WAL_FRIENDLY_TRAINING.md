# M147 / Track 10: WAL-Friendly Training Probe

**Date:** 2026-04-20
**Status:** ❌ Negative result
**Goal:** Test if WAL-aware regularization improves weight compatibility with quantization.

## Method

```
1. Build atom table from model
2. For each weight, find nearest atom×coeff
3. Measure "WAL distance" = mean |weight - recon|
4. Apply one-step regularizer:
   a. WAL reg: push weights toward nearest atom×coeff
   b. L2 baseline: simple shrinkage toward 0
5. Compare improvement
```

## Results

### Per Layer

| Layer | Before | WAL reg | Improvement | L2 shrink | Improvement |
|-------|--------|---------|-------------|-----------|-------------|
| q_proj | 0.000018 | 0.000018 | **+1.0%** | 0.000017 | **+2.5%** |
| k_proj | 0.000046 | 0.000045 | **+1.0%** | 0.000044 | **+3.1%** |
| v_proj | 0.000007 | 0.000007 | **+1.0%** | 0.000007 | **+1.7%** |
| gate_proj | 0.000009 | 0.000009 | **+1.0%** | 0.000009 | **+2.2%** |
| o_proj | 0.000009 | 0.000009 | **+1.0%** | 0.000009 | **+2.3%** |

### Average

| Method | Avg Distance | Improvement |
|--------|-------------|-------------|
| Before | 0.000018 | — |
| WAL reg | 0.000017 | **+1.0%** |
| L2 shrink | 0.000017 | **+2.4%** |

## Analysis

### WAL Regularizer Underperforms

Pushing weights toward nearest atom×coefficient gives only **1.0% improvement** — worse than simple L2 shrinkage (2.4%).

**Why:**
1. **Nearest atom×coeff is not the optimal target.** The k-means atoms are a rough approximation, not the true manifold of good weights.
2. **L2 shrinkage is more principled.** It pushes weights toward 0, which is the mode of the distribution (most weights are near 0).
3. **One-step is insufficient.** Real WAL-friendly training would need iterative optimization, not a single gradient step.

### What Would Work Better

For actual WAL-friendly training:
- **Gumbel-softmax** for differentiable atom/coeff selection
- **Straight-Through Estimator (STE)** for hard quantization
- **Curriculum learning:** start with soft quantization, gradually harden
- **Loss function** that directly optimizes for low quantization error

## Conclusion

**Simple WAL regularizer does not outperform L2.**

Post-hoc WAL encoding works well enough that naive regularization toward quantization cells is unnecessary. True WAL-aware training would require more sophisticated methods (differentiable program selection, STE, curriculum).

## Artifacts

- `experiments/m147_wal_friendly_training.py`
- `experiments/m147_wal_friendly_training.json`
