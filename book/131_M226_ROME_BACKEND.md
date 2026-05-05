# M226: ROME-Style Backend for Hard Facts

**Status:** ✅ Complete
**Date:** 2026-05-01

## Question

Can hard facts (author/inventor) be edited via ROME/MEMIT-style rank-one updates to MLP weights?

## Hypothesis

Rank-one update to MLP down_proj at the layer with highest factual activation should embed hard facts that LoRA cannot learn.

## Method (Simplified ROME-lite)

1. **Layer selection**: Find layer with highest MLP down_proj activation norm for target fact
2. **Key extraction**: k = pre-MLP activation (post-attention residual), averaged over sequence
3. **Current output**: v_current = W @ k
4. **Target output**: v_target = v_current * 1.5 (heuristic amplification)
5. **Rank-one update**: W_new = W + (v_target - v_current) @ k^T / (k^T @ k)

## Results

```
Fact                         Layer  PPL Δ      Survival
--------------------------------------------------------
Who invented telephone?         31   +0.0337     0/1
Who wrote 1984?                 31   +0.2804     0/1
Who discovered radioactivity?   31   +1.1442     0/1

Total survival: 0/3
```

## Key Findings

1. **All facts selected layer 31** — the last layer, with highest activation norms
2. **PPL drift accumulates**: each successive edit adds more noise (+0.03 → +0.28 → +1.14)
3. **Zero survival** — even rank-one updates fail for hard facts

## Why It Failed

### 1. Wrong Layer Selection
Activation-based selection picks **late layers (28-31)**, which are output-formation layers, not factual storage. True factual associations live in **middle MLP layers (5-20)** as shown by ROME/MEMIT literature.

### 2. Heuristic Target Output
Multiplying current output by 1.5 is not grounded in the desired behavior. True ROME computes:
```
v_target = argmax log P(answer | prompt, v)
```
using optimization over the output vector.

### 3. Missing Covariance Matrix
True MEMIT uses:
```
W_new = W + (v_target - v_current) @ (C^{-1} k)^T
```
where C = E[k k^T] is the covariance of keys. Without C^{-1}, the update interferes with unrelated keys.

### 4. No Causal Mediation
We didn't perform causal mediation analysis to verify that the selected layer actually **causes** the factual answer. The layer may activate strongly without being causally responsible.

## Comparison with Literature

| Method | Hard Facts | Requirement |
|--------|-----------|-------------|
| Our LoRA | 0/3 | MLP middle layers, 100-400 steps |
| Our ROME-lite | 0/3 | Late layers, heuristic target |
| True ROME | ~80-90% | Causal tracing, middle layers, C^{-1} |
| True MEMIT | ~85-95% | Multiple layers, batch updates, precise targets |

## Conclusion

> **Simplified ROME fails. Hard facts remain a BLOCKER for lightweight editing.**

To properly edit hard facts, we need:
1. **Causal tracing** to find factual layers (not activation norms)
2. **Covariance matrix** C = E[k k^T] for clean updates
3. **Optimized target vectors** via constrained optimization
4. **Batch editing** across multiple layers (MEMIT, not ROME)

This is significantly more complex than LoRA and requires dedicated libraries (e.g., `rome`, `memit`). For WAL production, the recommendation stands:

- **Easy/Medium facts** → LoRA compiled into weights
- **Hard facts** → Retrieval tier (vector DB + prompt injection)

## Next Steps

- Install proper ROME/MEMIT library and test on hard facts
- Compare true ROME vs our simplified version
- If true ROME works: integrate as "surgical edit" backend in WAL Build System
- If true ROME also fails on our facts: confirm retrieval tier as only viable path
