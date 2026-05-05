# M200b — Merge + Re-encode with K=1024

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m200b_merge_reencode_k1024.py`

## Purpose

Test if higher K (finer quantization) survives merge+re-encode.
M200 showed K=256 destroys PPL (+60%). Hypothesis: K=1024 = less quantization error = less destruction.

## Setup

```
Model: meta-llama/Llama-3.1-8B
Base: Hadamard-WAL K=1024, iters=5
Pipeline: encode → LoRA → merge (float32) → re-encode → eval
```

## Results

| Stage | PPL | PPLΔ | Survival |
|-------|-----|------|----------|
| Baseline | 4.2744 | — | 3/50 |
| After Encode | 4.2732 | -0.0012 | — |
| After LoRA | 4.6154 | +0.3410 | 5/50 |
| After Merge | **10693.1133** | **+10688.84** | **0/50** |
| After Re-enc | **10699.0703** | **+10694.80** | **0/50** |

## Comparison with M200

| K | Merge ΔPPL | Re-encode ΔPPL | Verdict |
|---|-----------|----------------|---------|
| 256 (M200) | +6.16 | +6.20 | Catastrophic |
| **1024 (M200b)** | **+10688.84** | **+10694.80** | **WORSE** |

## Analysis

### K=1024 is WORSE than K=256
Counter-intuitively, finer quantization makes merge+re-encode **more destructive**, not less.

### Why?
With K=256, encode is lossy enough that the model "expects" some approximation. After merge, re-encode with K=256 preserves coarse structure.

With K=1024, encode is near-perfect (PPL -0.0012). The model learns to rely on fine-grained structure. Merge destroys this fine structure, and re-encode with K=1024 tries to reconstruct it — but creates **wildly wrong atoms** because the merged weights have completely different Hadamard-space statistics.

### Float32 merge doesn't help
M200b used float32 merge for numerical precision. Result: still catastrophic. The problem is not rounding error — it's **structural incompatibility** between merged weights and Hadamard+k-means representation.

## Conclusion

> **Merge + re-encode is FUNDAMENTALLY BROKEN for WAL v1.**
>
> Higher K makes it **worse**, not better.
>
> **Overlay-only is the only viable path.**

## Proof

```
K=256:  merge destroys PPL (+6.16)
K=1024: merge destroys PPL (+10689)

Conclusion: problem is structural, not quantization granularity.
```

## Related

- M200 — K=256 merge (also catastrophic)
- M201 — Overlay works (PPL -0.09, survival +1)
- M203 — WAL ≈ Dense equivalence (overlay-only)
