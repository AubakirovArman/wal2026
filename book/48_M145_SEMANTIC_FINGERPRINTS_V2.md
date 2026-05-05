# M145 / Track 8: Semantic Fingerprints v2

**Date:** 2026-04-20
**Status:** ⚠️ Partial result
**Goal:** Build WAL-based model forensics benchmark.

## Background

M137 showed early positive result: WAL fingerprints can sense model state differences (small delta for seed, moderate for noise, large for K change). This track attempts a systematic benchmark.

## Method

```
1. Load base model
2. Create 8 synthetic variants:
   - Base
   - Noisy (small/medium/large Gaussian)
   - Quantized (int8 simulation)
   - Sparse (zero small weights)
   - Scaled up/down
3. Compute fingerprints from weight distributions:
   - Histogram entropy
   - Sparsity
   - Top singular value
   - SV entropy
   - Skewness, kurtosis
   - IQR, peakiness
4. k-NN classification (leave-one-out)
5. Measure pairwise separability
```

## Results

### Classification

| Test | Predicted | Correct? |
|------|-----------|----------|
| base | noisy_small | ❌ |
| noisy_small | base | ❌ |
| noisy_medium | noisy_small | ❌ |
| noisy_large | noisy_medium | ❌ |
| quantized | base | ❌ |
| sparse | quantized | ❌ |
| scaled_up | base | ❌ |
| scaled_down | noisy_small | ❌ |

**Accuracy: 0/8 = 0.0%**

### Pairwise Separability

| Pair | Distance | Separable (>1.0)? |
|------|----------|-------------------|
| base vs noisy_large | **11.16** | ✅ |
| base vs sparse | **19.99** | ✅ |
| noisy_large vs sparse | **31.08** | ✅ |
| noisy_medium vs sparse | **20.26** | ✅ |
| scaled_up vs scaled_down | 1.29 | ✅ |
| base vs noisy_small | 0.0048 | ❌ |
| base vs noisy_medium | 0.267 | ❌ |
| base vs quantized | 0.458 | ❌ |

**Separable pairs: 14/28 (50%)**

## Analysis

### What Works
Fingerprints **can** distinguish sufficiently different variants:
- **Sparse** (zero threshold=0.01) is radically different from all others
- **Noisy_large** (σ=0.01) is clearly separable from base
- **50% of pairs** are separable at threshold 1.0

### What Doesn't Work
- **k-NN fails** because many synthetic variants are too similar
- **base vs noisy_small** (σ=0.0001): distance = 0.0048 — essentially identical
- **scaled_up vs scaled_down**: distance = 1.29 — borderline

### Root Cause
The experimental design uses **synthetic variants on the same model**. Real-world variants (base vs instruct vs code) would be much more different.

### Comparison with M137
M137 showed:
- Different seed: small but detectable shift
- Different K: massive shift (entropy drops 1.02, atoms_used halves)
- Noise + re-encode: moderate shift

This confirms fingerprints are sensitive to **real architectural changes**.

## Conclusion

**Fingerprinting is promising but needs real models.**

Synthetic variants on the same model are too similar for reliable classification. However:
- Sufficiently different variants **are** separable
- M137 proved real changes (seed, K, noise) produce detectable shifts
- Next step: benchmark on actual fine-tuned models

## Recommendations for Real Benchmark

```python
models_to_test = [
    "meta-llama/Llama-3.1-8B",           # base
    "meta-llama/Llama-3.1-8B-Instruct",  # instruct
    "codellama/CodeLlama-7b-hf",         # code
    # math, medical, safety variants
]
```

## Artifacts

- `experiments/m145_semantic_fingerprints_v2.py`
- `experiments/m145_semantic_fingerprints_v2.json`
