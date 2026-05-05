# M195b — Hadamard Adaptive K + k-means

**Goal:** Test adaptive K budget with proper k-means quantization (not uniform).

## Method

- Chunked torch k-means on GPU (avoids sklearn/OpenBLAS issues)
- 3 Lloyd iterations, chunk_size=1M
- K-means++ initialization on 100K sample
- Same percentile policy as M195

## Results

| Config | PPL | Δ | Size MB |
|--------|-----|---|---------|
| Baseline | 5.7152 | — | — |
| Uniform K=256 + k-means | 5.8536 | +0.1384 | 7424.00 |
| Adaptive K + k-means | **5.7818** | **+0.0666** | 7542.50 |

## Key Finding: Adaptive K 2× Better Than Uniform

- Uniform degradation: +0.1384 PPL
- Adaptive degradation: +0.0666 PPL
- **Adaptive improves degradation by 2.08×**

## k-means vs Uniform Quantization

| Method | Uniform Δ | Adaptive Δ |
|--------|-----------|------------|
| Uniform quant (M195) | +0.0615 | +0.0378 |
| k-means (M195b) | +0.1384 | +0.0666 |

k-means gives worse absolute PPL than uniform quantization. Possible reasons:
1. Only 3 Lloyd iterations — insufficient convergence
2. K-means++ init on small sample — suboptimal centers
3. Chunked assignment after convergence — floating point drift

**But the relative adaptive/uniform improvement is consistent**: ~40–50% less degradation in both cases.

## Size Analysis

Adaptive size (7542 MB) > Uniform size (7424 MB) because:
- K=512 modules (44, mostly gate_proj/up_proj) are large modules with +1 bit each
- K=128 modules (68, mostly q/k/v/o) are smaller modules with -1 bit each
- The K=512 overhead exceeds the K=128 savings

For real size reduction, a more aggressive K=128 policy is needed.

## Encode Time

1044s = 17.4 minutes for full model. Acceptable for offline, but needs optimization for production.

## Conclusion

**Adaptive K is conceptually validated across two quantization methods.**

Next steps for production:
1. More k-means iterations (5–10) with better init
2. More aggressive K=128 policy for size reduction
3. Target: M181-quality near-lossless + adaptive budget
