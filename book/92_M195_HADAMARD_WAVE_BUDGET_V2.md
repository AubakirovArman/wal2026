# M195 — Hadamard Wave-Guided Budget v2

**Goal:** Allocate K per module using Hadamard-WAL based on wave risk.

## Improvements over M190

1. **Hadamard transform** instead of raw-WAL
2. **Percentile policy**: bottom 30% → K=128, middle 50% → K=256, top 20% → K=512
3. **Wave risk** = spectral_norm + 0.01 × frob_norm
4. **Uniform quantization** (O(N) memory, fast)

## Results

| Config | PPL | Δ |
|--------|-----|---|
| Baseline | 5.7152 | — |
| Uniform K=256 | 5.7768 | +0.0615 |
| **Adaptive K** | **5.7530** | **+0.0378** |

## Key Finding: Adaptive K Outperforms Uniform K=256

**This is a direct success after M190's failure:**
- Uniform K=256: PPL +0.0615
- Adaptive K: PPL +0.0378
- **Adaptive is better by 0.0237 PPL** (~40% less degradation)

M190 (raw-WAL adaptive) gave PPL +1.70. M195 (Hadamard adaptive) gives PPL +0.04. The difference is Hadamard transform.

## K Distribution

- K=128: 68 modules (30% lowest risk)
- K=256: 112 modules (50% middle risk)
- K=512: 44 modules (20% highest risk)

## Top Risk Modules (K=512)

```
Layer 18 gate_proj: risk=14.41
Layer 29 gate_proj: risk=14.71
Layer 31 gate_proj: risk=25.40
Layer  0 q_proj:    risk=32.89
Layer  0 k_proj:    risk=19.40
```

Consistent with M188: **gate_proj is the most sensitive module**, especially in late layers.

## Why It Works

1. Hadamard transform provides a good basis for scalar quantization
2. Percentile policy correctly distributes the budget
3. High-risk modules get more atoms (K=512)
4. Low-risk modules save size (K=128)

## Limitations

1. **Uniform quantization** is less accurate than k-means — PPL degradation is larger than M181
2. Size calculation needs per-module bit counting
3. Only PPL tested — downstream task metrics needed
4. Encode time 7s — fast but needs optimized runtime for production

## Comparison

| Exp | Method | PPL Δ | Result |
|-----|--------|-------|--------|
| M181 | Hadamard K=256 k-means | +0.0004 | Near-lossless |
| M190 | Raw-WAL adaptive K | +1.7009 | Failed |
| M195 | Hadamard adaptive K uniform | +0.0378 | Better than uniform |

## Conclusion

**Wave-guided adaptive K budget is viable with Hadamard-WAL.**

The production path:
1. Hadamard transform
2. Per-module wave risk assessment
3. Percentile-based K allocation
4. k-means quantization (not uniform) for near-lossless quality

Recommended policy:
- gate_proj (late layers): K=512
- q_proj (early layers): K=512
- v_proj, k_proj: K=128–256
- Stable modules: K=128
