# M143 / Track 6: Wave-Atom ISA Probe

**Date:** 2026-04-20
**Status:** ❌ Negative result
**Goal:** Test if wave-based atoms can replace scalar atoms in WAL.

## Hypothesis

> Instead of scalar atoms (atom[id] × coeff[id]), use wave generators: weight[i,j] = Σ A_k cos(ωx_k i + ωy_k j + φ_k) + residual

## Method

```
For each layer:
  1. Apply DCT2 → frequency coefficients
  2. Keep top-K coefficients (by magnitude)
  3. Reconstruct via IDCT2 with top-K only
  4. Compare with scalar WAL (256 k-means atoms)
  5. Test K = 64, 128, 256, 512, 1024
```

## Results

### Wave DCT K=256 vs Scalar WAL (256 atoms)

| Layer | Wave DCT MSE | Scalar WAL MSE | Scalar Better | Wave PSNR | Scalar PSNR |
|-------|-------------|----------------|---------------|-----------|-------------|
| q_proj | 0.000352 | 0.000005 | **65×** | 32.1 dB | 50.2 dB |
| k_proj | 0.000732 | 0.000007 | **112×** | 28.3 dB | 48.8 dB |
| v_proj | 0.000053 | 0.00000009 | **584×** | 18.6 dB | 46.3 dB |
| o_proj | 0.000070 | 0.000001 | **123×** | 34.9 dB | 55.8 dB |
| gate_proj | 0.000165 | 0.0000004 | **412×** | 33.4 dB | 59.5 dB |

**Wave-WAL wins: 0/5 layers**

### Effect of K

| K | q_proj MSE | Improvement vs K=64 |
|---|-----------|-------------------|
| 64 | 0.00035226 | 1.0× |
| 128 | 0.00035219 | 1.0002× |
| 256 | 0.00035207 | 1.0005× |
| 512 | 0.00035187 | 1.0011× |
| 1024 | 0.00035152 | 1.0021× |

Even K=1024 (4× more parameters than scalar WAL) gives only **0.2% improvement**.

## Analysis

### Why Wave-Atoms Fail

1. **Smooth energy spectrum:** DCT coefficients decay gradually. There is no sharp "elbow" where truncation is natural.

2. **Non-adaptive:** DCT basis is fixed (cosines). Scalar atoms adapt to the specific weight distribution via k-means.

3. **Global vs local:** Each DCT coefficient affects the entire matrix. Scalar atoms are local — each weight gets its own atom+coeff pair.

### Why Transform-WAL (Track 5) Succeeds

The key difference:
```
Track 5 (Transform-WAL):    W → Transform → Scalar Quantize → Inverse Transform
Track 6 (Wave-Atom ISA):    W → DCT → Keep top-K → IDCT
```

Track 5 applies transform **before** scalar quantization. The scalar quantizer still adapts to the transformed distribution.

Track 6 replaces scalar atoms with DCT coefficients. The DCT basis is fixed and non-adaptive.

## Conclusion

**Wave-atom ISA does not outperform scalar WAL.**

- Scalar atom × coeff remains the best base ISA
- Transform should be applied **before** quantization, not **instead of** atoms
- DCT/FFT truncation loses too much information for weight matrices

## Implications for WAL v2

| Direction | Verdict |
|-----------|---------|
| Transform BEFORE scalar WAL | ✅ Works (Track 5) |
| Transform REPLACE scalar WAL | ❌ Fails (Track 6) |

## Artifacts

- `experiments/m143_wave_atom_isa.py`
- `experiments/m143_wave_atom_isa.json`
