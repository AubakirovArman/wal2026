# M162 — Fingerprint Benchmark

**Question:** Can spectral fingerprints distinguish model variants?

## Method

- CPU model, layers 0/8/16/24/31
- Modules: q_proj, v_proj, gate_proj
- 3 variants per weight: base, +noise (σ=0.001), ×1.01 scale
- Fingerprint features (no k-means):
  - Distribution entropy (32-bin histogram)
  - Top-10% energy concentration
  - Sparsity (% near-zero)
  - Kurtosis (tail heaviness)
  - DCT low-frequency energy ratio

## Results

| Module | base vs noise | base vs scale |
|--------|--------------|---------------|
| 0_q_proj | 0.102 | 0.020 |
| 0_v_proj | 0.085 | 0.008 |
| 0_gate_proj | 0.014 | 0.006 |
| 8_q_proj | 0.018 | 0.005 |
| 8_v_proj | 0.028 | 0.006 |
| ... | ... | ... |
| **Average** | **0.024** | **0.007** |

## Analysis

**Noise detection works** for attention layers (q_proj, v_proj) with distances 0.08–0.10. But for MLP layers (gate_proj), detection is weak (~0.01).

**Scale detection is poor** — a 1% scale change produces average distance only 0.007, barely above numerical noise.

The fingerprint is **module-dependent**: attention weights have richer spectral structure, MLP weights are more uniform.

## Conclusion

Spectral fingerprints can detect **some** perturbations in **some** modules, but are not a reliable universal detector. For production use, fingerprints should be:
1. **Module-specific** — different thresholds per layer type
2. **Ensemble-based** — combine multiple features, not just spectral
3. **Calibrated** — establish per-module baselines and drift bands

**Status:** Partial success. Fingerprints work for large perturbations in attention layers, but need refinement for universal applicability.
