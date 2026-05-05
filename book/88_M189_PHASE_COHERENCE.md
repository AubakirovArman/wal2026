# M189 — Phase Coherence Test

**Goal:** Test whether wave features are amplitude-driven or phase-sensitive.

## Method

- Llama-3.1-8B, layers 0, 16, 31
- 7 modules per layer
- Phase shuffle: preserve FFT amplitude spectrum, randomize phases with Hermitian symmetry
- Compare spectral features before/after shuffle

## Results

### Average % Change After Phase Shuffle

| Feature | Avg % Change | Invariant? |
|---------|-------------|------------|
| top1_energy | 0.03% | ✅ YES |
| top10_energy | 0.01% | ✅ YES |
| spectral_entropy | 0.00% | ✅ YES |
| spectral_norm | 59.71% | ❌ NO |

### Per-Module Examples (Layer 0)

| Module | top1 | top10 | sn | entropy |
|--------|------|-------|----|---------|
| q_proj | 0.0% | 0.0% | 91.0% | 0.0% |
| k_proj | 0.1% | 0.0% | 85.7% | 0.0% |
| v_proj | 0.0% | 0.0% | 64.5% | 0.0% |
| gate_proj | 0.0% | 0.0% | 29.3% | 0.0% |

## Analysis

### Amplitude Features Are Phase-Invariant

top1_energy, top10_energy, and spectral_entropy are determined entirely by the amplitude spectrum. Shuffling phases (while preserving amplitudes) leaves these metrics essentially unchanged (< 0.1%).

This means:
1. **Wave features are robust signals** — they don't depend on fine-grained phase structure
2. **Amplitude spectrum captures the essential wave structure** discovered in M186
3. **Phase information is irrelevant for energy-based wave analysis**

### Spectral Norm Is Phase-Sensitive

Spectral norm changes by 60% on average after phase shuffle. This is because:
- Spectral norm = maximum singular value = operator norm
- Singular values depend on the full matrix structure, including phase relationships between rows/columns
- Randomizing phases destroys the structured correlations that determine the operator norm

### Implications for Safety Scoring

From M188: spectral norm is the primary risk signal.
From M189: spectral norm is phase-sensitive, while amplitude features are phase-invariant.

This means spectral norm captures **structural perturbation risk** that amplitude features miss:
- Amplitude features: "how much energy is concentrated?"
- Spectral norm: "how much does the operator amplify inputs?"

Both are useful, but for safety: **spectral norm > amplitude features**.

## Conclusion

**Wave features are amplitude-driven and robust to phase noise. Spectral norm is phase-sensitive and captures structural risk.**

Recommended feature stack:
1. **Primary:** Spectral norm (phase-sensitive, structural risk)
2. **Secondary:** top10_energy (amplitude-driven, energy concentration)
3. **Tertiary:** Spectral entropy (amplitude-driven, uniformity)
