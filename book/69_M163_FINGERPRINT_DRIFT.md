# M163 — Fingerprint Drift During Training

**Question:** How does fingerprint change as LoRA edit magnitude increases?

## Method

- Layer 0 q_proj, base model
- Synthetic LoRA: rank-4 random matrices
- Scale LoRA magnitude: [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
- Measure spectral fingerprint distance from base

## Results

| Scale | Distance from Base |
|-------|-------------------|
| 0.0 | 0.0000 |
| 0.001 | 0.0000 |
| 0.003 | 0.0003 |
| 0.01 | 0.0039 |
| 0.03 | 0.0373 |
| 0.1 | 0.3826 |
| 0.3 | 2.9453 |
| 1.0 | 13.0074 |

## Analysis

Fingerprint drift is **non-linear** with edit magnitude:
- **Tiny edits** (scale < 0.01): barely detectable (distance < 0.004)
- **Small edits** (scale 0.01–0.03): weak signal (0.004–0.04)
- **Medium edits** (scale 0.03–0.1): clear drift (0.04–0.38)
- **Large edits** (scale > 0.1): dominates fingerprint (>0.38)

The relationship is approximately **power-law**: distance ∝ scale^1.5–2.0

## Implications

1. **Safety scoring**: Fingerprint drift can supplement spectral norm safety score. Two independent signals = better guardrail.
2. **Training monitoring**: During LoRA training, fingerprint trajectory can detect divergence before PPL degrades.
3. **Threshold calibration**: Per-module thresholds needed. Attention layers drift faster than MLP.

## Conclusion

Fingerprint drift is a **viable secondary safety signal**. It cannot replace spectral norm (which directly measures delta magnitude), but provides orthogonal information about weight distribution changes.

**Recommended guardrail stack:**
1. Primary: Spectral norm safety score (direct, calibrated)
2. Secondary: Fingerprint drift (distribution-aware, catches non-spectral changes)
3. Tertiary: PPL gate (final validation on real data)
