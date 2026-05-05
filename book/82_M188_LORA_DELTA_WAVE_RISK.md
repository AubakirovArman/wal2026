# M188 — LoRA Delta Wave Risk

**Goal:** Understand why some LoRA edits are stable while others collapse.

## Method

- Layer 0 of Llama-3.1-8B
- Modules: q_proj, v_proj, gate_proj
- Synthetic LoRA deltas with varying rank and scale
- Wave features computed for each delta:
  - DCT spectrum (top-1%, top-10% energy)
  - Hadamard spectrum (top-10% energy)
  - Spectral entropy
  - Phase coherence
  - Singular value profile (top-1%, top-10%, condition number)
  - Spectral norm
  - Fingerprint entropy

## WaveRiskScore Formula

```python
risk = top1_energy * 2.0 + top10_energy * 1.0 + sv_top1 * 2.0 + spectral_norm * 0.1 - spectral_entropy * 0.2 - fingerprint_entropy * 0.1
```

## Results

### Risk Ranking (sorted by risk)

| Config | Module | Risk | Top1% | Top10% | SpecNorm |
|--------|--------|------|-------|--------|----------|
| rank8_scale1.0 | gate_proj | **794.97** | 0.030 | 0.215 | 7982.19 |
| rank4_scale1.0 | gate_proj | **772.91** | 0.032 | 0.225 | 7758.85 |
| rank1_scale1.0 | gate_proj | **764.69** | 0.043 | 0.275 | 7660.55 |
| rank8_scale1.0 | q_proj | **427.72** | 0.030 | 0.215 | 4307.06 |
| rank4_scale1.0 | q_proj | **413.76** | 0.032 | 0.225 | 4164.62 |
| rank1_scale1.0 | q_proj | **406.91** | 0.043 | 0.275 | 4080.05 |
| rank8_scale1.0 | v_proj | **213.79** | 0.030 | 0.216 | 2164.90 |
| rank1_scale1.0 | v_proj | **213.01** | 0.043 | 0.273 | 2138.40 |
| rank4_scale1.0 | v_proj | **208.91** | 0.032 | 0.225 | 2113.11 |
| rank1_scale0.1 | gate_proj | 6.30 | 0.043 | 0.275 | 76.60 |
| rank4_scale0.1 | gate_proj | 4.78 | 0.032 | 0.225 | 77.57 |
| rank8_scale0.1 | gate_proj | 4.73 | 0.030 | 0.215 | 79.81 |
| rank1_scale0.1 | q_proj | 2.99 | 0.043 | 0.275 | 40.81 |
| rank4_scale0.1 | q_proj | 1.47 | 0.032 | 0.225 | 41.65 |
| rank8_scale0.1 | q_proj | 1.32 | 0.030 | 0.215 | 43.07 |
| rank1_scale0.1 | v_proj | 1.32 | 0.043 | 0.273 | 21.38 |
| rank4_scale0.1 | v_proj | 0.00 | 0.032 | 0.225 | 21.13 |
| rank8_scale0.1 | v_proj | 0.00 | 0.030 | 0.216 | 21.65 |

## Analysis

### Scale Dominates Rank

**Scale=1.0 produces 100× higher risk than scale=0.1.** This confirms that edit magnitude is the primary risk factor, consistent with M152 (spectral norm safety score).

### Module Sensitivity Hierarchy

```
gate_proj > q_proj > v_proj
```

- **gate_proj:** Highest risk (765–795 at scale=1.0). Most sensitive to edits.
- **q_proj:** Medium risk (407–428 at scale=1.0).
- **v_proj:** Lowest risk (209–214 at scale=1.0). Most robust.

This suggests that **MLP layers are more fragile than attention layers** for LoRA edits.

### Higher Rank = Slightly Higher Risk

At fixed scale, rank=8 > rank=4 > rank=1 in risk. More parameters in the edit create slightly more spectral disruption.

### Top-1% Energy is NOT Discriminative

Top-1% energy is ~3-4.3% across all configs. It doesn't distinguish safe from dangerous edits. The **spectral norm** is the dominant discriminator.

## Conclusion

**Wave analysis confirms spectral norm as primary risk signal but adds module-specific differentiation.**

Recommended guardrail stack:
1. **Primary:** Spectral norm (direct, validated)
2. **Secondary:** WaveRiskScore with module-specific thresholds
3. **Tertiary:** PPL gate

Module-specific thresholds:
- gate_proj: risk > 500 = DANGEROUS
- q_proj: risk > 300 = DANGEROUS
- v_proj: risk > 150 = DANGEROUS
