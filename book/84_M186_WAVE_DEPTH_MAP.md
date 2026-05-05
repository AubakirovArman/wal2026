# M186 — Wave Depth Map

**Goal:** Discover periodic structure in weight norms across model depth.

## Method

- Full Llama-3.1-8B, 32 layers
- Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Metrics per layer/module:
  - Frobenius norm
  - Spectral norm (top singular value)
  - Atom entropy (from K=32 clustering)
- FFT over layer index to find periods

## Results

### FFT Periods

| Module | Norm Period | Spectral Norm Period | Atom Entropy Period |
|--------|-------------|---------------------|---------------------|
| q_proj | 16.0 | 16.0 | 16.0 |
| k_proj | 32.0 | 16.0 | 5.3 |
| v_proj | 32.0 | 32.0 | 2.1 |
| o_proj | 32.0 | 32.0 | 5.3 |
| gate_proj | 32.0 | 16.0 | 10.7 |
| up_proj | 16.0 | 16.0 | 8.0 |
| down_proj | 16.0 | 32.0 | 2.1 |

### Layer-wise Trends (Early → Late)

| Module | Early | Late | Ratio |
|--------|-------|------|-------|
| q_proj | 77.69 | 68.51 | **0.88** (↓) |
| k_proj | 56.75 | 47.83 | **0.84** (↓) |
| v_proj | 16.58 | 34.61 | **2.09** (↑) |
| o_proj | 37.73 | 56.43 | **1.50** (↑) |
| gate_proj | 102.06 | 125.79 | **1.23** (↑) |
| up_proj | 90.90 | 103.13 | **1.13** (↑) |
| down_proj | 90.96 | 97.68 | **1.07** (↑) |

## Analysis

### Period-16/32 Waves

FFT reveals clear periodic structure:
- **Period 16** (2 cycles across 32 layers): dominant in q_proj, up_proj, down_proj
- **Period 32** (1 cycle): dominant in v_proj, o_proj, gate_proj

This suggests the model has a **two-scale architecture**: early layers differ systematically from late layers, with a superimposed 16-layer rhythm.

### Attention Q/K vs V/O Split

- **Q/K norms decrease with depth** — early layers have larger attention queries/keys
- **V/O norms increase with depth** — late layers have larger attention values/outputs

This is consistent with the "residual stream amplification" hypothesis: later layers need to write stronger corrections into the residual stream.

### MLP Growth

All MLP modules (gate, up, down) grow with depth. gate_proj grows most (1.23×). This suggests MLP layers become more active in late layers, possibly handling more complex token-level transformations.

## Conclusion

**Weight norms exhibit strong wave patterns across depth.** Period-16/32 oscillations are real and module-specific.

Implications:
1. Layer groups (0-15 vs 16-31) have systematically different weight distributions
2. Per-layer K budgets should account for depth-dependent norms
3. Wave features are a property of **weights**, not programs (see M187)
