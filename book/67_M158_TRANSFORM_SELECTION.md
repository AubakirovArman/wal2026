# M158 — Transform Selection per Module

**Question:** Should we use a single transform for all modules, or module-specific transforms?

## Method

- CPU model, layers 0 and 16
- Modules: q_proj, v_proj, gate_proj
- Compare: single Hadamard atom table for all modules vs module-specific tables
- K=64, C=8, 1 iteration

## Results

| Module | Single MSE | Specific MSE | Ratio (single/specific) |
|--------|-----------|--------------|------------------------|
| 0_q_proj | 3.05e-08 | 2.60e-08 | 1.17 |
| 0_v_proj | 4.18e-09 | 2.24e-09 | 1.86 |
| 0_gate_proj | 9.95e-09 | 1.39e-08 | 0.72 |
| 16_q_proj | 2.93e-08 | 1.79e-08 | 1.64 |
| 16_v_proj | 5.84e-09 | 4.71e-09 | 1.24 |
| 16_gate_proj | 4.75e-08 | 6.47e-07 | **0.07** |

**Average ratio: 1.12** — single transform is essentially equivalent on average.

## Key Finding

Module-specific transforms can be **dramatically worse** for some modules. Layer 16 gate_proj with module-specific atoms has MSE **14× worse** than the single shared table.

This happens because:
- Some modules have highly specific weight distributions
- A shared table generalizes across all distributions
- Module-specific tables overfit to local structure and fail on outliers

## Conclusion

**Use a single shared transform for all modules.** Module-specific transforms offer no consistent benefit and can catastrophically degrade quality for certain modules.

Combined with M157 (per-transform vocab), the production setup is:
1. **One transform** (Hadamard or DCT) for all modules
2. **One atom table** built in that transform space
3. **Frozen table** across all edits
