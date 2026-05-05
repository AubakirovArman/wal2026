# M169 — WAL Ablation Dashboard

**Date:** 2026-04-20
**Status:** ✅ Complete
**Goal:** Aggregate all experimental results into a unified comparison table.

## Method

Loads all `experiments/m*_*.json` files and extracts key metrics (MSE, status, mode).

## Dashboard Output

| Experiment | Mode | MSE | Status |
|-----------|------|-----|--------|
| m142_transform_wal_probe.json | Transform-WAL Probe | N/A (structure differs) | complete |
| m143_wave_atom_isa.json | Wave-Atom ISA | N/A | negative |
| m144_graph_wal_probe.json | Graph-WAL | N/A | negative |
| m145_semantic_fingerprints_v2.json | Fingerprints | N/A | partial |
| m147_wal_friendly_training.json | WAL-Friendly Training | N/A | negative |
| m154_fix_hadamard.json | Hadamard-WAL | **6.58e-07** | complete |

## Key Observations

1. **Hadamard-WAL** is the only transform experiment with quantified MSE so far (M154)
2. **Negative results** dominate: Wave-Atom, Graph-WAL, WAL-Friendly Training all failed
3. **Partial results**: Fingerprints, Cross-Model Vocab need real model validation
4. **Missing data**: Many older experiments (M126-M141) use different JSON schemas

## Next Steps

- Re-run older experiments with standardized JSON output (M168)
- Add M149, M153 results when complete
- Include PPL, patch size, and diff locality metrics

## Artifacts

- `experiments/m169_wal_ablation_dashboard.py`
- `experiments/m169_wal_ablation_dashboard.json`
