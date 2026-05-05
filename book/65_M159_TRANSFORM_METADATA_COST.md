# M159 — Transform Metadata Cost

**Date:** 2026-04-20
**Status:** ✅ Complete
**Goal:** Compare metadata storage overhead for different transforms.

## Method

Analytical calculation of metadata bytes per transform strategy.

## Results

### Per Layer (7 modules)

| Transform | Metadata per Layer |
|-----------|-------------------|
| Raw | 0 MB |
| Hadamard | 0 MB |
| DCT | 0 MB |
| RandOrth (full Q) | **3,079 MB** |
| RandOrth (seed) | **0 MB** |
| RandOrth (shared) | **96 MB** |

### Full Model (32 layers)

| Transform | Total Metadata |
|-----------|---------------|
| Raw | **0.00 MB** |
| Hadamard | **0.00 MB** |
| DCT | **0.00 MB** |
| RandOrth (full Q) | **98,516 MB** (~98.5 GB) |
| RandOrth (seed) | **0.00 MB** |
| RandOrth (shared) | **3,079 MB** (~3 GB) |

## Key Findings

1. **Hadamard/DCT: zero metadata** — deterministic from dimensions, no storage needed
2. **RandOrth full Q: 98.5 GB** — completely impractical for production
3. **RandOrth seed-based: 0 bytes** — only viable RandOrth option
4. **RandOrth shared: 3 GB** — expensive but manageable

## Production Recommendation

| Transform | Metadata | MSE Quality | Diff Locality | Production Viable |
|-----------|----------|-------------|---------------|-------------------|
| Hadamard | 0 MB | Good | Good | ✅ YES |
| DCT | 0 MB | Good | Unknown | ✅ YES |
| RandOrth (seed) | 0 MB | Best | Terrible | ⚠️ Only if no edits |
| RandOrth (full) | 98 GB | Best | Terrible | ❌ NO |

## Artifacts

- `experiments/m159_transform_metadata_cost.py`
- `experiments/m159_transform_metadata_cost.json`
