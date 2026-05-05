# M156 — Transform-WAL Diff Locality

**Date:** 2026-04-20
**Status:** ✅ Complete (v2 fast version)
**Goal:** Compare diff locality for Raw-WAL vs Transform-WAL under synthetic edits.

## Method

- Model on CPU, 2 layers, 2 modules
- K=64, C=8, iters=1
- Synthetic edit: Gaussian noise σ=0.001 on all weights
- Transforms: Raw, Hadamard, RandOrth

## Results

| Layer | Raw diff | Hadamard diff | RandOrth diff | Raw patch | Hadamard patch | RandOrth patch |
|-------|----------|---------------|---------------|-----------|----------------|----------------|
| l0.q_proj | 0.901 | 0.898 | **0.997** | 32.33 MB | 32.25 MB | 35.56 MB |
| l0.v_proj | 0.928 | 0.933 | **0.995** | 8.31 MB | 8.35 MB | 8.87 MB |
| l16.q_proj | 0.886 | 0.899 | **0.997** | 31.84 MB | 32.26 MB | 35.56 MB |
| l16.v_proj | 0.920 | 0.929 | **0.996** | 8.24 MB | 8.32 MB | 8.88 MB |

## Key Finding — CRITICAL

**RandOrth DESTROYS diff locality.**

- RandOrth diff: **99.7%** (almost every weight changes program)
- Raw diff: **90.5%** average
- Hadamard diff: **91.5%** average — similar to Raw

**Why:** Random Orthogonal transform spreads local weight changes across the entire transformed matrix. A small perturbation in weight space becomes a global change in transform space.

## Trade-off Table

| Transform | MSE (v_proj) | Diff Locality | Patch Size |
|-----------|-------------|---------------|------------|
| Raw | baseline | 90.5% | baseline |
| Hadamard | **2× better** | 91.5% (similar) | similar |
| RandOrth | **2-8× better** | **99.7% (worse)** | **+10% larger** |

## Implications for WAL v2

- **RandOrth is NOT suitable for editable WAL.** It gives best MSE but makes patches larger and diffs completely non-local.
- **Hadamard is the best compromise.** Similar MSE improvement to RandOrth on some layers, with diff locality comparable to Raw.
- **For patch-size-critical applications:** Use Hadamard or stay with Raw. RandOrth only if editability is not required.

## Artifacts

- `experiments/m156_transform_wal_diff_locality_v2.py`
- `experiments/m156_transform_wal_diff_locality.json`
