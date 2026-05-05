# M46 WAL SCALAR 70B E2E V2

## Date
2026 (exact date from git log or experiment run)

## Goal
M46v2: WAL Scalar end-to-end on Llama 3.3 70B — lmax=2, K=128, skip spiky.

## Configuration
K=128, K_ATOMS=128, KMEANS_ITERS=5, batch=262144, iters=5

## Method / What was tested
See `experiments/m46_wal_scalar_70b_e2e_v2.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m46_wal_scalar_70b_e2e_v2.py`
- `experiments/m46_wal_scalar_70b_e2e_v2.log`