# M26 B2 Narrow Gate

Date: 2026-04-19

Code path:
- `src/triton_stage_local_hot_cold.py`: `stage_local_hot_cold_b2`
- `src/runtime.py`: `stage_local_hot_cold_b2`
- `experiments/m26_b2_narrow_gate.py`

Goal:
- Check whether the true staged hot-palette kernel is honest enough and fast enough to justify a full same-encoding `l54_q_gu` run.

Authoritative command:

```bash
CUDA_VISIBLE_DEVICES=2,3,5 python experiments/m26_b2_narrow_gate.py --m 2048
```

Configuration:
- targets: `model.layers.54.mlp.gate_proj`, `model.layers.54.mlp.up_proj`
- active hot-path gate: `hot_min_stage_share = 0.0`
- viability stop-criterion: `hot_hit_rate_min >= 0.72`
- comparison baseline: `full_weight_hot_v2`

Results:

| target | onehot max abs diff | has_nan | hot_v2 ms | b2 ms | speedup | hit mean | hit min | pass |
|---|---:|:---:|---:|---:|---:|---:|---:|:---:|
| `gate_proj` | 0.001953125 | False | 78.8377 | 68.3557 | 1.1533 | 0.07696 | 0.07455 | No |
| `up_proj` | 0.001953125 | False | 78.8322 | 68.3550 | 1.1533 | 0.07692 | 0.07383 | No |

Additional diagnostic:
- With the historical `hot_min_stage_share = 0.6`, the hot path is skipped entirely on both layers (`hit_mean = 0.000`), so that setting is not an honest B2 gate.
- At `hot_topk = 32`, coverage rises only to about `0.15`, while B2 becomes slower again (`~148 ms`).

Verdict:
- B2 fixes the local compute shape that B1 still got wrong.
- B2 does not pass the agreed narrow gate for M26 stage 2.
- Do not run full same-encoding `l54_q_gu` 16w on this variant.
- Next step: redesign `_build_hot_cache` / stage-hot selection around true stage-local hit-rate economics.