# M26 B3 Narrow Gate

Date: 2026-04-19

Code path:
- `src/runtime.py`: `stage_local_hot_cold_b3`
- `src/triton_stage_local_hot_cold.py`: B2 kernel reused by B3
- `experiments/m26_b3_narrow_gate.py`

Goal:
- Test the concrete B3 hypothesis: keep the B2 compute shape, but replace the old weak hot selection with a thresholded stage-local policy.

Important runtime detail:
- B3 also fixes the practical Triton launch restriction for this family by padding staged hot palettes to the next power of two, so non-power-of-2 settings such as `hot_topk = 48` are now executable.

Authoritative commands:

```bash
CUDA_VISIBLE_DEVICES=2,3,5 python experiments/m26_b3_narrow_gate.py --m 2048 --hot-topk 48
CUDA_VISIBLE_DEVICES=2,3,5 python experiments/m26_b3_narrow_gate.py --m 2048 --hot-topk 64
```

Configuration:
- targets: `model.layers.54.mlp.gate_proj`, `model.layers.54.mlp.up_proj`
- baseline: `full_weight_hot_v2`
- strategy under test: `stage_local_hot_cold_b3`
- score mode: `count`
- threshold ratio: `0.65`
- viability targets: `max_abs_diff < 5e-3`, `has_nan = False`, `hot_hit_rate_mean >= 0.70`, `speedup >= 1.20x`

Results:

| topk | target | onehot max abs diff | has_nan | hot_v2 ms | b3 ms | speedup | hit mean | hit min | pass |
|---:|---|---:|:---:|---:|---:|---:|---:|---:|:---:|
| 48 | `gate_proj` | 0.001953125 | False | 78.869 | 352.222 | 0.224 | 0.174 | 0.040 | No |
| 48 | `up_proj` | 0.001953125 | False | 78.859 | 351.682 | 0.224 | 0.170 | 0.024 | No |
| 64 | `gate_proj` | 0.001953125 | False | 78.876 | 351.827 | 0.224 | 0.222 | 0.040 | No |
| 64 | `up_proj` | 0.001953125 | False | 78.860 | 351.684 | 0.224 | 0.219 | 0.024 | No |

Upper-bound diagnostic:
- Pure stage-local `count` selection gives the practical count-based upper bound for small-k coverage on this slice.
- Measured upper bound:
  - `topk = 64`: `hit_mean ~0.312`, `speedup ~0.224x`
  - `topk = 128`: `hit_mean ~0.578`, `speedup ~0.133x`
  - `topk = 256`: `hit_mean = 1.0`, `speedup ~0.128x`

Interpretation:
- B3 does not rescue M26 stage 2.
- More importantly, the upper-bound diagnostic falsifies the stronger working hypothesis too: with the current encoding on `l54.gate/up`, a small stage-local hot palette (`k <= 64`) cannot reach the target `hot_hit_rate ~0.70-0.75`.
- Therefore the failure is no longer just “wrong kernel” or “wrong hot policy”. The stage vocabulary itself is too diffuse for this exact small-k stage-local palette idea.

Verdict:
- Keep `stage_local_hot_cold_b3` as a documented negative probe.
- Do not run full same-encoding `l54_q_gu` 16w on this variant.
- Next step must either change the representation / runtime surface jointly or retire small-k stage-local hot palettes as an M26 frontier direction on these layers.