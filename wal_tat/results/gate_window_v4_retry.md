# WAL-TAT atomic compensation window

Candidate: `model.layers.27.mlp.gate_proj`; new groups: 6144; linked down groups: 16384.

| Arm | Gate | Wiki ratio | Code ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|---:|
| candidate_only | commit | 0.979347 | 1.018982 | 68.75% | 0.000000 |
| linked_down | commit | 0.978712 | 1.018227 | 68.75% | 0.000001 |

Fresh verify: pass; Wiki/code ratios 0.978716/1.018237.
