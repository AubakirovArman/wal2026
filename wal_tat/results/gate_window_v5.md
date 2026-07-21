# WAL-TAT atomic compensation window

Candidate: `model.layers.27.mlp.gate_proj`; new groups: 6144; linked down groups: 16384.

| Arm | Gate | Wiki ratio | Code ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|---:|
| candidate_only | commit | 0.979370 | 1.019315 | 75.00% | 0.000000 |
| linked_down | commit | 0.978557 | 1.018246 | 75.00% | 0.000000 |

Fresh verify: pass; Wiki/code ratios 0.978552/1.018250.
