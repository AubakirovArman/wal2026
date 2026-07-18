# WAL-TAT atomic compensation window

Candidate: `model.layers.27.mlp.gate_proj`; new groups: 3072; linked down groups: 8192.

| Arm | Gate | Wiki ratio | Code ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|---:|
| linked_down | rollback | 0.980358 | 1.020076 | 78.12% | 0.000000 |
| linked_mlp | commit | 0.979999 | 1.019665 | 81.25% | 0.000000 |

Fresh verify: pass; Wiki/code ratios 0.979970/1.019674.
