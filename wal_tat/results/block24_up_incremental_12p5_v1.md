# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 12288; linked down groups: 12288.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001127 | 12.50% | 0.000000 |
<!-- candidate_only: c4_train=0.995609, squad_train=1.001127, vendor_code_train=0.980903 -->

Fresh verify: pass; c4_train=0.995592, squad_train=1.001093, vendor_code_train=0.980935.
