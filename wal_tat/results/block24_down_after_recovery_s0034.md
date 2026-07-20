# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001859 | 9.86% | 0.000000 |
<!-- candidate_only: c4_train=0.995143, squad_train=1.001859, vendor_code_train=0.983107 -->
| candidate_bf16_mlp | commit | 1.001715 | 9.86% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995260, squad_train=1.001715, vendor_code_train=0.983401 -->

Fresh verify: pass; c4_train=0.995260, squad_train=1.001780, vendor_code_train=0.983346.
