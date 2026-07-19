# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001542 | 74.71% | 0.000000 |
<!-- candidate_only: c4_train=0.995127, squad_train=1.001542, vendor_code_train=0.982806 -->
| candidate_bf16_mlp | commit | 1.001530 | 74.71% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995167, squad_train=1.001530, vendor_code_train=0.983265 -->

Fresh verify: pass; c4_train=0.995166, squad_train=1.001538, vendor_code_train=0.983304.
