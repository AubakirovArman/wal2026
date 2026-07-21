# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001735 | 75.29% | 0.000000 |
<!-- candidate_only: c4_train=0.995253, squad_train=1.001735, vendor_code_train=0.983359 -->
| candidate_bf16_mlp | commit | 1.002040 | 75.29% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995490, squad_train=1.002040, vendor_code_train=0.984824 -->

Fresh verify: pass; c4_train=0.995242, squad_train=1.001726, vendor_code_train=0.983345.
