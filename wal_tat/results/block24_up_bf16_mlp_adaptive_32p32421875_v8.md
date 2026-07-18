# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 384; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_bf16_mlp | commit | 1.001620 | 32.32% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.996051, squad_train=1.001620, vendor_code_train=0.982141 -->

Fresh verify: pass; c4_train=0.996045, squad_train=1.001624, vendor_code_train=0.982133.
