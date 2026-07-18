# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 192; linked down groups: 192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001616 | 0.68% | 0.000000 |
<!-- candidate_only: c4_train=0.996011, squad_train=1.001616, vendor_code_train=0.982053 -->
| candidate_bf16_mlp | commit | 1.001674 | 0.68% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.996013, squad_train=1.001674, vendor_code_train=0.982030 -->

Fresh verify: pass; c4_train=0.996033, squad_train=1.001614, vendor_code_train=0.982065.
