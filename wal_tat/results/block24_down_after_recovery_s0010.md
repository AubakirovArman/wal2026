# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001545 | 4.59% | 0.000000 |
<!-- candidate_only: c4_train=0.994906, squad_train=1.001545, vendor_code_train=0.980981 -->
| candidate_bf16_mlp | commit | 1.001633 | 4.59% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.994922, squad_train=1.001633, vendor_code_train=0.981059 -->

Fresh verify: pass; c4_train=0.994884, squad_train=1.001526, vendor_code_train=0.980949.
