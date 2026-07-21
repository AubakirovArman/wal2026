# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000640 | 35.64% | 0.000000 |
<!-- candidate_only: c4_train=0.995361, squad_train=1.000640, vendor_code_train=0.980931 -->
| candidate_bf16_mlp | commit | 1.000595 | 35.64% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995412, squad_train=1.000595, vendor_code_train=0.981294 -->

Fresh verify: pass; c4_train=0.995395, squad_train=1.000615, vendor_code_train=0.981253.
