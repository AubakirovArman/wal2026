# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000275 | 48.14% | 0.000000 |
<!-- candidate_only: c4_train=0.995272, squad_train=1.000275, vendor_code_train=0.980450 -->
| candidate_bf16_mlp | commit | 1.000133 | 48.14% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995358, squad_train=1.000133, vendor_code_train=0.980743 -->

Fresh verify: pass; c4_train=0.995367, squad_train=1.000130, vendor_code_train=0.980794.
