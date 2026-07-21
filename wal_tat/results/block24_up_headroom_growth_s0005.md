# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 0.999954 | 45.02% | 0.000000 |
<!-- candidate_only: c4_train=0.995153, squad_train=0.999954, vendor_code_train=0.980169 -->
| candidate_bf16_mlp | commit | 0.999669 | 45.02% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995227, squad_train=0.999669, vendor_code_train=0.980310 -->

Fresh verify: pass; c4_train=0.995232, squad_train=0.999678, vendor_code_train=0.980374.
