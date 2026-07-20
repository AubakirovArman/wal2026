# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002001 | 75.68% | 0.000000 |
<!-- candidate_only: c4_train=0.995424, squad_train=1.002001, vendor_code_train=0.983893 -->
| candidate_bf16_mlp | commit | 1.002071 | 75.68% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995497, squad_train=1.002071, vendor_code_train=0.984302 -->

Fresh verify: pass; c4_train=0.995429, squad_train=1.001991, vendor_code_train=0.983831.
