# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001292 | 9.08% | 0.000000 |
<!-- candidate_only: c4_train=0.994997, squad_train=1.001292, vendor_code_train=0.982019 -->
| candidate_bf16_mlp | commit | 1.001474 | 9.08% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995086, squad_train=1.001474, vendor_code_train=0.982542 -->

Fresh verify: pass; c4_train=0.995031, squad_train=1.001273, vendor_code_train=0.982068.
