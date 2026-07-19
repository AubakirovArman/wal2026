# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001222 | 8.89% | 0.000000 |
<!-- candidate_only: c4_train=0.994989, squad_train=1.001222, vendor_code_train=0.982116 -->
| candidate_bf16_mlp | commit | 1.001222 | 8.89% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995106, squad_train=1.001222, vendor_code_train=0.982544 -->

Fresh verify: pass; c4_train=0.995008, squad_train=1.001186, vendor_code_train=0.982105.
