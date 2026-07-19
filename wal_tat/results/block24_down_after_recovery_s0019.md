# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001867 | 8.11% | 0.000000 |
<!-- candidate_only: c4_train=0.995148, squad_train=1.001867, vendor_code_train=0.981865 -->
| candidate_bf16_mlp | commit | 1.001891 | 8.11% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995229, squad_train=1.001891, vendor_code_train=0.981975 -->

Fresh verify: pass; c4_train=0.995174, squad_train=1.001956, vendor_code_train=0.981864.
