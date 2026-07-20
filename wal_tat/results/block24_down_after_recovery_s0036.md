# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001870 | 10.06% | 0.000000 |
<!-- candidate_only: c4_train=0.995313, squad_train=1.001870, vendor_code_train=0.983511 -->
| candidate_bf16_mlp | commit | 1.001870 | 10.06% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995275, squad_train=1.001870, vendor_code_train=0.983610 -->

Fresh verify: pass; c4_train=0.995273, squad_train=1.001867, vendor_code_train=0.983475.
