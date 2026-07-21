# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001206 | 28.12% | 0.000000 |
<!-- candidate_only: c4_train=0.996041, squad_train=1.001206, vendor_code_train=0.981931 -->

Fresh verify: pass; c4_train=0.996072, squad_train=1.001247, vendor_code_train=0.981950.
