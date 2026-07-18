# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 1536; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001456 | 26.56% | 0.000000 |
<!-- candidate_only: c4_train=0.996078, squad_train=1.001456, vendor_code_train=0.981848 -->

Fresh verify: pass; c4_train=0.996041, squad_train=1.001475, vendor_code_train=0.981844.
