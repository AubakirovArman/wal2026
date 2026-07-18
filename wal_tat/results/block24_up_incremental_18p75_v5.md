# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 4096.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001506 | 18.75% | 0.000000 |
<!-- candidate_only: c4_train=0.995828, squad_train=1.001506, vendor_code_train=0.981445 -->

Fresh verify: pass; c4_train=0.995837, squad_train=1.001468, vendor_code_train=0.981447.
