# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 4096.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001474 | 21.88% | 0.000000 |
<!-- candidate_only: c4_train=0.995862, squad_train=1.001474, vendor_code_train=0.981613 -->

Fresh verify: pass; c4_train=0.995826, squad_train=1.001559, vendor_code_train=0.981616.
