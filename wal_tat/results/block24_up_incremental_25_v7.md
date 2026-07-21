# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 4096.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001620 | 25.00% | 0.000000 |
<!-- candidate_only: c4_train=0.995969, squad_train=1.001620, vendor_code_train=0.981748 -->

Fresh verify: pass; c4_train=0.995987, squad_train=1.001559, vendor_code_train=0.981764.
