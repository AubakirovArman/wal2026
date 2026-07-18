# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 4096.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001202 | 15.62% | 0.000000 |
<!-- candidate_only: c4_train=0.995668, squad_train=1.001202, vendor_code_train=0.981231 -->

Fresh verify: pass; c4_train=0.995710, squad_train=1.001145, vendor_code_train=0.981249.
