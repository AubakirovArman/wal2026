# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 192; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001239 | 28.71% | 0.000000 |
<!-- candidate_only: c4_train=0.996084, squad_train=1.001239, vendor_code_train=0.981979 -->

Fresh verify: pass; c4_train=0.996102, squad_train=1.001253, vendor_code_train=0.981946.
