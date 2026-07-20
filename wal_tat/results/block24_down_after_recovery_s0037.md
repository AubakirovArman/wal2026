# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001840 | 10.16% | 0.000000 |
<!-- candidate_only: c4_train=0.995245, squad_train=1.001840, vendor_code_train=0.983472 -->
| candidate_bf16_mlp | commit | 1.001889 | 10.16% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995309, squad_train=1.001889, vendor_code_train=0.983589 -->

Fresh verify: pass; c4_train=0.995260, squad_train=1.001839, vendor_code_train=0.983465.
