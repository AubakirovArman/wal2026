# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001599 | 57.52% | 0.000000 |
<!-- candidate_only: c4_train=0.995203, squad_train=1.001599, vendor_code_train=0.980916 -->
| candidate_bf16_mlp | commit | 1.001694 | 57.52% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995221, squad_train=1.001694, vendor_code_train=0.981049 -->

Fresh verify: pass; c4_train=0.995229, squad_train=1.001627, vendor_code_train=0.980892.
