# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000012 | 54.39% | 0.000000 |
<!-- candidate_only: c4_train=0.995087, squad_train=1.000012, vendor_code_train=0.980530 -->
| candidate_bf16_mlp | commit | 1.000009 | 54.39% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995182, squad_train=1.000009, vendor_code_train=0.980986 -->

Fresh verify: pass; c4_train=0.995181, squad_train=1.000040, vendor_code_train=0.980947.
