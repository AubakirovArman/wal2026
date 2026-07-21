# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001909 | 1.37% | 0.000000 |
<!-- candidate_only: c4_train=0.995375, squad_train=1.001909, vendor_code_train=0.981336 -->
| candidate_bf16_mlp | commit | 1.001976 | 1.37% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995404, squad_train=1.001976, vendor_code_train=0.981544 -->

Fresh verify: pass; c4_train=0.995355, squad_train=1.001844, vendor_code_train=0.981347.
