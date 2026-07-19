# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001762 | 5.37% | 0.000000 |
<!-- candidate_only: c4_train=0.995260, squad_train=1.001762, vendor_code_train=0.983280 -->
| candidate_bf16_mlp | commit | 1.001893 | 5.37% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995255, squad_train=1.001893, vendor_code_train=0.983626 -->

Fresh verify: pass; c4_train=0.995254, squad_train=1.001736, vendor_code_train=0.983277.
