# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001856 | 6.05% | 0.000000 |
<!-- candidate_only: c4_train=0.995290, squad_train=1.001856, vendor_code_train=0.983493 -->
| candidate_bf16_mlp | commit | 1.002017 | 6.05% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995516, squad_train=1.002017, vendor_code_train=0.984142 -->

Fresh verify: pass; c4_train=0.995259, squad_train=1.001847, vendor_code_train=0.983431.
