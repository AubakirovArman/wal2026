# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001937 | 4.39% | 0.000000 |
<!-- candidate_only: c4_train=0.995417, squad_train=1.001937, vendor_code_train=0.983282 -->
| candidate_bf16_mlp | rollback | 1.002244 | 4.30% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995834, squad_train=1.002244, vendor_code_train=0.984459 -->

Fresh verify: pass; c4_train=0.995417, squad_train=1.001945, vendor_code_train=0.983276.
