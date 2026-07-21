# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001610 | 1.95% | 0.000000 |
<!-- candidate_only: c4_train=0.995225, squad_train=1.001610, vendor_code_train=0.981383 -->
| candidate_bf16_mlp | commit | 1.001657 | 1.95% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995294, squad_train=1.001657, vendor_code_train=0.981679 -->

Fresh verify: pass; c4_train=0.995259, squad_train=1.001615, vendor_code_train=0.981400.
