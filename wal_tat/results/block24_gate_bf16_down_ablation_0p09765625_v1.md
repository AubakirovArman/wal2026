# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001496 | 0.10% | 0.000000 |
<!-- candidate_only: c4_train=0.996002, squad_train=1.001496, vendor_code_train=0.982106 -->
| candidate_bf16_mlp | commit | 1.001530 | 0.10% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.996012, squad_train=1.001530, vendor_code_train=0.982126 -->

Fresh verify: pass; c4_train=0.995998, squad_train=1.001471, vendor_code_train=0.982080.
