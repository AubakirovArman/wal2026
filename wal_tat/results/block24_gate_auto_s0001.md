# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001492 | 0.20% | 0.000000 |
<!-- candidate_only: c4_train=0.996001, squad_train=1.001492, vendor_code_train=0.982101 -->
| candidate_bf16_mlp | commit | 1.001491 | 0.20% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995990, squad_train=1.001491, vendor_code_train=0.982134 -->

Fresh verify: pass; c4_train=0.995991, squad_train=1.001564, vendor_code_train=0.982146.
