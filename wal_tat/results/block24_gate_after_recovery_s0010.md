# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001535 | 2.15% | 0.000000 |
<!-- candidate_only: c4_train=0.994876, squad_train=1.001535, vendor_code_train=0.980973 -->
| candidate_bf16_mlp | commit | 1.001611 | 2.15% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995002, squad_train=1.001611, vendor_code_train=0.981348 -->

Fresh verify: pass; c4_train=0.994918, squad_train=1.001509, vendor_code_train=0.980987.
