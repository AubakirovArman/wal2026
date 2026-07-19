# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001265 | 4.69% | 0.000000 |
<!-- candidate_only: c4_train=0.995001, squad_train=1.001265, vendor_code_train=0.982118 -->
| candidate_bf16_mlp | commit | 1.001350 | 4.69% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995099, squad_train=1.001350, vendor_code_train=0.982600 -->

Fresh verify: pass; c4_train=0.994971, squad_train=1.001299, vendor_code_train=0.982157.
