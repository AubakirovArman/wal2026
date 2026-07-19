# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001244 | 4.59% | 0.000000 |
<!-- candidate_only: c4_train=0.994973, squad_train=1.001244, vendor_code_train=0.982160 -->
| candidate_bf16_mlp | commit | 1.001453 | 4.59% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995090, squad_train=1.001453, vendor_code_train=0.982693 -->

Fresh verify: pass; c4_train=0.994997, squad_train=1.001313, vendor_code_train=0.982176.
