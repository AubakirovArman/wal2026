# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001210 | 4.49% | 0.000000 |
<!-- candidate_only: c4_train=0.994992, squad_train=1.001210, vendor_code_train=0.982157 -->
| candidate_bf16_mlp | commit | 1.001337 | 4.49% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995035, squad_train=1.001337, vendor_code_train=0.982574 -->

Fresh verify: pass; c4_train=0.995000, squad_train=1.001239, vendor_code_train=0.982168.
