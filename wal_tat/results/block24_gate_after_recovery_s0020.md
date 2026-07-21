# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002042 | 4.10% | 0.000000 |
<!-- candidate_only: c4_train=0.995285, squad_train=1.002042, vendor_code_train=0.982217 -->
| candidate_bf16_mlp | commit | 1.002119 | 4.10% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995335, squad_train=1.002119, vendor_code_train=0.982716 -->

Fresh verify: pass; c4_train=0.995234, squad_train=1.002035, vendor_code_train=0.982205.
