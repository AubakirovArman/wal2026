# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001742 | 75.39% | 0.000000 |
<!-- candidate_only: c4_train=0.995254, squad_train=1.001742, vendor_code_train=0.983502 -->
| candidate_bf16_mlp | commit | 1.001980 | 75.39% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995556, squad_train=1.001980, vendor_code_train=0.984275 -->

Fresh verify: pass; c4_train=0.995267, squad_train=1.001754, vendor_code_train=0.983538.
