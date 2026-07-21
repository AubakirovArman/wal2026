# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 0.999832 | 0.59% | 0.000000 |
<!-- candidate_only: c4_train=0.995280, squad_train=0.999832, vendor_code_train=0.980392 -->
| candidate_bf16_mlp | commit | 0.999815 | 0.59% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995259, squad_train=0.999815, vendor_code_train=0.980489 -->

Fresh verify: pass; c4_train=0.995269, squad_train=0.999862, vendor_code_train=0.980491.
