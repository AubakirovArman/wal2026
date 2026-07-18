# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001575 | 28.81% | 0.000000 |
<!-- candidate_only: c4_train=0.996051, squad_train=1.001575, vendor_code_train=0.981933 -->
| candidate_bf16_mlp | commit | 1.001502 | 28.81% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.996039, squad_train=1.001502, vendor_code_train=0.981936 -->

Fresh verify: pass; c4_train=0.996098, squad_train=1.001541, vendor_code_train=0.981931.
