# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001337 | 2.34% | 0.000000 |
<!-- candidate_only: c4_train=0.994988, squad_train=1.001337, vendor_code_train=0.981011 -->
| candidate_bf16_mlp | commit | 1.001495 | 2.34% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995043, squad_train=1.001495, vendor_code_train=0.981400 -->

Fresh verify: pass; c4_train=0.994996, squad_train=1.001351, vendor_code_train=0.981110.
