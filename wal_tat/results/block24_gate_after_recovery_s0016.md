# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001262 | 3.32% | 0.000000 |
<!-- candidate_only: c4_train=0.995079, squad_train=1.001262, vendor_code_train=0.981579 -->
| candidate_bf16_mlp | commit | 1.001321 | 3.32% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995111, squad_train=1.001321, vendor_code_train=0.982003 -->

Fresh verify: pass; c4_train=0.995050, squad_train=1.001259, vendor_code_train=0.981632.
