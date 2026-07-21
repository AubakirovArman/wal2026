# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001761 | 3.71% | 0.000000 |
<!-- candidate_only: c4_train=0.995092, squad_train=1.001761, vendor_code_train=0.981830 -->
| candidate_bf16_mlp | commit | 1.001825 | 3.71% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995158, squad_train=1.001825, vendor_code_train=0.982409 -->

Fresh verify: pass; c4_train=0.995118, squad_train=1.001676, vendor_code_train=0.981872.
