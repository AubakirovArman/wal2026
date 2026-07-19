# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002026 | 9.67% | 0.000000 |
<!-- candidate_only: c4_train=0.995253, squad_train=1.002026, vendor_code_train=0.983678 -->
| candidate_bf16_mlp | commit | 1.001997 | 9.67% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995274, squad_train=1.001997, vendor_code_train=0.984012 -->

Fresh verify: pass; c4_train=0.995274, squad_train=1.002003, vendor_code_train=0.983960.
