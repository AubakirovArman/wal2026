# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001838 | 75.49% | 0.000000 |
<!-- candidate_only: c4_train=0.995265, squad_train=1.001838, vendor_code_train=0.983454 -->
| candidate_bf16_mlp | rollback | 1.002262 | 75.39% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995678, squad_train=1.002262, vendor_code_train=0.984899 -->

Fresh verify: pass; c4_train=0.995277, squad_train=1.001821, vendor_code_train=0.983482.
