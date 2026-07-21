# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001624 | 4.20% | 0.000000 |
<!-- candidate_only: c4_train=0.995188, squad_train=1.001624, vendor_code_train=0.981400 -->
| candidate_bf16_mlp | commit | 1.001661 | 4.20% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995203, squad_train=1.001661, vendor_code_train=0.981410 -->

Fresh verify: pass; c4_train=0.995200, squad_train=1.001627, vendor_code_train=0.981374.
