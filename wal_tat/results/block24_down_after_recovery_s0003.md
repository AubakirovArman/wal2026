# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000165 | 1.86% | 0.000000 |
<!-- candidate_only: c4_train=0.995318, squad_train=1.000165, vendor_code_train=0.980656 -->
| candidate_bf16_mlp | commit | 1.000172 | 1.86% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995347, squad_train=1.000172, vendor_code_train=0.980604 -->

Fresh verify: pass; c4_train=0.995354, squad_train=1.000161, vendor_code_train=0.980660.
