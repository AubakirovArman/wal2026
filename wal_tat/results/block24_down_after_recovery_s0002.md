# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 0.999848 | 1.46% | 0.000000 |
<!-- candidate_only: c4_train=0.995264, squad_train=0.999848, vendor_code_train=0.980392 -->
| candidate_bf16_mlp | commit | 0.999821 | 1.46% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995268, squad_train=0.999821, vendor_code_train=0.980378 -->

Fresh verify: pass; c4_train=0.995264, squad_train=0.999806, vendor_code_train=0.980336.
