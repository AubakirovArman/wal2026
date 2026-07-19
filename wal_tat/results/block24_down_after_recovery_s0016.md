# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001302 | 6.93% | 0.000000 |
<!-- candidate_only: c4_train=0.995056, squad_train=1.001302, vendor_code_train=0.981624 -->
| candidate_bf16_mlp | commit | 1.001380 | 6.93% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995220, squad_train=1.001380, vendor_code_train=0.982214 -->

Fresh verify: pass; c4_train=0.995070, squad_train=1.001318, vendor_code_train=0.981635.
