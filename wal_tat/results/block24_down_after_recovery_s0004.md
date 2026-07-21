# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000773 | 2.25% | 0.000000 |
<!-- candidate_only: c4_train=0.995620, squad_train=1.000773, vendor_code_train=0.981314 -->
| candidate_bf16_mlp | commit | 1.000817 | 2.25% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995605, squad_train=1.000817, vendor_code_train=0.981222 -->

Fresh verify: pass; c4_train=0.995612, squad_train=1.000842, vendor_code_train=0.981316.
