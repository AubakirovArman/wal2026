# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001363 | 4.98% | 0.000000 |
<!-- candidate_only: c4_train=0.994976, squad_train=1.001363, vendor_code_train=0.981110 -->
| candidate_bf16_mlp | commit | 1.001339 | 4.98% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995080, squad_train=1.001339, vendor_code_train=0.981515 -->

Fresh verify: pass; c4_train=0.995037, squad_train=1.001315, vendor_code_train=0.981461.
