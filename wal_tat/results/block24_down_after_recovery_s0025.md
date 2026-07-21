# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001295 | 8.98% | 0.000000 |
<!-- candidate_only: c4_train=0.994978, squad_train=1.001295, vendor_code_train=0.982146 -->
| candidate_bf16_mlp | commit | 1.001399 | 8.98% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995075, squad_train=1.001399, vendor_code_train=0.982490 -->

Fresh verify: pass; c4_train=0.995008, squad_train=1.001306, vendor_code_train=0.982151.
