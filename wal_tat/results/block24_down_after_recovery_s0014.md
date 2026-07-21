# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001584 | 6.15% | 0.000000 |
<!-- candidate_only: c4_train=0.995242, squad_train=1.001584, vendor_code_train=0.982022 -->
| candidate_bf16_mlp | commit | 1.001617 | 6.15% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995325, squad_train=1.001617, vendor_code_train=0.982316 -->

Fresh verify: pass; c4_train=0.995245, squad_train=1.001615, vendor_code_train=0.982037.
