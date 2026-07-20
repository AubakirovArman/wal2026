# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001667 | 9.96% | 0.000000 |
<!-- candidate_only: c4_train=0.995257, squad_train=1.001667, vendor_code_train=0.983399 -->
| candidate_bf16_mlp | commit | 1.001713 | 9.96% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995312, squad_train=1.001713, vendor_code_train=0.983648 -->

Fresh verify: pass; c4_train=0.995240, squad_train=1.001706, vendor_code_train=0.983400.
