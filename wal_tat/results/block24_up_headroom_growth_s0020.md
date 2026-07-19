# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001857 | 69.24% | 0.000000 |
<!-- candidate_only: c4_train=0.995525, squad_train=1.001857, vendor_code_train=0.982663 -->
| candidate_bf16_mlp | commit | 1.002016 | 69.24% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995664, squad_train=1.002016, vendor_code_train=0.983985 -->

Fresh verify: pass; c4_train=0.995455, squad_train=1.001829, vendor_code_train=0.982711.
