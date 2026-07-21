# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 384; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001962 | 73.54% | 0.000000 |
<!-- candidate_only: c4_train=0.995214, squad_train=1.001962, vendor_code_train=0.982253 -->
| candidate_bf16_mlp | commit | 1.002089 | 73.54% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995240, squad_train=1.002089, vendor_code_train=0.982750 -->

Fresh verify: pass; c4_train=0.995218, squad_train=1.002008, vendor_code_train=0.982261.
