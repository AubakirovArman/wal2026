# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001867 | 6.54% | 0.000000 |
<!-- candidate_only: c4_train=0.995546, squad_train=1.001867, vendor_code_train=0.982753 -->
| candidate_bf16_mlp | commit | 1.001852 | 6.54% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995512, squad_train=1.001852, vendor_code_train=0.983118 -->

Fresh verify: pass; c4_train=0.995522, squad_train=1.001872, vendor_code_train=0.983050.
