# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001765 | 7.71% | 0.000000 |
<!-- candidate_only: c4_train=0.995102, squad_train=1.001765, vendor_code_train=0.981872 -->
| candidate_bf16_mlp | commit | 1.001855 | 7.71% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995165, squad_train=1.001855, vendor_code_train=0.982324 -->

Fresh verify: pass; c4_train=0.995107, squad_train=1.001739, vendor_code_train=0.981891.
