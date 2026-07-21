# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001422 | 9.28% | 0.000000 |
<!-- candidate_only: c4_train=0.995011, squad_train=1.001422, vendor_code_train=0.982055 -->
| candidate_bf16_mlp | commit | 1.001523 | 9.28% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995154, squad_train=1.001523, vendor_code_train=0.982746 -->

Fresh verify: pass; c4_train=0.995030, squad_train=1.001415, vendor_code_train=0.982073.
