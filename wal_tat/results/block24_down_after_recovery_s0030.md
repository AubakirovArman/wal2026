# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001632 | 9.47% | 0.000000 |
<!-- candidate_only: c4_train=0.995142, squad_train=1.001632, vendor_code_train=0.982960 -->
| candidate_bf16_mlp | commit | 1.001666 | 9.47% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995302, squad_train=1.001666, vendor_code_train=0.983353 -->

Fresh verify: pass; c4_train=0.995151, squad_train=1.001579, vendor_code_train=0.982958.
