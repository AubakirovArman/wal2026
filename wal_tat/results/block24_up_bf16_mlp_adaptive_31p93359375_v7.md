# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_bf16_mlp | commit | 1.001563 | 31.93% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.996025, squad_train=1.001563, vendor_code_train=0.982150 -->

Fresh verify: pass; c4_train=0.996063, squad_train=1.001582, vendor_code_train=0.982166.
