# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 192; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_bf16_mlp | commit | 1.001493 | 32.52% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.996025, squad_train=1.001493, vendor_code_train=0.982170 -->

Fresh verify: pass; c4_train=0.995992, squad_train=1.001441, vendor_code_train=0.982197.
