# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001940 | 74.12% | 0.000000 |
<!-- candidate_only: c4_train=0.995277, squad_train=1.001940, vendor_code_train=0.982576 -->
| candidate_bf16_mlp | commit | 1.002100 | 74.12% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995342, squad_train=1.002100, vendor_code_train=0.982633 -->

Fresh verify: pass; c4_train=0.995289, squad_train=1.001933, vendor_code_train=0.982565.
