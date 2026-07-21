# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_bf16_mlp | commit | 1.001527 | 29.59% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.996106, squad_train=1.001527, vendor_code_train=0.982009 -->

Fresh verify: pass; c4_train=0.996052, squad_train=1.001501, vendor_code_train=0.982061.
