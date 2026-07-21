# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001843 | 72.36% | 0.000000 |
<!-- candidate_only: c4_train=0.995041, squad_train=1.001843, vendor_code_train=0.982015 -->
| candidate_bf16_mlp | commit | 1.001927 | 72.36% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995207, squad_train=1.001927, vendor_code_train=0.982724 -->

Fresh verify: pass; c4_train=0.995061, squad_train=1.001842, vendor_code_train=0.982044.
