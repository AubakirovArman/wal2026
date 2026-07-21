# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001433 | 74.61% | 0.000000 |
<!-- candidate_only: c4_train=0.995047, squad_train=1.001433, vendor_code_train=0.982056 -->
| candidate_bf16_mlp | commit | 1.001582 | 74.61% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995358, squad_train=1.001582, vendor_code_train=0.982984 -->

Fresh verify: pass; c4_train=0.995030, squad_train=1.001443, vendor_code_train=0.982039.
