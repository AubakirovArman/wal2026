# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002056 | 75.59% | 0.000000 |
<!-- candidate_only: c4_train=0.995376, squad_train=1.002056, vendor_code_train=0.983843 -->
| candidate_bf16_mlp | commit | 1.002073 | 75.59% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995474, squad_train=1.002073, vendor_code_train=0.984671 -->

Fresh verify: pass; c4_train=0.995380, squad_train=1.002039, vendor_code_train=0.983864.
