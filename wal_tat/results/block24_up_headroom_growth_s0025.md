# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002102 | 73.14% | 0.000000 |
<!-- candidate_only: c4_train=0.995095, squad_train=1.002102, vendor_code_train=0.981908 -->
| candidate_bf16_mlp | rollback | 1.002141 | 72.36% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995321, squad_train=1.002141, vendor_code_train=0.982646 -->

Fresh verify: pass; c4_train=0.995162, squad_train=1.002064, vendor_code_train=0.981864.
