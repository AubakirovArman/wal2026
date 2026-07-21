# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001271 | 3.81% | 0.000000 |
<!-- candidate_only: c4_train=0.995016, squad_train=1.001271, vendor_code_train=0.981243 -->
| candidate_bf16_mlp | commit | 1.001326 | 3.81% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995064, squad_train=1.001326, vendor_code_train=0.981323 -->

Fresh verify: pass; c4_train=0.995050, squad_train=1.001257, vendor_code_train=0.981231.
