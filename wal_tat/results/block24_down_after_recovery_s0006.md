# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001745 | 3.03% | 0.000000 |
<!-- candidate_only: c4_train=0.995289, squad_train=1.001745, vendor_code_train=0.981057 -->
| candidate_bf16_mlp | commit | 1.001892 | 3.03% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995373, squad_train=1.001892, vendor_code_train=0.981336 -->

Fresh verify: pass; c4_train=0.995282, squad_train=1.001681, vendor_code_train=0.981082.
