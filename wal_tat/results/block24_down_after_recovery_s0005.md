# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000079 | 2.64% | 0.000000 |
<!-- candidate_only: c4_train=0.995107, squad_train=1.000079, vendor_code_train=0.980506 -->
| candidate_bf16_mlp | commit | 1.000130 | 2.64% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995121, squad_train=1.000130, vendor_code_train=0.980753 -->

Fresh verify: pass; c4_train=0.995147, squad_train=1.000075, vendor_code_train=0.980539.
