# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001724 | 7.32% | 0.000000 |
<!-- candidate_only: c4_train=0.995053, squad_train=1.001724, vendor_code_train=0.981721 -->
| candidate_bf16_mlp | commit | 1.001674 | 7.32% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995015, squad_train=1.001674, vendor_code_train=0.981788 -->

Fresh verify: pass; c4_train=0.995053, squad_train=1.001682, vendor_code_train=0.981712.
