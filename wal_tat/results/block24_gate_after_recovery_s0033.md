# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001721 | 5.86% | 0.000000 |
<!-- candidate_only: c4_train=0.995215, squad_train=1.001721, vendor_code_train=0.983392 -->
| candidate_bf16_mlp | commit | 1.001773 | 5.86% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995295, squad_train=1.001773, vendor_code_train=0.983900 -->

Fresh verify: pass; c4_train=0.995276, squad_train=1.001701, vendor_code_train=0.983410.
