# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001346 | 27.34% | 0.000000 |
<!-- candidate_only: c4_train=0.996020, squad_train=1.001346, vendor_code_train=0.981950 -->

Fresh verify: pass; c4_train=0.996006, squad_train=1.001371, vendor_code_train=0.981986.
