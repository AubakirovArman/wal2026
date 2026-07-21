# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 384; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001212 | 28.52% | 0.000000 |
<!-- candidate_only: c4_train=0.996100, squad_train=1.001212, vendor_code_train=0.981959 -->

Fresh verify: pass; c4_train=0.996104, squad_train=1.001240, vendor_code_train=0.981951.
