# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000777 | 0.98% | 0.000000 |
<!-- candidate_only: c4_train=0.995559, squad_train=1.000777, vendor_code_train=0.981184 -->
| candidate_bf16_mlp | commit | 1.000766 | 0.98% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995668, squad_train=1.000766, vendor_code_train=0.981611 -->

Fresh verify: pass; c4_train=0.995675, squad_train=1.000809, vendor_code_train=0.981553.
