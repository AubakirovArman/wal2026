# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001564 | 2.73% | 0.000000 |
<!-- candidate_only: c4_train=0.995234, squad_train=1.001564, vendor_code_train=0.981899 -->
| candidate_bf16_mlp | commit | 1.001522 | 2.73% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995294, squad_train=1.001522, vendor_code_train=0.982180 -->

Fresh verify: pass; c4_train=0.995206, squad_train=1.001496, vendor_code_train=0.981926.
