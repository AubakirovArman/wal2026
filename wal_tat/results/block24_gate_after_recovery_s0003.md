# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000230 | 0.78% | 0.000000 |
<!-- candidate_only: c4_train=0.995329, squad_train=1.000230, vendor_code_train=0.980645 -->
| candidate_bf16_mlp | commit | 1.000275 | 0.78% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995326, squad_train=1.000275, vendor_code_train=0.980715 -->

Fresh verify: pass; c4_train=0.995348, squad_train=1.000280, vendor_code_train=0.980655.
