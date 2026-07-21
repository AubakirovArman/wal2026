# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001355 | 1.76% | 0.000000 |
<!-- candidate_only: c4_train=0.995013, squad_train=1.001355, vendor_code_train=0.981343 -->
| candidate_bf16_mlp | commit | 1.001398 | 1.76% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995140, squad_train=1.001398, vendor_code_train=0.981673 -->

Fresh verify: pass; c4_train=0.995016, squad_train=1.001333, vendor_code_train=0.981355.
