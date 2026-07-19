# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001371 | 2.54% | 0.000000 |
<!-- candidate_only: c4_train=0.995031, squad_train=1.001371, vendor_code_train=0.981120 -->
| candidate_bf16_mlp | commit | 1.001510 | 2.54% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995423, squad_train=1.001510, vendor_code_train=0.982061 -->

Fresh verify: pass; c4_train=0.995028, squad_train=1.001410, vendor_code_train=0.981109.
