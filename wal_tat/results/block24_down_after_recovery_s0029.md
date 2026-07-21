# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001494 | 9.38% | 0.000000 |
<!-- candidate_only: c4_train=0.995196, squad_train=1.001494, vendor_code_train=0.982814 -->
| candidate_bf16_mlp | commit | 1.001578 | 9.38% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995199, squad_train=1.001578, vendor_code_train=0.983319 -->

Fresh verify: pass; c4_train=0.995187, squad_train=1.001516, vendor_code_train=0.982846.
