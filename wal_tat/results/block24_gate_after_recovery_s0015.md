# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001853 | 3.12% | 0.000000 |
<!-- candidate_only: c4_train=0.995498, squad_train=1.001853, vendor_code_train=0.982670 -->
| candidate_bf16_mlp | commit | 1.001948 | 3.12% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995543, squad_train=1.001948, vendor_code_train=0.982893 -->

Fresh verify: pass; c4_train=0.995550, squad_train=1.001874, vendor_code_train=0.982652.
