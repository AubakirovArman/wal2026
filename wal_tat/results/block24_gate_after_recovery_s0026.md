# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001490 | 4.79% | 0.000000 |
<!-- candidate_only: c4_train=0.995052, squad_train=1.001490, vendor_code_train=0.982003 -->
| candidate_bf16_mlp | commit | 1.001570 | 4.79% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995117, squad_train=1.001570, vendor_code_train=0.982518 -->

Fresh verify: pass; c4_train=0.995044, squad_train=1.001480, vendor_code_train=0.982074.
