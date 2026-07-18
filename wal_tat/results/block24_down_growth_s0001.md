# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001658 | 0.49% | 0.000000 |
<!-- candidate_only: c4_train=0.995957, squad_train=1.001658, vendor_code_train=0.982071 -->
| candidate_bf16_mlp | commit | 1.001646 | 0.49% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.996003, squad_train=1.001646, vendor_code_train=0.982051 -->

Fresh verify: pass; c4_train=0.996009, squad_train=1.001647, vendor_code_train=0.982020.
