# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001211 | 74.32% | 0.000000 |
<!-- candidate_only: c4_train=0.994952, squad_train=1.001211, vendor_code_train=0.982186 -->
| candidate_bf16_mlp | commit | 1.001317 | 74.32% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995070, squad_train=1.001317, vendor_code_train=0.983302 -->

Fresh verify: pass; c4_train=0.994952, squad_train=1.001211, vendor_code_train=0.982187.
