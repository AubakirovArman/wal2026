# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001202 | 74.22% | 0.000000 |
<!-- candidate_only: c4_train=0.994956, squad_train=1.001202, vendor_code_train=0.982186 -->
| candidate_bf16_mlp | commit | 1.001476 | 74.22% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995350, squad_train=1.001476, vendor_code_train=0.983420 -->

Fresh verify: pass; c4_train=0.994950, squad_train=1.001208, vendor_code_train=0.982162.
