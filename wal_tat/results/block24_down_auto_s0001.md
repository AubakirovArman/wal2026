# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001479 | 0.10% | 0.000000 |
<!-- candidate_only: c4_train=0.996006, squad_train=1.001479, vendor_code_train=0.982132 -->
| candidate_bf16_mlp | commit | 1.001470 | 0.10% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995990, squad_train=1.001470, vendor_code_train=0.982030 -->

Fresh verify: pass; c4_train=0.995985, squad_train=1.001506, vendor_code_train=0.982085.
