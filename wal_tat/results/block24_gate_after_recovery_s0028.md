# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001480 | 4.98% | 0.000000 |
<!-- candidate_only: c4_train=0.995161, squad_train=1.001480, vendor_code_train=0.982808 -->
| candidate_bf16_mlp | commit | 1.001852 | 4.98% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995554, squad_train=1.001852, vendor_code_train=0.984282 -->

Fresh verify: pass; c4_train=0.995153, squad_train=1.001510, vendor_code_train=0.982786.
