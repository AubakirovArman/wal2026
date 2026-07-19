# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001530 | 4.88% | 0.000000 |
<!-- candidate_only: c4_train=0.995123, squad_train=1.001530, vendor_code_train=0.982843 -->
| candidate_bf16_mlp | commit | 1.001811 | 4.88% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995368, squad_train=1.001811, vendor_code_train=0.983838 -->

Fresh verify: pass; c4_train=0.995150, squad_train=1.001559, vendor_code_train=0.982805.
