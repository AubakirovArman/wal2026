# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001571 | 70.80% | 0.000000 |
<!-- candidate_only: c4_train=0.995065, squad_train=1.001571, vendor_code_train=0.981562 -->
| candidate_bf16_mlp | commit | 1.001677 | 70.80% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995398, squad_train=1.001677, vendor_code_train=0.982730 -->

Fresh verify: pass; c4_train=0.995084, squad_train=1.001560, vendor_code_train=0.981606.
