# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001605 | 74.80% | 0.000000 |
<!-- candidate_only: c4_train=0.995150, squad_train=1.001605, vendor_code_train=0.982861 -->
| candidate_bf16_mlp | commit | 1.001656 | 74.80% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995344, squad_train=1.001656, vendor_code_train=0.984494 -->

Fresh verify: pass; c4_train=0.995157, squad_train=1.001634, vendor_code_train=0.982888.
