# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000572 | 51.27% | 0.000000 |
<!-- candidate_only: c4_train=0.995517, squad_train=1.000572, vendor_code_train=0.980741 -->
| candidate_bf16_mlp | commit | 1.000638 | 51.27% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995598, squad_train=1.000638, vendor_code_train=0.981217 -->

Fresh verify: pass; c4_train=0.995554, squad_train=1.000556, vendor_code_train=0.980755.
