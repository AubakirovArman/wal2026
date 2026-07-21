# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001834 | 6.15% | 0.000000 |
<!-- candidate_only: c4_train=0.995315, squad_train=1.001834, vendor_code_train=0.983615 -->
| candidate_bf16_mlp | commit | 1.001991 | 6.15% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995375, squad_train=1.001991, vendor_code_train=0.983808 -->

Fresh verify: pass; c4_train=0.995306, squad_train=1.001838, vendor_code_train=0.983605.
