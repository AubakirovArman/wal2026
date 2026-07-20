# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002008 | 6.35% | 0.000000 |
<!-- candidate_only: c4_train=0.995405, squad_train=1.002008, vendor_code_train=0.983893 -->
| candidate_bf16_mlp | rollback | 1.002328 | 6.25% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995787, squad_train=1.002328, vendor_code_train=0.984919 -->

Fresh verify: pass; c4_train=0.995421, squad_train=1.002022, vendor_code_train=0.983901.
