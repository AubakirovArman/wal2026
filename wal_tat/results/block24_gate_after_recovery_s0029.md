# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001662 | 5.18% | 0.000000 |
<!-- candidate_only: c4_train=0.995137, squad_train=1.001662, vendor_code_train=0.982849 -->
| candidate_bf16_mlp | commit | 1.001771 | 5.18% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995216, squad_train=1.001771, vendor_code_train=0.983236 -->

Fresh verify: pass; c4_train=0.995156, squad_train=1.001714, vendor_code_train=0.982861.
