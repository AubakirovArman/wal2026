# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001509 | 9.18% | 0.000000 |
<!-- candidate_only: c4_train=0.995006, squad_train=1.001509, vendor_code_train=0.981974 -->
| candidate_bf16_mlp | commit | 1.001738 | 9.18% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995145, squad_train=1.001738, vendor_code_train=0.983249 -->

Fresh verify: pass; c4_train=0.995001, squad_train=1.001475, vendor_code_train=0.982052.
