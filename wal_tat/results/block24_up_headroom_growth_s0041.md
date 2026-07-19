# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002114 | 75.20% | 0.000000 |
<!-- candidate_only: c4_train=0.995320, squad_train=1.002114, vendor_code_train=0.983528 -->
| candidate_bf16_mlp | rollback | 1.002324 | 75.10% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995534, squad_train=1.002324, vendor_code_train=0.984811 -->

Fresh verify: pass; c4_train=0.995357, squad_train=1.002145, vendor_code_train=0.983566.
