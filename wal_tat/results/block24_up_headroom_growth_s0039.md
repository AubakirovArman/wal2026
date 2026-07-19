# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002048 | 75.00% | 0.000000 |
<!-- candidate_only: c4_train=0.995242, squad_train=1.002048, vendor_code_train=0.983669 -->
| candidate_bf16_mlp | rollback | 1.002279 | 74.90% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995357, squad_train=1.002279, vendor_code_train=0.984313 -->

Fresh verify: pass; c4_train=0.995247, squad_train=1.002093, vendor_code_train=0.983669.
