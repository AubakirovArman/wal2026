# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 6144; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | rollback | 1.001998 | 32.52% | 0.000000 |
<!-- candidate_only: c4_train=0.995530, squad_train=1.001998, vendor_code_train=0.981004 -->
| candidate_bf16_mlp | rollback | 1.001978 | 32.52% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995537, squad_train=1.001978, vendor_code_train=0.980906 -->
