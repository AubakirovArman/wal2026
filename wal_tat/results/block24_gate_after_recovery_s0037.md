# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002062 | 6.25% | 0.000000 |
<!-- candidate_only: c4_train=0.995404, squad_train=1.002062, vendor_code_train=0.983902 -->
| candidate_bf16_mlp | rollback | 1.002189 | 6.15% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995488, squad_train=1.002189, vendor_code_train=0.984401 -->

Fresh verify: pass; c4_train=0.995412, squad_train=1.001997, vendor_code_train=0.983871.
