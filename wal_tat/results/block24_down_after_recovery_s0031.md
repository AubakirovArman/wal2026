# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001866 | 9.57% | 0.000000 |
<!-- candidate_only: c4_train=0.995221, squad_train=1.001866, vendor_code_train=0.983249 -->
| candidate_bf16_mlp | commit | 1.001822 | 9.57% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995285, squad_train=1.001822, vendor_code_train=0.983443 -->

Fresh verify: pass; c4_train=0.995277, squad_train=1.001796, vendor_code_train=0.983398.
