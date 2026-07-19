# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 1536; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001629 | 62.21% | 0.000000 |
<!-- candidate_only: c4_train=0.995159, squad_train=1.001629, vendor_code_train=0.981326 -->
| candidate_bf16_mlp | commit | 1.001588 | 62.21% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995180, squad_train=1.001588, vendor_code_train=0.981742 -->

Fresh verify: pass; c4_train=0.995198, squad_train=1.001588, vendor_code_train=0.981699.
