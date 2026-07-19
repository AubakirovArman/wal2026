# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 1536; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001613 | 63.77% | 0.000000 |
<!-- candidate_only: c4_train=0.994725, squad_train=1.001613, vendor_code_train=0.980456 -->
| candidate_bf16_mlp | commit | 1.001501 | 63.77% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.994873, squad_train=1.001501, vendor_code_train=0.981076 -->

Fresh verify: pass; c4_train=0.994880, squad_train=1.001539, vendor_code_train=0.981014.
