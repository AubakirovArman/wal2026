# Experiment Index

Use this file as the short master index for experiment notes.

## Quality / End-to-End Encoding

| Experiment | Model | Key Result | File |
|-----------|-------|-----------|------|
| M39 | Llama 3.1 8B | Hybrid encoder PoC, PPL +13% | `docs/diary/m39_hybrid_encoder_final.md` |
| M40 | Llama 3.1 8B | End-to-end PPL benchmark | `docs/diary/m39_hybrid_encoder_final.md` |
| **M43** | **Llama 3.3 70B** | **Scalar best PPL 4.26 (+78%), VRE catastrophic >7000** | **`docs/diary/m43_70b_end_to_end_encoding.md`** |

## WAL / Runtime

See `docs/diary/m30_path_a_diagnostic.md` through `m37_entropy_regularized.md` for WAL-LHA, WAL-SBC, and WAL-CDA experiments.

## Promoted-to-Paper

- M43 scalar DRL v2 results → target: Section 5 (Quality Evaluation)
- M43 VRE failure analysis → target: Appendix B (Ablation Studies)
