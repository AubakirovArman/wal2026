# Paper Claim Index

This file maps experiment notes to paper claims.

## Positive Claims

| Experiment | Artifact | Headline Numbers | Target Section | Status |
|-----------|----------|-----------------|---------------|--------|
| M43i Scalar 70B | `experiments/m43i_scalar_only.py` | PPL 4.29, Δ+79% vs dense 2.40 | §5.1 Quality Gate | Positive |
| M43zj Scalar skip L0 | `experiments/m43i_scalar_only.py` | PPL 4.26, Δ+78% | §5.1 Quality Gate | Positive |
| M43zd Late layers only | `experiments/m43i_scalar_only.py` | PPL 2.80 (layers 60-79 only) | §5.2 Layer Sensitivity | Positive |

## Negative Claims (Important for Honesty)

| Experiment | Artifact | Headline Numbers | Target Section | Status |
|-----------|----------|-----------------|---------------|--------|
| M43k VRE multi-layer | `experiments/m39_hybrid_encoder.py` | PPL 7799.62 (catastrophic) | Appendix B.3 | Negative |
| M43zk VRE selective L0 | `experiments/m43zk_vre_layer0_selective.py` | PPL 30.86 (selective VRE) | Appendix B.3 | Negative |
| M43ze Early spiky only | `experiments/m43i_scalar_only.py` | PPL 421.61 (early q/k/v/gate/up) | §5.2 Layer Sensitivity | Negative |
| M43y K=256 collapse | `experiments/m43i_scalar_only.py` | PPL 359.68 (Lloyd-Max collapse) | Appendix B.2 | Negative |

## Key Insight Claims

1. **Error structure > sign accuracy**: Scalar changes 92.5% signs but PPL ~2.40; VRE changes 21% signs but PPL 30.86 → §3.2 or §5.3
2. **Block-correlated errors are toxic to attention**: VRE relMSE 0.001 but catastrophic PPL → §3.2 or Appendix B.3
3. **Early attention projections are the bottleneck**: Early q/k/v/gate/up dominate quality loss → §5.2

## Scaffold-Only (Not Yet Claim-Worthy)

- M43zh Adaptive hybrid: PPL 2019.75 — confirms VRE+scalar mixing is non-viable
- M43zc lmax=12: PPL 5.67 — confirms more ladder steps ≠ better quality
