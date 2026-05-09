"""Quantile-based ladder seed.

Seed ladder by choosing scales that split the distribution of |w_norm|
at useful quantiles, then refine as usual. This is more robust than a pure
geometric seed when the weight distribution has heavy tails AND a thin bulk.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .route_encoder import L_MAX_DEFAULT, encode_routes


def _quantile_seed(w: Tensor, l_max: int) -> Tensor:
    """Seed a descending ladder by sampling |w| at geometric quantile positions.

    positions = 1 - 0.5 / 2^i for i in 0..l_max-1, i.e. 0.5, 0.75, 0.875, ..., ~1.0.
    We also force the top scale to 1.0 (caller typically pins it anyway).
    """
    aw = w.abs().flatten()
    if aw.numel() > 2_000_000:
        idx = torch.randint(0, aw.numel(), (2_000_000,), device=aw.device)
        aw = aw[idx]
    probs = torch.tensor(
        [max(0.5, 1.0 - 0.5 / (2 ** i)) for i in range(l_max)],
        device=aw.device,
        dtype=aw.dtype,
    )
    qs = torch.quantile(aw, probs).clamp_min(1e-8)
    # ensure strictly descending (quantiles should already be ascending over probs)
    seed, _ = torch.sort(qs, descending=True)
    # pin top exactly at 1.0
    seed[0] = 1.0
    return seed


def build_seed(w: Tensor, l_max: int, mode: str = "geometric") -> Tensor:
    if mode == "quantile":
        return _quantile_seed(w, l_max=l_max)
    # geometric (with top = max|w|)
    amax = w.abs().max().clamp_min(1e-12)
    seed = amax * (0.5 ** torch.arange(l_max, device=w.device, dtype=w.dtype))
    return seed
