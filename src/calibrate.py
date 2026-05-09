"""Per-family ladder calibration.

Given a sample of weights for a family (a flat real tensor), fit a
ladder of strictly decreasing positive scales s_1 > s_2 > ... > s_L
that minimizes overall relMSE under greedy residual encoding.

Strategy:
    - Start with a geometric seed: s_1 = c1 * max|w|, s_i = s_1 * ratio^(i-1).
    - Refine each scale by 1D coordinate descent on the residual distribution
      (closed-form optimum of sum_i d_i * s_i against r is s = sum(d*r) / sum(d*d)).

Keep this simple; sophistication comes later if quality requires.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .route_encoder import EncodedRoutes, L_MAX_DEFAULT, encode_routes
from .seeds import build_seed


def _geometric_seed(w: Tensor, l_max: int, ratio: float = 0.5, q: float = 0.999) -> Tensor:
    """Seed geometric ladder from a high quantile of |w| (robust to outliers)."""
    aw = w.abs().flatten()
    # torch.quantile is O(n log n) — sample if huge
    if aw.numel() > 10_000_000:
        idx = torch.randint(0, aw.numel(), (10_000_000,), device=aw.device)
        aw = aw[idx]
    top = torch.quantile(aw, q).clamp_min(1e-12)
    seed = top * (ratio ** torch.arange(l_max, device=w.device, dtype=w.dtype))
    return seed


def _refine_once(w: Tensor, ladder: Tensor, l_max: int, pin_top: bool = False) -> Tensor:
    enc = encode_routes(w, ladder, stop_threshold=0.0, l_max=l_max)
    digits = enc.digits.to(w.dtype)
    ladder = ladder.clone()
    for i in range(l_max):
        if pin_top and i == 0:
            continue  # keep ladder[0] at its pinned value
        contrib_wo_i = (digits * ladder).sum(dim=-1) - digits[..., i] * ladder[i]
        r_i = w - contrib_wo_i
        d_i = digits[..., i]
        denom = (d_i * d_i).sum().clamp_min(1.0)
        num = (d_i * r_i).sum()
        s_new = (num / denom).clamp_min(1e-8)
        ladder[i] = s_new
    top = ladder[0].item() if pin_top else None
    ladder, _ = torch.sort(ladder.abs().clamp_min(1e-8), descending=True)
    if pin_top:
        # enforce that the pinned value stays at position 0, even if some refined
        # scale grew larger (it shouldn't on well-normalized data).
        ladder = torch.cat([torch.tensor([top], device=ladder.device, dtype=ladder.dtype),
                            ladder[ladder < top][: l_max - 1]])
        if ladder.numel() < l_max:
            pad = torch.full((l_max - ladder.numel(),), 1e-8, device=ladder.device, dtype=ladder.dtype)
            ladder = torch.cat([ladder, pad])
    return ladder


def calibrate_ladder(
    w_sample: Tensor,
    l_max: int = L_MAX_DEFAULT,
    ratio: float = 0.5,
    refine_iters: int = 8,
    pin_top: bool = False,
    top_value: float | None = None,
    seed: str = "geometric",
) -> Tensor:
    """Fit a descending positive ladder of length l_max.

    seed='geometric' starts from s_i = max * ratio^i.
    seed='quantile'  starts from quantile positions of |w| (robust to heavy tails).
    """
    w = w_sample.detach().to(torch.float32).flatten()
    if w.numel() > 2_000_000:
        idx = torch.randint(0, w.numel(), (2_000_000,), device=w.device)
        w = w[idx]
    ladder = build_seed(w, l_max=l_max, mode=seed)
    if pin_top:
        tv = 1.0 if top_value is None else float(top_value)
        ladder[0] = tv
    for _ in range(refine_iters):
        ladder = _refine_once(w, ladder, l_max=l_max, pin_top=pin_top)
    return ladder
