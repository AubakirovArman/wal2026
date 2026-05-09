"""Dynamic-depth ternary route encoder.

w_hat = sum_{i=1..L} d_i * s_i,  d_i in {-1,0,+1}, L <= L_max per-weight.

Greedy residual algorithm, fully vectorized over the input tensor.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

L_MAX_DEFAULT: int = 12


@dataclass
class EncodedRoutes:
    digits: Tensor       # int8, shape (*w.shape, L_max), values {-1,0,+1}
    stop_depth: Tensor   # int8, shape w.shape, values 0..L_max
    residual: Tensor     # same dtype as input, shape w.shape

    @property
    def l_max(self) -> int:
        return int(self.digits.shape[-1])


def encode_routes(
    w: Tensor,
    ladder: Tensor,
    stop_threshold: float = 0.0,
    l_max: int | None = None,
) -> EncodedRoutes:
    if ladder.ndim != 1 or ladder.numel() < 1:
        raise ValueError("ladder must be 1D non-empty")
    ladder = ladder.to(device=w.device, dtype=w.dtype)
    max_possible = min(int(ladder.numel()), L_MAX_DEFAULT)
    lmax = max_possible if l_max is None else min(int(l_max), max_possible)
    if lmax < 1:
        raise ValueError("l_max must be >= 1")

    shape = w.shape
    digits = torch.zeros(*shape, lmax, dtype=torch.int8, device=w.device)
    stop_depth = torch.zeros(shape, dtype=torch.int8, device=w.device)
    residual = w.clone()
    active = torch.ones(shape, dtype=torch.bool, device=w.device)

    for i in range(lmax):
        s = ladder[i]
        take = (residual.abs() >= 0.5 * s) & active
        sign = torch.sign(residual)
        step = (sign * s).to(w.dtype)
        d_i = torch.where(
            take, sign.to(torch.int8), torch.zeros_like(sign, dtype=torch.int8)
        )
        digits[..., i] = d_i
        residual = torch.where(take, residual - step, residual)
        stop_depth = torch.where(
            take, torch.full_like(stop_depth, i + 1), stop_depth
        )
        if stop_threshold > 0.0:
            active = active & (residual.abs() >= stop_threshold)
            if not bool(active.any()):
                break

    return EncodedRoutes(digits=digits, stop_depth=stop_depth, residual=residual)


def decode_routes(
    enc: EncodedRoutes, ladder: Tensor, out_dtype: torch.dtype | None = None
) -> Tensor:
    lmax = enc.l_max
    ladder = ladder[:lmax].to(device=enc.digits.device)
    out_dtype = out_dtype or torch.float32
    digits = enc.digits.to(out_dtype)
    idx_shape = [1] * (enc.digits.ndim - 1) + [lmax]
    idx = torch.arange(lmax, device=enc.digits.device).view(*idx_shape)
    mask = (idx < enc.stop_depth.to(idx.dtype).unsqueeze(-1)).to(out_dtype)
    return (digits * mask * ladder.to(out_dtype)).sum(dim=-1)


def rel_mse(w: Tensor, w_hat: Tensor, eps: float = 1e-12) -> Tensor:
    num = (w - w_hat).pow(2).mean()
    den = w.pow(2).mean().clamp_min(eps)
    return num / den
