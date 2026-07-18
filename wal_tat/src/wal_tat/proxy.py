"""Training-only proxy codes for hard-forward ternary optimization."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_ternary_proxy(proxy: torch.Tensor, temperature: float) -> torch.Tensor:
    """Smooth three-level staircase with transitions at -0.5 and +0.5."""
    tau = max(float(temperature), 1e-4)
    return torch.sigmoid((proxy - 0.5) / tau) - torch.sigmoid(
        (-proxy - 0.5) / tau
    )


class ProxyTernaryMatrix(nn.Module):
    """Optimize a scalar proxy per weight while always executing hard codes."""

    def __init__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        *,
        compute_dtype: torch.dtype,
        temperature: float = 0.35,
    ):
        super().__init__()
        if codes.ndim != 3 or scales.shape != codes.shape[:2]:
            raise ValueError("codes must be [out, groups, group_size] with matching scales")
        if not torch.all((codes >= -1) & (codes <= 1)):
            raise ValueError("codes must be ternary")
        self.proxy_code = nn.Parameter(codes.detach().float().clone())
        self.group_scale = nn.Parameter(scales.detach().float().clone())
        self.register_buffer("initial_codes", codes.detach().to(torch.int8).clone())
        self.compute_dtype = compute_dtype
        self.temperature = float(temperature)

    @property
    def out_features(self) -> int:
        return self.proxy_code.shape[0]

    @property
    def in_features(self) -> int:
        return self.proxy_code.shape[1] * self.proxy_code.shape[2]

    @property
    def group_size(self) -> int:
        return self.proxy_code.shape[2]

    def hard_codes(self) -> torch.Tensor:
        return self.proxy_code.detach().round().clamp(-1, 1).to(torch.int8)

    def code_churn(self) -> float:
        return float((self.hard_codes() != self.initial_codes).float().mean().item())

    def effective_weight(self) -> torch.Tensor:
        soft = soft_ternary_proxy(self.proxy_code, self.temperature)
        hard = self.proxy_code.round().clamp(-1, 1)
        # Exact hard forward with the smooth staircase supplying the gradient.
        code = hard.detach() + soft - soft.detach()
        value = code * self.group_scale.abs().clamp_min(1e-5).unsqueeze(-1)
        return value.reshape(self.out_features, self.in_features).to(self.compute_dtype)

    @torch.no_grad()
    def constrain_(self) -> None:
        self.proxy_code.clamp_(-1.5, 1.5)
        self.group_scale.clamp_(min=1e-5)


class ProxyTernaryLinear(nn.Module):
    def __init__(self, matrix: ProxyTernaryMatrix, bias=None):
        super().__init__()
        self.matrix = matrix
        self.bias = None if bias is None else nn.Parameter(
            bias.detach().clone(), requires_grad=False
        )
        self.in_features = matrix.in_features
        self.out_features = matrix.out_features

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.matrix.effective_weight().to(value.dtype), self.bias)
