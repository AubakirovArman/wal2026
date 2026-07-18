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
        committed_mask: torch.Tensor | None = None,
        master_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        if codes.ndim != 3 or scales.shape != codes.shape[:2]:
            raise ValueError("codes must be [out, groups, group_size] with matching scales")
        if not torch.all((codes >= -1) & (codes <= 1)):
            raise ValueError("codes must be ternary")
        if committed_mask is None:
            committed_mask = torch.ones(codes.shape[:2], dtype=torch.bool, device=codes.device)
        if committed_mask.shape != codes.shape[:2] or committed_mask.dtype != torch.bool:
            raise ValueError("committed_mask must match the code group shape")
        full_in_features = codes.shape[1] * codes.shape[2]
        if master_weight is None:
            if not committed_mask.all():
                raise ValueError("partial proxy matrices require master_weight")
            base_weight = torch.zeros_like(codes, dtype=torch.float32)
            in_features = full_in_features
        else:
            if master_weight.ndim != 2 or master_weight.shape[0] != codes.shape[0]:
                raise ValueError("master_weight must match the code output dimension")
            if not 0 < master_weight.shape[1] <= full_in_features:
                raise ValueError("master_weight has an invalid input dimension")
            if full_in_features - master_weight.shape[1] >= codes.shape[2]:
                raise ValueError("master_weight padding must be smaller than one group")
            in_features = int(master_weight.shape[1])
            padded = F.pad(
                master_weight.detach().float(), (0, full_in_features - in_features)
            )
            base_weight = padded.view_as(codes)
        self.proxy_code = nn.Parameter(codes.detach().float().clone())
        self.group_scale = nn.Parameter(scales.detach().float().clone())
        self.register_buffer("initial_codes", codes.detach().to(torch.int8).clone())
        self.register_buffer("committed_mask", committed_mask.detach().clone())
        self.register_buffer("base_weight", base_weight.detach().clone())
        self._in_features = in_features
        self.compute_dtype = compute_dtype
        self.temperature = float(temperature)

    @property
    def out_features(self) -> int:
        return self.proxy_code.shape[0]

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def group_size(self) -> int:
        return self.proxy_code.shape[2]

    def hard_codes(self) -> torch.Tensor:
        return self.proxy_code.detach().round().clamp(-1, 1).to(torch.int8)

    def code_churn(self) -> float:
        changed = self.hard_codes() != self.initial_codes
        return float(changed[self.committed_mask].float().mean().item())

    def proxy_anchor_loss(self) -> torch.Tensor:
        """Squared proxy displacement over deployed ternary groups only."""
        delta = (self.proxy_code - self.initial_codes.float()).square()
        return delta[self.committed_mask].mean()

    def effective_weight(self) -> torch.Tensor:
        soft = soft_ternary_proxy(self.proxy_code, self.temperature)
        hard = self.proxy_code.round().clamp(-1, 1)
        # Exact hard forward with the smooth staircase supplying the gradient.
        code = hard.detach() + soft - soft.detach()
        value = code * self.group_scale.abs().clamp_min(1e-5).unsqueeze(-1)
        mixed = torch.where(self.committed_mask.unsqueeze(-1), value, self.base_weight)
        return mixed.reshape(self.out_features, -1)[:, : self.in_features].to(
            self.compute_dtype
        )

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
