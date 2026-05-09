from __future__ import annotations

import torch
from torch import Tensor, nn

from .full_layer_tiled_runtime import build_grouped_local_plan, full_layer_grouped_local_matmul


class GroupedLocalRouteLinear(nn.Module):
    def __init__(
        self,
        ids: Tensor,
        codebook_w: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        group_rows: int,
        group_cols: int,
    ) -> None:
        super().__init__()
        if codebook_w.shape[0] <= 32767 and ids.dtype != torch.int16:
            ids = ids.to(torch.int16)
        self.register_buffer("ids", ids.contiguous())
        self.register_buffer("codebook_sum", codebook_w.sum(dim=-1).contiguous().to(torch.float16))
        self.register_buffer("row_scale", row_scale.to(torch.float16).contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float16).contiguous())
        else:
            self.bias = None
        self.group_rows = min(int(group_rows), ids.shape[0])
        self.group_cols = min(int(group_cols), ids.shape[1])
        self.out_features = ids.shape[0]
        self.in_features = ids.shape[1]
        self._grouped_plan: list[dict[str, Tensor | int]] | None = None
        self._plan_device: torch.device | None = None

    @classmethod
    def from_encoded(
        cls,
        ids: Tensor,
        codebook_digits: Tensor,
        ladder: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        group_rows: int,
        group_cols: int,
    ) -> "GroupedLocalRouteLinear":
        codebook_w = (codebook_digits.to(torch.float32) * ladder.to(torch.float32)).to(torch.float16)
        return cls(ids, codebook_w, row_scale.to(torch.float16), bias, group_rows=group_rows, group_cols=group_cols)

    def reconstruct_weight(self) -> Tensor:
        return self.codebook_sum[self.ids.long()] * self.row_scale

    def _plan(self) -> list[dict[str, Tensor | int]]:
        if self._grouped_plan is None or self._plan_device != self.ids.device:
            routed_norm = self.codebook_sum[self.ids.long()]
            self._grouped_plan = build_grouped_local_plan(routed_norm, self.group_rows, self.group_cols)
            self._plan_device = self.ids.device
        return self._grouped_plan

    def forward(self, x: Tensor) -> Tensor:
        out = full_layer_grouped_local_matmul(x, self._plan(), self.row_scale)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out