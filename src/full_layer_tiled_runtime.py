from __future__ import annotations

from typing import Iterator

import torch
from torch import Tensor

from .tile_palette_runtime import build_hotprefix_local_palette_repr, build_local_palette_repr
from .triton_id_matmul import id_route_linear_matmul
from .triton_local_palette_hotprefix_matmul import local_palette_hotprefix_linear_matmul
from .triton_local_palette_matmul import local_palette_linear_matmul


def iter_tiles(n_rows: int, n_cols: int, tile_rows: int, tile_cols: int) -> Iterator[tuple[int, int, int, int]]:
    for row0 in range(0, n_rows, tile_rows):
        row1 = min(row0 + tile_rows, n_rows)
        for col0 in range(0, n_cols, tile_cols):
            col1 = min(col0 + tile_cols, n_cols)
            yield row0, row1, col0, col1


def build_exact_local_tile_plan(
    routed_norm: Tensor,
    tile_rows: int,
    tile_cols: int,
) -> list[dict[str, Tensor | int]]:
    return build_grouped_local_plan(routed_norm, tile_rows, tile_cols)


def build_grouped_local_plan(
    routed_norm: Tensor,
    group_rows: int,
    group_cols: int,
) -> list[dict[str, Tensor | int]]:
    plan: list[dict[str, Tensor | int]] = []
    for row0, row1, col0, col1 in iter_tiles(routed_norm.shape[0], routed_norm.shape[1], group_rows, group_cols):
        tile = routed_norm[row0:row1, col0:col1]
        palette, local_idx = build_local_palette_repr(tile)
        plan.append(
            {
                "row0": row0,
                "row1": row1,
                "col0": col0,
                "col1": col1,
                "palette": palette,
                "local_idx": local_idx,
            }
        )
    return plan


def build_grouped_local_hotprefix_plan(
    routed_norm: Tensor,
    group_rows: int,
    group_cols: int,
) -> list[dict[str, Tensor | int]]:
    plan: list[dict[str, Tensor | int]] = []
    for row0, row1, col0, col1 in iter_tiles(routed_norm.shape[0], routed_norm.shape[1], group_rows, group_cols):
        tile = routed_norm[row0:row1, col0:col1]
        palette, local_idx = build_hotprefix_local_palette_repr(tile)
        plan.append(
            {
                "row0": row0,
                "row1": row1,
                "col0": col0,
                "col1": col1,
                "palette": palette,
                "local_idx": local_idx,
            }
        )
    return plan


def full_layer_global_tiled_matmul(
    x: Tensor,
    ids: Tensor,
    codebook_sum: Tensor,
    row_scale: Tensor,
    tile_rows: int,
    tile_cols: int,
) -> Tensor:
    return full_layer_grouped_global_matmul(x, ids, codebook_sum, row_scale, tile_rows, tile_cols)


def full_layer_grouped_global_matmul(
    x: Tensor,
    ids: Tensor,
    codebook_sum: Tensor,
    row_scale: Tensor,
    group_rows: int,
    group_cols: int,
) -> Tensor:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x_tiles = {
        (col0, col1): x_2d[:, col0:col1].contiguous()
        for _, _, col0, col1 in iter_tiles(ids.shape[0], ids.shape[1], group_rows, group_cols)
    }
    out = torch.zeros((x_2d.shape[0], ids.shape[0]), device=x.device, dtype=torch.float32)
    for row0, row1, col0, col1 in iter_tiles(ids.shape[0], ids.shape[1], group_rows, group_cols):
        out[:, row0:row1] += id_route_linear_matmul(
            x_tiles[(col0, col1)],
            ids[row0:row1, col0:col1],
            codebook_sum,
            row_scale[row0:row1],
        ).float()
    return out.to(x.dtype).reshape(*x.shape[:-1], ids.shape[0])


def full_layer_local_tiled_matmul(
    x: Tensor,
    plan: list[dict[str, Tensor | int]],
    row_scale: Tensor,
) -> Tensor:
    return full_layer_grouped_local_matmul(x, plan, row_scale)


def full_layer_grouped_local_matmul(
    x: Tensor,
    plan: list[dict[str, Tensor | int]],
    row_scale: Tensor,
) -> Tensor:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x_tiles = {
        (int(item["col0"]), int(item["col1"])): x_2d[:, int(item["col0"]):int(item["col1"])].contiguous()
        for item in plan
    }
    out_features = max(int(item["row1"]) for item in plan)
    out = torch.zeros((x_2d.shape[0], out_features), device=x.device, dtype=torch.float32)
    for item in plan:
        row0 = int(item["row0"])
        row1 = int(item["row1"])
        col0 = int(item["col0"])
        col1 = int(item["col1"])
        out[:, row0:row1] += local_palette_linear_matmul(
            x_tiles[(col0, col1)],
            item["local_idx"],
            item["palette"],
            row_scale[row0:row1],
        ).float()
    return out.to(x.dtype).reshape(*x.shape[:-1], out_features)


def full_layer_grouped_local_hotprefix_matmul(
    x: Tensor,
    plan: list[dict[str, Tensor | int]],
    row_scale: Tensor,
    hot_size: int,
) -> Tensor:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x_tiles = {
        (int(item["col0"]), int(item["col1"])): x_2d[:, int(item["col0"]):int(item["col1"])].contiguous()
        for item in plan
    }
    out_features = max(int(item["row1"]) for item in plan)
    out = torch.zeros((x_2d.shape[0], out_features), device=x.device, dtype=torch.float32)
    for item in plan:
        row0 = int(item["row0"])
        row1 = int(item["row1"])
        col0 = int(item["col0"])
        col1 = int(item["col1"])
        out[:, row0:row1] += local_palette_hotprefix_linear_matmul(
            x_tiles[(col0, col1)],
            item["local_idx"],
            item["palette"],
            row_scale[row0:row1],
            hot_size=hot_size,
        ).float()
    return out.to(x.dtype).reshape(*x.shape[:-1], out_features)


def mean_local_unique(plan: list[dict[str, Tensor | int]]) -> float:
    return sum(int(item["palette"].numel()) for item in plan) / max(len(plan), 1)


def total_local_unique(plan: list[dict[str, Tensor | int]]) -> int:
    return sum(int(item["palette"].numel()) for item in plan)


def launch_count(plan: list[dict[str, Tensor | int]]) -> int:
    return len(plan)