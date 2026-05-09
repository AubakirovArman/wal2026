from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class TileStat:
    row0: int
    col0: int
    unique_routes: int
    usage_entropy_bits: float


def _entropy_bits(values: Tensor) -> float:
    _, counts = torch.unique(values.reshape(-1), return_counts=True)
    probs = counts.to(torch.float32) / counts.sum().clamp_min(1)
    ent = -(probs * probs.clamp_min(1e-12).log2()).sum()
    return float(ent.item())


def sample_tile_stats(
    ids: Tensor,
    tile_rows: int,
    tile_cols: int,
    sample_tiles: int = 16,
    seed: int = 0,
) -> list[TileStat]:
    n_rows, n_cols = ids.shape
    n_tr = n_rows // tile_rows
    n_tc = n_cols // tile_cols
    total = n_tr * n_tc
    if total < 1:
        raise ValueError("tile size is larger than ids shape")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(total, generator=gen)[: min(sample_tiles, total)].tolist()
    stats: list[TileStat] = []
    for flat_idx in order:
        tile_r = flat_idx // n_tc
        tile_c = flat_idx % n_tc
        row0 = tile_r * tile_rows
        col0 = tile_c * tile_cols
        tile = ids[row0 : row0 + tile_rows, col0 : col0 + tile_cols]
        stats.append(
            TileStat(
                row0=row0,
                col0=col0,
                unique_routes=int(torch.unique(tile).numel()),
                usage_entropy_bits=_entropy_bits(tile),
            )
        )
    stats.sort(key=lambda item: (item.unique_routes, item.usage_entropy_bits), reverse=True)
    return stats


def _init_palette(values: Tensor, palette_size: int) -> Tensor:
    uniq, counts = torch.unique(values.reshape(-1), return_counts=True)
    order = counts.argsort(descending=True)
    palette = uniq[order[: min(palette_size, uniq.numel())]]
    if palette.numel() < palette_size:
        pad_src = palette if palette.numel() > 0 else values.reshape(-1)[:1]
        pad = pad_src[torch.randint(0, pad_src.numel(), (palette_size - palette.numel(),), device=values.device)]
        palette = torch.cat([palette, pad], dim=0)
    return palette.to(torch.float32)


def _project_palette(palette: Tensor, candidates: Tensor) -> Tensor:
    uniq = torch.unique(candidates.reshape(-1)).to(torch.float32)
    dist = (palette[:, None] - uniq[None, :]).abs()
    return uniq[dist.argmin(dim=1)]


def _refine_projected_palette(
    palette: Tensor,
    teacher: Tensor,
    candidates: Tensor,
    refine_iters: int = 6,
) -> Tensor:
    uniq = torch.unique(candidates.reshape(-1)).to(torch.float32)
    refined = palette.clone().to(torch.float32)
    target = teacher.reshape(-1, 1)
    for _ in range(refine_iters):
        assign = (target - refined.view(1, -1)).abs().argmin(dim=-1)
        next_palette = refined.clone()
        for idx in range(refined.numel()):
            mask = assign == idx
            if not bool(mask.any()):
                continue
            mean = target[mask].mean()
            next_palette[idx] = uniq[(uniq - mean).abs().argmin()]
        refined = next_palette
    return refined


def distill_tile_palette(
    teacher_tile: Tensor,
    current_tile: Tensor,
    candidate_values: Tensor,
    palette_size: int = 16,
    activation_batch: int = 256,
    steps: int = 120,
    lr: float = 5e-2,
    entropy_reg: float = 1e-3,
    seed: int = 0,
) -> dict[str, object]:
    device = teacher_tile.device
    teacher = teacher_tile.to(torch.float32)
    current = current_tile.to(torch.float32)
    init_palette = _init_palette(current, palette_size)
    logits_init = -8.0 * (current.unsqueeze(-1) - init_palette.view(1, 1, -1)).abs()
    palette = nn.Parameter(init_palette.clone())
    logits = nn.Parameter(logits_init)
    opt = torch.optim.Adam([palette, logits], lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        activation_batch,
        teacher.shape[1],
        generator=gen,
        device=device,
        dtype=torch.float32,
    )
    target = x @ teacher.t()
    base_pred = x @ current.t()
    base_output_mse = float(F.mse_loss(base_pred, target).item())
    base_weight_mse = float(F.mse_loss(current, teacher).item())
    max_entropy_bits = math.log2(max(palette_size, 2))
    for step_idx in range(steps):
        alpha = step_idx / max(steps - 1, 1)
        temp = 1.5 + (0.1 - 1.5) * alpha
        probs = torch.softmax(logits / temp, dim=-1)
        soft_tile = (probs * palette.view(1, 1, -1)).sum(dim=-1)
        pred = x @ soft_tile.t()
        ent = -(probs * probs.clamp_min(1e-12).log2()).sum(dim=-1).mean()
        loss = F.mse_loss(pred, target) + entropy_reg * (ent / max_entropy_bits)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        projected = _project_palette(palette.detach(), candidate_values)
        projected = _refine_projected_palette(projected, teacher, candidate_values)
        assign = (teacher.unsqueeze(-1) - projected.view(1, 1, -1)).abs().argmin(dim=-1)
        hard_tile = projected[assign]
        out = x @ hard_tile.t()
        distilled_output_mse = float(F.mse_loss(out, target).item())
        distilled_weight_mse = float(F.mse_loss(hard_tile, teacher).item())
    return {
        "base_unique": int(torch.unique(current).numel()),
        "distilled_unique": int(torch.unique(hard_tile).numel()),
        "palette_size": int(projected.numel()),
        "base_usage_entropy_bits": _entropy_bits(current),
        "distilled_usage_entropy_bits": _entropy_bits(hard_tile),
        "base_output_mse": base_output_mse,
        "distilled_output_mse": distilled_output_mse,
        "base_weight_mse": base_weight_mse,
        "distilled_weight_mse": distilled_weight_mse,
        "projected_palette": [float(v) for v in projected.tolist()],
    }


def tile_stats_to_json(stats: list[TileStat]) -> list[dict[str, object]]:
    return [asdict(item) for item in stats]