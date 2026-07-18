"""Structured MLP channel windows for atomic up/down compensation."""
from __future__ import annotations

import math
from typing import Tuple

import torch

from .scoring import select_group_mask


def structured_channel_candidate_mask(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    *,
    count: int,
    channel_block_size: int = 128,
    block_count: int = 4,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Select low-damage up-projection groups inside a few channel blocks.

    Rows of an up projection are intermediate MLP channels. Constraining them
    to a small number of g128 row blocks bounds the linked down-projection
    compensation window.
    """
    if scores.ndim != 2 or eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError("scores and eligible must be matching two-dimensional tensors")
    if channel_block_size <= 0 or block_count <= 0:
        raise ValueError("channel_block_size and block_count must be positive")
    available = int(eligible.sum().item())
    if not 0 < count <= available:
        raise ValueError(f"count {count} is outside [1, {available}]")
    blocks = math.ceil(scores.shape[0] / channel_block_size)
    if block_count > blocks:
        raise ValueError("block_count exceeds the number of channel blocks")

    quota = math.ceil(count / block_count)
    ranked = []
    for block in range(blocks):
        start = block * channel_block_size
        stop = min(start + channel_block_size, scores.shape[0])
        values = scores[start:stop][eligible[start:stop]].float()
        if values.numel() == 0:
            continue
        local = torch.topk(values, min(quota, values.numel()), largest=False).values
        ranked.append((float(local.mean().item()), block, int(values.numel())))
    ranked.sort()
    if len(ranked) < block_count:
        raise ValueError("not enough eligible channel blocks")

    chosen = [block for _, block, _ in ranked[:block_count]]
    capacity = sum(available_in_block for _, block, available_in_block in ranked if block in chosen)
    cursor = block_count
    while capacity < count and cursor < len(ranked):
        _, block, available_in_block = ranked[cursor]
        chosen.append(block)
        capacity += available_in_block
        cursor += 1
    if capacity < count:
        raise ValueError("selected channel blocks do not contain enough eligible groups")

    restricted = torch.zeros_like(eligible)
    for block in chosen:
        start = block * channel_block_size
        restricted[start : start + channel_block_size] = eligible[
            start : start + channel_block_size
        ]
    mask = select_group_mask(scores, count, eligible=restricted, strategy="lowest")
    return mask, tuple(sorted(chosen))


def linked_down_group_mask(
    down_group_state: torch.Tensor,
    channel_blocks: Tuple[int, ...],
) -> torch.Tensor:
    """Reopen all down-projection output rows for linked input-channel blocks."""
    if down_group_state.ndim != 2 or down_group_state.dtype != torch.bool:
        raise ValueError("down_group_state must be a two-dimensional bool tensor")
    result = torch.zeros_like(down_group_state)
    for block in channel_blocks:
        if not 0 <= block < result.shape[1]:
            raise ValueError(f"channel block {block} is outside down group range")
        result[:, block] = True
    return result
