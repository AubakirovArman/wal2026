"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
from __future__ import annotations

import torch


def build_residual_phrase_bank(
    sequences: torch.Tensor,
    phrase_len: int,
    num_slots: int,
    num_variants: int,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    base = int(sequences.max().item()) + 1
    bank = torch.empty(num_slots, num_variants, phrase_len, dtype=torch.uint8)
    summaries: list[dict[str, object]] = []
    for slot_idx in range(num_slots):
        start = slot_idx * phrase_len
        end = start + phrase_len
        phrases = sequences[:, start:end].to(torch.int64)
        codes = phrases[:, 0].clone()
        for pos_idx in range(1, phrase_len):
            codes = codes * base + phrases[:, pos_idx]
        uniq, counts = torch.unique(codes, return_counts=True)
        order = counts.argsort(descending=True)
        take = min(int(order.numel()), num_variants)
        selected = []
        top_counts = []
        for idx in order[:take]:
            code = int(uniq[int(idx)].item())
            top_counts.append(int(counts[int(idx)].item()))
            tokens = []
            for _ in range(phrase_len):
                tokens.append(code % base)
                code //= base
            selected.append(list(reversed(tokens)))
        if not selected:
            selected = [[0] * phrase_len]
            top_counts = [0]
        while len(selected) < num_variants:
            selected.append(selected[-1][:])
            top_counts.append(top_counts[-1])
        bank[slot_idx] = torch.tensor(selected[:num_variants], dtype=torch.uint8)
        summaries.append(
            {
                "slot_index": int(slot_idx),
                "unique_phrase_count": int(uniq.numel()),
                "top_phrase_counts": top_counts[: min(len(top_counts), 8)],
            }
        )
    return bank, summaries


def prepare_slot_phrase_block_bank(
    stage_vector_bank: torch.Tensor,
    phrase_bank: torch.Tensor,
    phrase_len: int,
) -> torch.Tensor:
    num_slots, num_variants, _, block_size = int(phrase_bank.shape[0]), int(phrase_bank.shape[1]), int(phrase_bank.shape[2]), int(stage_vector_bank.shape[2])
    blocks = torch.zeros(num_slots, num_variants, block_size, dtype=torch.float32)
    for slot_idx in range(num_slots):
        start = slot_idx * phrase_len
        for variant_idx in range(num_variants):
            for pos_idx in range(phrase_len):
                token = phrase_bank[slot_idx, variant_idx, pos_idx].to(torch.int64)
                blocks[slot_idx, variant_idx] += stage_vector_bank[start + pos_idx, token]
    return blocks


def slot_blocks_from_tokens(
    stage_vector_bank: torch.Tensor,
    slot_idx: int,
    phrase_len: int,
    tokens: torch.Tensor,
) -> torch.Tensor:
    block_size = int(stage_vector_bank.shape[2])
    blocks = torch.zeros(int(tokens.shape[0]), block_size, dtype=torch.float32)
    start = slot_idx * phrase_len
    for pos_idx in range(phrase_len):
        blocks += stage_vector_bank[start + pos_idx, tokens[:, pos_idx].to(torch.int64)]
    return blocks


def slot_block_rel_mse(pred_blocks: torch.Tensor, target_blocks: torch.Tensor) -> torch.Tensor:
    num = (pred_blocks - target_blocks).square().mean(dim=1)
    den = target_blocks.square().mean(dim=1).clamp_min(1e-6)
    return num / den


def slot_output_rel_mse(
    activation_chunks: torch.Tensor,
    pred_blocks: torch.Tensor,
    target_blocks: torch.Tensor,
) -> torch.Tensor:
    pred = torch.einsum("nab,nb->na", activation_chunks, pred_blocks.to(torch.float32))
    target = torch.einsum("nab,nb->na", activation_chunks, target_blocks.to(torch.float32))
    num = (pred - target).square().mean(dim=1)
    den = target.square().mean(dim=1).clamp_min(1e-6)
    return num / den


def best_residual_match(
    activation_chunks: torch.Tensor,
    target_blocks: torch.Tensor,
    residual_tokens: torch.Tensor,
    residual_blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_out = torch.einsum("nab,nb->na", activation_chunks, target_blocks.to(torch.float32))
    pred_out = torch.einsum("nab,vb->nva", activation_chunks, residual_blocks.to(torch.float32))
    block_num = (residual_blocks.unsqueeze(0) - target_blocks.unsqueeze(1)).square().mean(dim=2)
    block_den = target_blocks.square().mean(dim=1, keepdim=True).clamp_min(1e-6)
    block_rel = block_num / block_den
    out_num = (pred_out - target_out.unsqueeze(1)).square().mean(dim=2)
    out_den = target_out.square().mean(dim=1, keepdim=True).clamp_min(1e-6)
    out_rel = out_num / out_den
    best_idx = (out_rel + block_rel).argmin(dim=1)
    gather_idx = best_idx.unsqueeze(1)
    best_block = block_rel.gather(1, gather_idx).squeeze(1)
    best_out = out_rel.gather(1, gather_idx).squeeze(1)
    best_tokens = residual_tokens[best_idx]
    return best_idx, best_tokens, best_block, best_out