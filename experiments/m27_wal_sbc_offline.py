"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
from __future__ import annotations

from pathlib import Path

import torch

from m27_wal_dr_proto import _sample_training_corpus
from m27_wal_lha_proto import (
    HighExpressivityAtomISAEncoder,
    _activation_chunks,
    _clone_group_with_program_matrix,
    _family_names_for_role,
    _flatten_group_sequences,
    _output_rel_mse_vector,
    _predict_lha_hard_from_features,
    _prepare_stage_vector_bank,
    _reconstruct_from_program_bank,
    _role_name,
    _selected_residual_blocks,
    _selected_scale_features,
    _semantic_features,
)
from m27_wal_sbc_core import (
    best_residual_match,
    prepare_slot_phrase_block_bank,
    slot_block_rel_mse,
    slot_blocks_from_tokens,
    slot_output_rel_mse,
)
from dwl2_dynamic_route.src.block_vq import GroupedBlockRVQEncoding


def load_sbc_artifacts(artifact_path: str | Path) -> dict[str, dict[str, object]]:
    raw = torch.load(Path(artifact_path), map_location="cpu", weights_only=False)
    states: dict[str, dict[str, object]] = {}
    for name, artifact in raw["artifacts"].items():
        base_phrases = artifact["base_phrase_tokens"].to(torch.uint8)
        state_dict = artifact["state_dict"]
        model = HighExpressivityAtomISAEncoder(
            input_dim=int(artifact["feature_mean"].numel()),
            hidden_dim=int(state_dict["backbone.0.weight"].shape[0]),
            atom_hidden_dim=int(state_dict["atom_generator.context_proj.weight"].shape[0]),
            num_families=len(artifact["family_names"]),
            num_slots=int(base_phrases.shape[1]),
            num_slot_variants=int(base_phrases.shape[2]),
            phrase_len=int(base_phrases.shape[3]),
            base=int(state_dict["atom_generator.base_phrase_logits"].shape[-1]),
            init_phrases=base_phrases,
        ).cpu()
        model.load_state_dict(state_dict)
        model.eval()
        states[name] = {**artifact, "model": model}
    return states


def _stage_feature_stats(enc: GroupedBlockRVQEncoding, train_samples: int, phrase_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    sample_sequences, _, _, _ = _sample_training_corpus(enc, train_samples, seed=0)
    base = int(sample_sequences.max().item()) + 1
    stage_surprisal = -(
        (
            torch.stack(
                [torch.bincount(sample_sequences[:, pos].to(torch.int64), minlength=base) for pos in range(int(sample_sequences.shape[1]))],
                dim=0,
            ).to(torch.float32)
            / max(int(sample_sequences.shape[0]), 1)
        )
        .clamp_min(1e-12)
        .log()
    )
    modal_tokens = torch.stack(
        [torch.bincount(sample_sequences[:, pos].to(torch.int64), minlength=base).argmax() for pos in range(int(sample_sequences.shape[1]))],
        dim=0,
    ).to(torch.uint8)
    return stage_surprisal, modal_tokens


def iter_scored_chunks(
    name: str,
    enc: GroupedBlockRVQEncoding,
    activation_bank: torch.Tensor,
    layer_state: dict[str, object],
    train_samples: int,
    assign_chunk_size: int,
    sample_only: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    group = enc.groups[0]
    phrase_len = int(layer_state["base_phrase_tokens"].shape[3])
    num_slots = int(layer_state["base_phrase_tokens"].shape[1])
    stage_surprisal, modal_tokens = _stage_feature_stats(enc, train_samples, phrase_len)
    stage_vector_bank = _prepare_stage_vector_bank(group, int(layer_state["model"].base))
    residual_bank = layer_state["residual_phrase_bank"].to(torch.uint8)
    residual_block_bank = prepare_slot_phrase_block_bank(stage_vector_bank, residual_bank, phrase_len)
    if sample_only:
        sequences, _, scale_features, flat_indices = _sample_training_corpus(enc, train_samples, seed=0)
    else:
        sequences = _flatten_group_sequences(group)
        flat_indices = torch.arange(int(sequences.shape[0]), dtype=torch.int64)
        scale_features = _selected_scale_features(group, flat_indices)
    meta = {
        "name": name,
        "group": group,
        "phrase_len": phrase_len,
        "num_slots": num_slots,
        "stage_vector_bank": stage_vector_bank,
        "sequences": sequences,
        "flat_indices": flat_indices,
        "blocks_per_row": int(group.stage_shape[1]),
    }
    chunks = []
    for start in range(0, int(sequences.shape[0]), assign_chunk_size):
        end = min(start + assign_chunk_size, int(sequences.shape[0]))
        chunk = sequences[start:end]
        flat_chunk = flat_indices[start:end]
        raw_features = torch.cat(
            [
                _semantic_features(chunk, stage_surprisal, modal_tokens, phrase_len).to(torch.float32),
                scale_features[start:end].to(torch.float32),
            ],
            dim=1,
        )
        norm_features = ((raw_features - layer_state["feature_mean"]) / layer_state["feature_std"]).to(torch.float32)
        hard_family_orig, _, _, approx_chunk, _, _ = _predict_lha_hard_from_features(layer_state["model"], norm_features, assign_chunk_size)
        semantic_family_ids = layer_state["family_inverse"][hard_family_orig]
        act_chunks = _activation_chunks(activation_bank, flat_chunk, int(group.stage_shape[1]))
        slot_payloads = []
        for slot_idx in range(num_slots):
            start_pos = slot_idx * phrase_len
            end_pos = start_pos + phrase_len
            target_tokens = chunk[:, start_pos:end_pos]
            atom_tokens = approx_chunk[:, start_pos:end_pos]
            target_blocks = slot_blocks_from_tokens(stage_vector_bank, slot_idx, phrase_len, target_tokens)
            atom_blocks = slot_blocks_from_tokens(stage_vector_bank, slot_idx, phrase_len, atom_tokens)
            residual_idx, residual_tokens, residual_block_rel, residual_output_rel = best_residual_match(
                act_chunks, target_blocks, residual_bank[slot_idx], residual_block_bank[slot_idx]
            )
            slot_payloads.append(
                {
                    "slot_idx": slot_idx,
                    "target_tokens": target_tokens,
                    "atom_tokens": atom_tokens,
                    "residual_idx": residual_idx,
                    "residual_tokens": residual_tokens,
                    "atom_block_rel": slot_block_rel_mse(atom_blocks, target_blocks),
                    "atom_output_rel": slot_output_rel_mse(act_chunks, atom_blocks, target_blocks),
                    "residual_block_rel": residual_block_rel,
                    "residual_output_rel": residual_output_rel,
                }
            )
        chunks.append({"start": start, "end": end, "semantic_family_ids": semantic_family_ids, "slot_payloads": slot_payloads})
    return meta, chunks


def build_budgeted_layer_from_chunks(
    name: str,
    enc: GroupedBlockRVQEncoding,
    activation_bank: torch.Tensor,
    layer_state: dict[str, object],
    meta: dict[str, object],
    chunks: list[dict[str, object]],
    atom_output_budget: float,
    residual_output_budget: float,
    residual_program_cost: float,
    atom_block_budget: float | None,
    residual_block_budget: float | None,
) -> tuple[GroupedBlockRVQEncoding, dict[str, float]]:
    sequences = meta["sequences"]
    phrase_len = int(meta["phrase_len"])
    budgeted_program = sequences.clone()
    atom_calls = torch.zeros(int(sequences.shape[0]), dtype=torch.int32)
    residual_calls = torch.zeros(int(sequences.shape[0]), dtype=torch.int32)
    literal_slots = torch.zeros(int(sequences.shape[0]), dtype=torch.int32)
    program_units = torch.ones(int(sequences.shape[0]), dtype=torch.float32)
    for chunk in chunks:
        for slot in chunk["slot_payloads"]:
            atom_accept = slot["atom_output_rel"] <= float(atom_output_budget)
            if atom_block_budget is not None:
                atom_accept &= slot["atom_block_rel"] <= float(atom_block_budget)
            residual_accept = (~atom_accept) & (slot["residual_output_rel"] <= float(residual_output_budget))
            if residual_block_budget is not None:
                residual_accept &= slot["residual_block_rel"] <= float(residual_block_budget)
            literal_mask = ~(atom_accept | residual_accept)
            row_slice = slice(int(chunk["start"]), int(chunk["end"]))
            start_pos = int(slot["slot_idx"]) * phrase_len
            end_pos = start_pos + phrase_len
            if bool(atom_accept.any()):
                budgeted_program[row_slice, start_pos:end_pos][atom_accept] = slot["atom_tokens"][atom_accept]
            if bool(residual_accept.any()):
                budgeted_program[row_slice, start_pos:end_pos][residual_accept] = slot["residual_tokens"][residual_accept]
            atom_calls[row_slice] += atom_accept.to(torch.int32)
            residual_calls[row_slice] += residual_accept.to(torch.int32)
            literal_slots[row_slice] += literal_mask.to(torch.int32)
            program_units[row_slice] += atom_accept.to(torch.float32)
            program_units[row_slice] += residual_accept.to(torch.float32) * float(residual_program_cost)
            program_units[row_slice] += literal_mask.to(torch.float32) * float(phrase_len)
    budgeted_single = _clone_group_with_program_matrix(meta["group"], budgeted_program)
    budgeted_enc = GroupedBlockRVQEncoding(groups=(budgeted_single,), row_slices=enc.row_slices, original_shape=enc.original_shape)
    residual_blocks = _selected_residual_blocks(meta["group"], meta["flat_indices"])
    pred_blocks = _reconstruct_from_program_bank(meta["stage_vector_bank"], budgeted_program, meta["group"], meta["flat_indices"], residual_blocks)
    target_blocks = _reconstruct_from_program_bank(meta["stage_vector_bank"], sequences, meta["group"], meta["flat_indices"], residual_blocks)
    output_rel = _output_rel_mse_vector(
        activation_bank=activation_bank,
        flat_indices=meta["flat_indices"],
        pred_blocks=pred_blocks,
        target_blocks=target_blocks,
        blocks_per_row=meta["blocks_per_row"],
        chunk_size=8192,
    )
    num_tokens = int(sequences.shape[0]) * int(sequences.shape[1])
    stats = {
        "avg_program_length": float(program_units.mean().item()),
        "avg_low_level_calls": float(atom_calls.to(torch.float32).mean().item()),
        "avg_residual_calls": float(residual_calls.to(torch.float32).mean().item()),
        "avg_literal_slots": float(literal_slots.to(torch.float32).mean().item()),
        "accepted_budget_token_coverage": float(int((atom_calls + residual_calls).sum().item()) * phrase_len / max(num_tokens, 1)),
        "budgeted_output_rel_mse": float(output_rel.mean().item()),
        "family_entropy": float(
            -(
                (torch.bincount(torch.cat([chunk["semantic_family_ids"] for chunk in chunks]), minlength=len(layer_state["family_names"])).to(torch.float64) / max(int(sequences.shape[0]), 1)).clamp_min(1e-12)
                * (torch.bincount(torch.cat([chunk["semantic_family_ids"] for chunk in chunks]), minlength=len(layer_state["family_names"])).to(torch.float64) / max(int(sequences.shape[0]), 1)).clamp_min(1e-12).log()
            ).sum().item()
        ),
    }
    return budgeted_enc, {"name": name, **stats}


def build_full_score_surface(
    enc: GroupedBlockRVQEncoding,
    activation_bank: torch.Tensor,
    layer_state: dict[str, object],
    train_samples: int,
    assign_chunk_size: int,
) -> dict[str, object]:
    group = enc.groups[0]
    sequences = _flatten_group_sequences(group)
    rows = int(sequences.shape[0])
    phrase_len = int(layer_state["base_phrase_tokens"].shape[3])
    num_slots = int(layer_state["base_phrase_tokens"].shape[1])
    flat_indices = torch.arange(rows, dtype=torch.int64)
    scale_features = _selected_scale_features(group, flat_indices)
    stage_surprisal, modal_tokens = _stage_feature_stats(enc, train_samples, phrase_len)
    approx_program = sequences.clone()
    residual_program = sequences.clone()
    atom_output_rel = torch.empty(rows, num_slots, dtype=torch.float16)
    residual_output_rel = torch.empty(rows, num_slots, dtype=torch.float16)
    family_ids = torch.empty(rows, dtype=torch.int16)
    stage_vector_bank = _prepare_stage_vector_bank(group, int(layer_state["model"].base))
    residual_bank = layer_state["residual_phrase_bank"].to(torch.uint8)
    residual_block_bank = prepare_slot_phrase_block_bank(stage_vector_bank, residual_bank, phrase_len)
    for start in range(0, rows, assign_chunk_size):
        end = min(start + assign_chunk_size, rows)
        chunk = sequences[start:end]
        flat_chunk = flat_indices[start:end]
        raw_features = torch.cat(
            [
                _semantic_features(chunk, stage_surprisal, modal_tokens, phrase_len).to(torch.float32),
                scale_features[start:end].to(torch.float32),
            ],
            dim=1,
        )
        norm_features = ((raw_features - layer_state["feature_mean"]) / layer_state["feature_std"]).to(torch.float32)
        hard_family_orig, _, _, approx_chunk, _, _ = _predict_lha_hard_from_features(layer_state["model"], norm_features, assign_chunk_size)
        family_ids[start:end] = layer_state["family_inverse"][hard_family_orig].to(torch.int16)
        act_chunks = _activation_chunks(activation_bank, flat_chunk, int(group.stage_shape[1]))
        approx_program[start:end] = approx_chunk
        for slot_idx in range(num_slots):
            start_pos = slot_idx * phrase_len
            end_pos = start_pos + phrase_len
            target_tokens = chunk[:, start_pos:end_pos]
            atom_tokens = approx_chunk[:, start_pos:end_pos]
            target_blocks = slot_blocks_from_tokens(stage_vector_bank, slot_idx, phrase_len, target_tokens)
            atom_blocks = slot_blocks_from_tokens(stage_vector_bank, slot_idx, phrase_len, atom_tokens)
            _, residual_tokens, _, residual_output = best_residual_match(
                act_chunks, target_blocks, residual_bank[slot_idx], residual_block_bank[slot_idx]
            )
            residual_program[start:end, start_pos:end_pos] = residual_tokens
            atom_output_rel[start:end, slot_idx] = slot_output_rel_mse(act_chunks, atom_blocks, target_blocks).to(torch.float16)
            residual_output_rel[start:end, slot_idx] = residual_output.to(torch.float16)
    return {
        "group": group,
        "sequences": sequences,
        "approx_program": approx_program,
        "residual_program": residual_program,
        "atom_output_rel": atom_output_rel,
        "residual_output_rel": residual_output_rel,
        "family_ids": family_ids,
        "stage_vector_bank": stage_vector_bank,
        "flat_indices": flat_indices,
        "blocks_per_row": int(group.stage_shape[1]),
        "phrase_len": phrase_len,
        "num_slots": num_slots,
    }


def build_budgeted_layer_from_surface(
    name: str,
    enc: GroupedBlockRVQEncoding,
    activation_bank: torch.Tensor,
    layer_state: dict[str, object],
    surface: dict[str, object],
    atom_output_budget: float,
    residual_output_budget: float,
    residual_program_cost: float,
) -> tuple[GroupedBlockRVQEncoding, dict[str, float]]:
    sequences = surface["sequences"]
    num_slots = int(surface["num_slots"])
    phrase_len = int(surface["phrase_len"])
    atom_accept = surface["atom_output_rel"].to(torch.float32) <= float(atom_output_budget)
    residual_accept = (~atom_accept) & (surface["residual_output_rel"].to(torch.float32) <= float(residual_output_budget))
    literal_mask = ~(atom_accept | residual_accept)
    budgeted_program = sequences.clone()
    for slot_idx in range(num_slots):
        start_pos = slot_idx * phrase_len
        end_pos = start_pos + phrase_len
        slot_atom = atom_accept[:, slot_idx]
        slot_residual = residual_accept[:, slot_idx]
        if bool(slot_atom.any()):
            budgeted_program[:, start_pos:end_pos][slot_atom] = surface["approx_program"][:, start_pos:end_pos][slot_atom]
        if bool(slot_residual.any()):
            budgeted_program[:, start_pos:end_pos][slot_residual] = surface["residual_program"][:, start_pos:end_pos][slot_residual]
    atom_calls = atom_accept.to(torch.float32).sum(dim=1)
    residual_calls = residual_accept.to(torch.float32).sum(dim=1)
    literal_slots = literal_mask.to(torch.float32).sum(dim=1)
    program_units = 1.0 + atom_calls + residual_calls * float(residual_program_cost) + literal_slots * float(phrase_len)
    budgeted_single = _clone_group_with_program_matrix(surface["group"], budgeted_program)
    budgeted_enc = GroupedBlockRVQEncoding(groups=(budgeted_single,), row_slices=enc.row_slices, original_shape=enc.original_shape)
    residual_blocks = _selected_residual_blocks(surface["group"], surface["flat_indices"])
    pred_blocks = _reconstruct_from_program_bank(surface["stage_vector_bank"], budgeted_program, surface["group"], surface["flat_indices"], residual_blocks)
    target_blocks = _reconstruct_from_program_bank(surface["stage_vector_bank"], sequences, surface["group"], surface["flat_indices"], residual_blocks)
    output_rel = _output_rel_mse_vector(activation_bank, surface["flat_indices"], pred_blocks, target_blocks, int(surface["blocks_per_row"]), 8192)
    probs = torch.bincount(surface["family_ids"].to(torch.int64), minlength=len(layer_state["family_names"])).to(torch.float64)
    probs = probs / max(int(surface["family_ids"].numel()), 1)
    stats = {
        "name": name,
        "avg_program_length": float(program_units.mean().item()),
        "avg_low_level_calls": float(atom_calls.mean().item()),
        "avg_residual_calls": float(residual_calls.mean().item()),
        "avg_literal_slots": float(literal_slots.mean().item()),
        "accepted_budget_token_coverage": float(((atom_calls + residual_calls).sum().item() * phrase_len) / max(int(sequences.numel()), 1)),
        "budgeted_output_rel_mse": float(output_rel.mean().item()),
        "family_entropy": float(-(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum().item()),
    }
    return budgeted_enc, stats