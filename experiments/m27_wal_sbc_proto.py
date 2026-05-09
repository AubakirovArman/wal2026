"""M27 WAL-SBC: budgeted exactness on top of WAL-LHA expressive atoms."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from m27_wal_lha_proto import (
    MAX_LEN,
    MODEL_DIR,
    ROOT,
    STRIDE,
    TARGETS,
    _activation_chunks,
    _build_activation_bank,
    _clone_group_with_program_matrix,
    _build_current_cache,
    _compare_encodings,
    _eval_ids,
    _flatten_group_sequences,
    _output_rel_mse_vector,
    _predict_lha_hard_from_features,
    _prepare_stage_vector_bank,
    _reconstruct_from_program_bank,
    _role_name,
    _run_dense_eval,
    _run_preencoded_eval,
    _load_model,
    _selected_residual_blocks,
    _selected_scale_features,
    _semantic_features,
    _train_lha_isa,
    _family_names_for_role,
)
from m27_wal_sbc_core import (
    best_residual_match,
    build_residual_phrase_bank,
    prepare_slot_phrase_block_bank,
    slot_block_rel_mse,
    slot_blocks_from_tokens,
    slot_output_rel_mse,
)
from dwl2_dynamic_route.src.block_vq import GroupedBlockRVQEncoding
from dwl2_dynamic_route.src.encoding_io import save_grouped_encoding_map


DEFAULTS = {
    "text_source": "local",
    "num_windows": 16,
    "lha_calibration_windows": 2,
    "capture_max_len": 1024,
    "activation_rows": 128,
    "group_rows": 28672,
    "block_size": 32,
    "codebook_size": 256,
    "num_stages": 3,
    "product_splits": 4,
    "calibrate_stage_scales": False,
    "sample_limit": 65536,
    "kmeans_iters": 8,
    "batch_size": 16384,
    "num_families": 4,
    "num_slot_variants": 4,
    "phrase_len": 3,
    "train_samples": 65536,
    "atom_iters": 6,
    "assign_chunk_size": 8192,
    "init_family_iters": 4,
    "init_semantic_weight": 1.0,
    "init_sequence_weight": 0.35,
    "hidden_dim": 64,
    "atom_hidden_dim": 32,
    "train_steps": 240,
    "train_batch_size": 1024,
    "log_every": 20,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "warmup_steps": 80,
    "init_supervision_weight": 0.15,
    "family_temp": 1.0,
    "atom_temp": 1.0,
    "phrase_temp": 1.0,
    "output_recon_weight": 1.0,
    "block_aux_weight": 0.15,
    "program_cost_weight": 0.35,
    "correction_weight": 0.25,
    "overlength_weight": 0.25,
    "low_level_gain_weight": 0.12,
    "family_atom_coupling_weight": 0.20,
    "decision_weight": 0.10,
    "atom_selection_weight": 0.10,
    "atom_usage_bonus_weight": 0.08,
    "program_length_target": 12.0,
    "min_atom_calls_per_program": 1.5,
    "decision_margin": 0.05,
    "contrastive_weight": 0.06,
    "cohesion_weight": 0.05,
    "balance_weight": 0.10,
    "usage_floor_weight": 0.20,
    "min_family_mass": 0.06,
    "phrase_entropy_weight": 0.005,
    "assignment_entropy_weight": 0.01,
    "persistent_supervision_weight": 0.03,
    "report_semantic_weight": 1.0,
    "report_sequence_weight": 0.35,
    "matmul_strategy": "full_weight_fast",
    "rebuild_cache": False,
    "bootstrap_cache": str(ROOT / "dwl2_dynamic_route/results/m27_wal_lo_current_l54_gate_up.pt"),
    "current_cache": str(ROOT / "dwl2_dynamic_route/results/m27_wal_sbc_current_l54_gate_up.pt"),
    "budgeted_cache": str(ROOT / "dwl2_dynamic_route/results/m27_wal_sbc_budgeted_l54_gate_up.pt"),
    "sbc_artifact": str(ROOT / "dwl2_dynamic_route/results/m27_wal_sbc_isa_l54_gate_up.pt"),
    "out": str(ROOT / "dwl2_dynamic_route/results/m27_wal_sbc_proto_summary.json"),
    "atom_block_budget": 0.05,
    "atom_output_budget": 0.05,
    "residual_block_budget": 0.16,
    "residual_output_budget": 0.16,
    "residual_bank_size": 8,
    "residual_program_cost": 2.0,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-source", choices=("raw", "local"))
    parser.add_argument("--num-windows", type=int)
    parser.add_argument("--lha-calibration-windows", type=int)
    parser.add_argument("--capture-max-len", type=int)
    parser.add_argument("--activation-rows", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--assign-chunk-size", type=int)
    parser.add_argument("--matmul-strategy")
    parser.add_argument("--atom-block-budget", type=float)
    parser.add_argument("--atom-output-budget", type=float)
    parser.add_argument("--residual-block-budget", type=float)
    parser.add_argument("--residual-output-budget", type=float)
    parser.add_argument("--residual-bank-size", type=int)
    parser.add_argument("--residual-program-cost", type=float)
    parser.add_argument("--current-cache")
    parser.add_argument("--budgeted-cache")
    parser.add_argument("--sbc-artifact")
    parser.add_argument("--out")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--calibrate-stage-scales", action="store_true")
    raw = parser.parse_args()
    config = dict(DEFAULTS)
    for key, value in vars(raw).items():
        if value is not None and value is not False:
            config[key] = value
    if raw.rebuild_cache:
        config["rebuild_cache"] = True
    if raw.calibrate_stage_scales:
        config["calibrate_stage_scales"] = True
    return argparse.Namespace(**config)


def _collect_layer_inputs_budgeted(
    ids: torch.Tensor,
    num_windows: int,
    target_names: tuple[str, ...],
    max_rows_per_layer: int,
    capture_max_len: int,
) -> dict[str, torch.Tensor]:
    model = _load_model()
    module_map = dict(model.named_modules())
    captures: dict[str, list[torch.Tensor]] = {name: [] for name in target_names}
    handles = []
    for name in target_names:
        module = module_map[name]

        def hook(_module, inputs, _output, layer_name=name):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            tensor = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(device="cpu", dtype=torch.bfloat16)
            captures[layer_name].append(tensor.contiguous())

        handles.append(module.register_forward_hook(hook))

    device = model.get_input_embeddings().weight.device
    total_len = ids.size(1)
    capture_len = min(int(capture_max_len), int(MAX_LEN), total_len)
    capture_stride = min(int(STRIDE), capture_len)
    max_windows = max(1, (total_len - capture_len) // max(capture_stride, 1) + 1)
    num_windows = min(num_windows, max_windows)
    with torch.inference_mode():
        for idx in range(num_windows):
            begin = idx * capture_stride
            end = min(begin + capture_len, total_len)
            model(ids[:, begin:end].to(device))

    for handle in handles:
        handle.remove()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    layer_inputs: dict[str, torch.Tensor] = {}
    for name in target_names:
        if not captures[name]:
            raise RuntimeError(f"no calibration activations captured for {name}")
        rows = torch.cat(captures[name], dim=0).to(torch.float32)
        if int(rows.shape[0]) > max_rows_per_layer:
            pick = torch.randperm(int(rows.shape[0]), generator=generator)[:max_rows_per_layer]
            rows = rows[pick]
        layer_inputs[name] = rows.contiguous()
    return layer_inputs


def _build_sbc_layer(name: str, enc: GroupedBlockRVQEncoding, activation_bank: torch.Tensor, args: argparse.Namespace):
    trained = _train_lha_isa(name, enc, activation_bank, args)
    group = trained["group"]
    seq_mat = _flatten_group_sequences(group)
    rows = int(seq_mat.shape[0])
    raw_len = int(trained["raw_len"])
    phrase_len = int(args.phrase_len)
    num_slots = raw_len // phrase_len
    flat_all = torch.arange(rows, dtype=torch.int64)
    scale_features = _selected_scale_features(group, flat_all)
    stage_vector_bank = _prepare_stage_vector_bank(group, int(trained["base"]))
    residual_bank, residual_bank_summaries = build_residual_phrase_bank(trained["sample_sequences"], phrase_len, num_slots, int(args.residual_bank_size))
    residual_block_bank = prepare_slot_phrase_block_bank(stage_vector_bank, residual_bank, phrase_len)
    budgeted_program = seq_mat.clone()
    group_family_ids = torch.empty(rows, dtype=torch.int64)
    atom_calls = torch.zeros(rows, dtype=torch.int32)
    residual_calls = torch.zeros(rows, dtype=torch.int32)
    literal_slots = torch.zeros(rows, dtype=torch.int32)
    program_units = torch.ones(rows, dtype=torch.float32)
    atom_block_rel_sum = 0.0
    atom_output_rel_sum = 0.0
    accepted_atom_block_rel_sum = 0.0
    accepted_atom_output_rel_sum = 0.0
    accepted_residual_block_rel_sum = 0.0
    accepted_residual_output_rel_sum = 0.0
    accepted_atom_count = 0
    accepted_residual_count = 0
    total_slot_count = rows * num_slots
    family_counts = torch.zeros(int(args.num_families), dtype=torch.int64)
    family_names = _family_names_for_role(_role_name(name), int(args.num_families))
    for start in range(0, rows, int(args.assign_chunk_size)):
        end = min(start + int(args.assign_chunk_size), rows)
        chunk = seq_mat[start:end]
        flat_chunk = flat_all[start:end]
        raw_features = torch.cat([
            _semantic_features(chunk, trained["stage_surprisal"], trained["modal_tokens"], phrase_len).to(torch.float32),
            scale_features[start:end].to(torch.float32),
        ], dim=1)
        norm_features = ((raw_features - trained["feature_mean"]) / trained["feature_std"]).to(torch.float32)
        hard_family_orig, hard_atom_ids, _legacy_use_atom, approx_chunk, _logit_margin, decision_prob = _predict_lha_hard_from_features(model=trained["model"], norm_features=norm_features, chunk_size=int(args.assign_chunk_size))
        semantic_family_ids = trained["family_inverse"][hard_family_orig]
        group_family_ids[start:end] = semantic_family_ids
        family_counts += torch.bincount(semantic_family_ids, minlength=int(args.num_families))
        act_chunks = _activation_chunks(activation_bank, flat_chunk, int(group.stage_shape[1]))
        for slot_idx in range(num_slots):
            slot_start = slot_idx * phrase_len
            slot_end = slot_start + phrase_len
            target_tokens = chunk[:, slot_start:slot_end]
            atom_tokens = approx_chunk[:, slot_start:slot_end]
            target_slot_blocks = slot_blocks_from_tokens(stage_vector_bank, slot_idx, phrase_len, target_tokens)
            atom_slot_blocks = slot_blocks_from_tokens(stage_vector_bank, slot_idx, phrase_len, atom_tokens)
            atom_block_rel = slot_block_rel_mse(atom_slot_blocks, target_slot_blocks)
            atom_output_rel = slot_output_rel_mse(act_chunks, atom_slot_blocks, target_slot_blocks)
            atom_accept = (atom_block_rel <= float(args.atom_block_budget)) & (atom_output_rel <= float(args.atom_output_budget))
            _, residual_tokens, residual_block_rel, residual_output_rel = best_residual_match(act_chunks, target_slot_blocks, residual_bank[slot_idx], residual_block_bank[slot_idx])
            residual_accept = (~atom_accept) & (residual_block_rel <= float(args.residual_block_budget)) & (residual_output_rel <= float(args.residual_output_budget))
            selected_tokens = target_tokens.clone()
            if bool(atom_accept.any()):
                selected_tokens[atom_accept] = atom_tokens[atom_accept]
            if bool(residual_accept.any()):
                selected_tokens[residual_accept] = residual_tokens[residual_accept]
            budgeted_program[start:end, slot_start:slot_end] = selected_tokens
            atom_calls[start:end] += atom_accept.to(torch.int32)
            residual_calls[start:end] += residual_accept.to(torch.int32)
            literal_mask = ~(atom_accept | residual_accept)
            literal_slots[start:end] += literal_mask.to(torch.int32)
            program_units[start:end] += atom_accept.to(torch.float32)
            program_units[start:end] += residual_accept.to(torch.float32) * float(args.residual_program_cost)
            program_units[start:end] += literal_mask.to(torch.float32) * float(phrase_len)
            atom_block_rel_sum += float(atom_block_rel.sum().item())
            atom_output_rel_sum += float(atom_output_rel.sum().item())
            accepted_atom_count += int(atom_accept.sum().item())
            accepted_residual_count += int(residual_accept.sum().item())
            if bool(atom_accept.any()):
                accepted_atom_block_rel_sum += float(atom_block_rel[atom_accept].sum().item())
                accepted_atom_output_rel_sum += float(atom_output_rel[atom_accept].sum().item())
            if bool(residual_accept.any()):
                accepted_residual_block_rel_sum += float(residual_block_rel[residual_accept].sum().item())
                accepted_residual_output_rel_sum += float(residual_output_rel[residual_accept].sum().item())

    family_probs = family_counts.to(torch.float64) / max(rows, 1)
    family_entropy = float(-(family_probs.clamp_min(1e-12) * family_probs.clamp_min(1e-12).log()).sum().item())
    residual_blocks = _selected_residual_blocks(group, flat_all)
    budgeted_blocks = _reconstruct_from_program_bank(stage_vector_bank, budgeted_program, group, flat_all, residual_blocks)
    target_blocks = _reconstruct_from_program_bank(stage_vector_bank, seq_mat, group, flat_all, residual_blocks)
    budgeted_output_rel = _output_rel_mse_vector(activation_bank, flat_all, budgeted_blocks, target_blocks, int(group.stage_shape[1]), int(args.assign_chunk_size))
    budgeted_single = _clone_group_with_program_matrix(group, budgeted_program)
    budgeted_enc = GroupedBlockRVQEncoding(groups=(budgeted_single,), row_slices=enc.row_slices, original_shape=enc.original_shape)
    num_tokens = rows * raw_len
    exact_stats = {
        "name": name,
        "raw_program_length": int(raw_len),
        "avg_program_length": float(program_units.mean().item()),
        "avg_low_level_calls": float(atom_calls.to(torch.float32).mean().item()),
        "avg_residual_calls": float(residual_calls.to(torch.float32).mean().item()),
        "avg_literal_slots": float(literal_slots.to(torch.float32).mean().item()),
        "low_level_token_coverage": float(int(atom_calls.sum().item()) * phrase_len / max(num_tokens, 1)),
        "residual_token_coverage": float(int(residual_calls.sum().item()) * phrase_len / max(num_tokens, 1)),
        "accepted_budget_token_coverage": float(int((atom_calls + residual_calls).sum().item()) * phrase_len / max(num_tokens, 1)),
        "program_compression_ratio": float(program_units.sum().item() / max(num_tokens, 1)),
        "active_family_count": int(sum(1 for count in family_counts.tolist() if count > 0)),
        "family_entropy": float(family_entropy),
        "budgeted_token_match": float((budgeted_program == seq_mat).to(torch.float32).mean().item()),
        "budgeted_output_rel_mse": float(budgeted_output_rel.mean().item()),
        "avg_atom_block_rel_mse": float(atom_block_rel_sum / max(total_slot_count, 1)),
        "avg_atom_output_rel_mse": float(atom_output_rel_sum / max(total_slot_count, 1)),
        "accepted_atom_block_rel_mse": float(accepted_atom_block_rel_sum / max(accepted_atom_count, 1)),
        "accepted_atom_output_rel_mse": float(accepted_atom_output_rel_sum / max(accepted_atom_count, 1)),
        "accepted_residual_block_rel_mse": float(accepted_residual_block_rel_sum / max(accepted_residual_count, 1)),
        "accepted_residual_output_rel_mse": float(accepted_residual_output_rel_sum / max(accepted_residual_count, 1)),
        "legacy_decision_mean": float(sum(item.get("sample_decision_rate", 0.0) for item in trained["family_sample_summaries"]) / max(len(trained["family_sample_summaries"]), 1)),
        "sample_lha_only_output_rel_mse": float(trained["sample_output_rel_mse"]),
        "residual_bank_summaries": residual_bank_summaries,
        "family_sample_summaries": trained["family_sample_summaries"],
        "training_history": trained["training_history"],
        "family_names": family_names,
    }
    artifact = {
        "family_names": family_names,
        "feature_mean": trained["feature_mean"],
        "feature_std": trained["feature_std"],
        "feature_centroids": trained["feature_centroids"],
        "sequence_centroids": trained["sequence_centroids"],
        "base_phrase_tokens": trained["base_phrase_tokens"][trained["family_order"]].clone(),
        "family_order": trained["family_order"],
        "family_inverse": trained["family_inverse"],
        "residual_phrase_bank": residual_bank,
        "residual_bank_summaries": residual_bank_summaries,
        "training_history": trained["training_history"],
        "state_dict": trained["state_dict"],
        "budgets": {
            "atom_block_budget": float(args.atom_block_budget),
            "atom_output_budget": float(args.atom_output_budget),
            "residual_block_budget": float(args.residual_block_budget),
            "residual_output_budget": float(args.residual_output_budget),
            "residual_program_cost": float(args.residual_program_cost),
        },
    }
    return budgeted_enc, exact_stats, artifact


def main() -> None:
    args = _parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)
    print(f"[activations] capture {args.lha_calibration_windows} windows for {len(TARGETS)} targets", flush=True)
    layer_inputs = _collect_layer_inputs_budgeted(
        ids,
        int(args.lha_calibration_windows),
        TARGETS,
        int(args.activation_rows),
        int(args.capture_max_len),
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    current_cache = Path(args.current_cache)
    current_enc = _build_current_cache(args, current_cache, TARGETS)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    activation_banks = {name: _build_activation_bank(layer_inputs[name], int(args.block_size)) for name in TARGETS}
    budgeted_encodings = {}
    wal_sbc_rows = []
    budgeted_compare = []
    artifacts = {}
    for name in TARGETS:
        budgeted_enc, exact_stats, artifact = _build_sbc_layer(name, current_enc[name], activation_banks[name], args)
        budgeted_encodings[name] = budgeted_enc
        wal_sbc_rows.append(exact_stats)
        budgeted_compare.append({"name": name, **_compare_encodings(current_enc[name], budgeted_enc)})
        artifacts[name] = artifact
        print(f"[wal-sbc] {name}: avg_program={exact_stats['avg_program_length']:.3f}/{exact_stats['raw_program_length']} atom_calls={exact_stats['avg_low_level_calls']:.3f} residual_calls={exact_stats['avg_residual_calls']:.3f} literal_slots={exact_stats['avg_literal_slots']:.3f} budgeted_out_rel={exact_stats['budgeted_output_rel_mse']:.3f}", flush=True)
    artifact_path = Path(args.sbc_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": 1, "targets": list(TARGETS), "artifacts": artifacts}, artifact_path)
    budgeted_cache = Path(args.budgeted_cache)
    save_grouped_encoding_map(budgeted_cache, budgeted_encodings)
    print("\n[dense] evaluate untouched model", flush=True)
    dense = _run_dense_eval(ids, args.num_windows)
    print(f"\n[eval] strict legacy via {args.matmul_strategy}", flush=True)
    legacy_eval = _run_preencoded_eval(ids, current_cache, args.matmul_strategy, args.num_windows)
    print(f"\n[eval] budgeted exact via {args.matmul_strategy}", flush=True)
    budgeted_eval = _run_preencoded_eval(ids, budgeted_cache, args.matmul_strategy, args.num_windows)
    result = {
        "targets": list(TARGETS),
        "text_source": args.text_source,
        "num_windows": int(args.num_windows),
        "lha_calibration_windows": int(args.lha_calibration_windows),
        "activation_rows": int(args.activation_rows),
        "matmul_strategy": args.matmul_strategy,
        "atom_block_budget": float(args.atom_block_budget),
        "atom_output_budget": float(args.atom_output_budget),
        "residual_block_budget": float(args.residual_block_budget),
        "residual_output_budget": float(args.residual_output_budget),
        "residual_bank_size": int(args.residual_bank_size),
        "residual_program_cost": float(args.residual_program_cost),
        "dense": dense,
        "current_cache": str(current_cache),
        "budgeted_cache": str(budgeted_cache),
        "sbc_artifact": str(artifact_path),
        "wal_sbc": wal_sbc_rows,
        "budgeted_compare": budgeted_compare,
        "legacy_eval": legacy_eval,
        "budgeted_eval": budgeted_eval,
        "delta_budgeted": {
            "ppl_delta": float(budgeted_eval["metrics"]["perplexity"] - legacy_eval["metrics"]["perplexity"]),
            "tok_s_delta": float(budgeted_eval["metrics"]["tok_s"] - legacy_eval["metrics"]["tok_s"]),
            "peak_mb_delta": float(budgeted_eval["eval_peak_mb"] - legacy_eval["eval_peak_mb"]),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    print(f"  dense:          ppl={dense['metrics']['perplexity']:.4f} tok/s={dense['metrics']['tok_s']:.2f} peak_mb={dense['eval_peak_mb']:.1f}", flush=True)
    print(f"  strict_legacy:  ppl={legacy_eval['metrics']['perplexity']:.4f} tok/s={legacy_eval['metrics']['tok_s']:.2f} peak_mb={legacy_eval['eval_peak_mb']:.1f}", flush=True)
    print(f"  budgeted_exact: ppl={budgeted_eval['metrics']['perplexity']:.4f} tok/s={budgeted_eval['metrics']['tok_s']:.2f} peak_mb={budgeted_eval['eval_peak_mb']:.1f}", flush=True)
    for row, compare in zip(wal_sbc_rows, budgeted_compare):
        print(f"  {row['name']}: avg_program={row['avg_program_length']:.3f}/{row['raw_program_length']} atom_calls={row['avg_low_level_calls']:.3f} residual_calls={row['avg_residual_calls']:.3f} literal_slots={row['avg_literal_slots']:.3f} family_entropy={row['family_entropy']:.3f} budgeted_rel_mse={compare['recon_rel_mse']:.6f}", flush=True)
    print(f"  delta(budgeted-legacy): ppl={result['delta_budgeted']['ppl_delta']:+.6f} tok/s={result['delta_budgeted']['tok_s_delta']:+.2f} peak_mb={result['delta_budgeted']['peak_mb_delta']:+.1f}", flush=True)
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()