"""Probe behavior gradients and test sparse adjacent ternary-code edits.

This is a checkpoint-neutral development experiment.  The source checkpoint is
never modified.  A target-local counterfactual teacher is used only to measure
the first-order direction in the already committed region of one matrix.  Each
candidate changes at most one code value per selected g128 group and optionally
refits only the scale of those groups.  The winning candidate must pass a
disjoint confirmation slice before a strict recode artifact is written.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from compensation_window import (
    PROJECT,
    WORKSPACE,
    bucket_kl,
    default_model_path,
    evaluate_domains,
    install_checkpoint,
    load_model,
    ratios,
    seed_everything,
    set_submodule,
    sha256_file,
    tensor_output,
)
from counterfactual_bf16_restore_audit import (
    restore_original_committed_groups,
    restore_source_state,
)
from matched_bf16_residual_recovery import (
    MaskedContinuousResidualLinear,
    cache_teacher,
    normalized_mse,
)
from wal_tat import TransactionalTernaryLinear


PROJECTION_MAP = {
    "up_proj": "mlp.up_proj",
    "gate_proj": "mlp.gate_proj",
    "down_proj": "mlp.down_proj",
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
}
SCALE_MODES = ("fixed", "counterfactual_ls")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-layer", type=int, default=24)
    parser.add_argument(
        "--target-projection", choices=tuple(PROJECTION_MAP), default="up_proj"
    )
    parser.add_argument("--calibration-sequences", type=int, default=96)
    parser.add_argument("--selection-start", type=int, default=0)
    parser.add_argument("--selection-count", type=int, default=64)
    parser.add_argument("--confirmation-start", type=int, default=64)
    parser.add_argument("--confirmation-count", type=int, default=64)
    parser.add_argument(
        "--group-budgets", default="64,128,256,512,1024"
    )
    parser.add_argument("--scale-modes", default=",".join(SCALE_MODES))
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--block-weight", type=float, default=0.2)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-topk", type=int, default=64)
    parser.add_argument("--kd-stride", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--min-selection-improvement", type=float, default=5e-5)
    parser.add_argument("--min-confirmation-improvement", type=float, default=0.0)
    parser.add_argument("--max-code-churn", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def sliced_gates(gates: dict, start: int, count: int) -> dict:
    result = {domain: chunks[start : start + count] for domain, chunks in gates.items()}
    if any(len(chunks) != count for chunks in result.values()):
        raise ValueError("suite lacks a requested selection/confirmation slice")
    return result


@torch.no_grad()
def adjacent_move_proposals(
    codes: torch.Tensor,
    scales: torch.Tensor,
    gradient: torch.Tensor,
    committed_mask: torch.Tensor,
    *,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one first-order adjacent move candidate for every code slot.

    Non-zero codes may move only toward zero.  Zero codes choose the sign whose
    fixed-scale displacement has the lower first-order loss.  Padding and
    uncommitted groups receive ``-inf`` scores.
    """
    if codes.ndim != 3 or scales.shape != codes.shape[:2]:
        raise ValueError("codes/scales shape mismatch")
    if gradient.ndim != 2 or gradient.shape[0] != codes.shape[0]:
        raise ValueError("gradient shape mismatch")
    if committed_mask.shape != scales.shape or committed_mask.dtype != torch.bool:
        raise ValueError("committed mask shape mismatch")
    group_size = codes.shape[-1]
    full_in_features = codes.shape[1] * group_size
    if not 0 < in_features <= full_in_features:
        raise ValueError("invalid input feature count")
    padded_gradient = F.pad(
        gradient.float(), (0, full_in_features - gradient.shape[1])
    ).view_as(codes)
    proposed = torch.zeros_like(codes)
    proposed = torch.where(codes < 0, torch.zeros_like(proposed), proposed)
    proposed = torch.where(codes > 0, torch.zeros_like(proposed), proposed)
    zero_sign = torch.where(
        padded_gradient >= 0,
        -torch.ones_like(proposed),
        torch.ones_like(proposed),
    )
    proposed = torch.where(codes == 0, zero_sign, proposed)
    delta = (proposed.float() - codes.float()) * scales.unsqueeze(-1)
    score = -(padded_gradient * delta)
    eligible = committed_mask.unsqueeze(-1).expand_as(codes).clone()
    if in_features < full_in_features:
        valid_columns = torch.arange(full_in_features, device=codes.device)
        valid_columns = valid_columns.view(1, codes.shape[1], group_size)
        eligible &= valid_columns < in_features
    score = torch.where(eligible, score, torch.full_like(score, -torch.inf))
    return proposed, delta, score


@torch.no_grad()
def apply_top_group_moves(
    source_codes: torch.Tensor,
    proposed_codes: torch.Tensor,
    scores: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the best single code move in each of the top scoring groups."""
    if budget < 1:
        raise ValueError("budget must be positive")
    if proposed_codes.shape != source_codes.shape or scores.shape != source_codes.shape:
        raise ValueError("proposal shape mismatch")
    group_scores, positions = scores.max(dim=-1)
    flat_scores = group_scores.reshape(-1)
    finite = torch.isfinite(flat_scores) & (flat_scores > 0)
    available = int(finite.sum().item())
    count = min(int(budget), available)
    if count == 0:
        raise ValueError("no positive first-order adjacent moves")
    masked_scores = torch.where(finite, flat_scores, torch.full_like(flat_scores, -torch.inf))
    chosen_scores, group_indices = torch.topk(masked_scores, count, sorted=True)
    flat_positions = positions.reshape(-1).index_select(0, group_indices)
    candidate = source_codes.clone().reshape(-1, source_codes.shape[-1])
    proposals = proposed_codes.reshape_as(candidate)
    row = torch.arange(count, device=source_codes.device)
    candidate[group_indices, flat_positions] = proposals[
        group_indices, flat_positions
    ]
    return (
        candidate.view_as(source_codes),
        group_indices,
        flat_positions,
        chosen_scores,
    )


@torch.no_grad()
def apply_global_top_group_moves(
    source_codes: dict[str, torch.Tensor],
    proposed_codes: dict[str, torch.Tensor],
    scores: dict[str, torch.Tensor],
    budget: int,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    """Select at most one adjacent code move per group across several matrices."""
    if budget < 1:
        raise ValueError("budget must be positive")
    names = tuple(source_codes)
    if not names or set(proposed_codes) != set(names) or set(scores) != set(names):
        raise ValueError("source, proposal and score maps must have the same keys")
    group_scores = []
    group_positions = {}
    group_offsets = {}
    offset = 0
    for name in names:
        if proposed_codes[name].shape != source_codes[name].shape:
            raise ValueError(f"proposal shape mismatch for {name}")
        if scores[name].shape != source_codes[name].shape:
            raise ValueError(f"score shape mismatch for {name}")
        best, positions = scores[name].max(dim=-1)
        flat = best.reshape(-1)
        group_scores.append(flat)
        group_positions[name] = positions.reshape(-1)
        group_offsets[name] = (offset, offset + flat.numel())
        offset += flat.numel()
    concatenated = torch.cat(group_scores)
    finite = torch.isfinite(concatenated) & (concatenated > 0)
    count = min(int(budget), int(finite.sum().item()))
    if count == 0:
        raise ValueError("no positive first-order adjacent moves")
    masked = torch.where(
        finite, concatenated, torch.full_like(concatenated, -torch.inf)
    )
    chosen_scores, chosen_global = torch.topk(masked, count, sorted=True)
    candidates = {name: value.clone() for name, value in source_codes.items()}
    selected_groups = {}
    selected_positions = {}
    for name in names:
        start, stop = group_offsets[name]
        belongs = (chosen_global >= start) & (chosen_global < stop)
        groups = chosen_global[belongs] - start
        positions = group_positions[name].index_select(0, groups)
        selected_groups[name] = groups
        selected_positions[name] = positions
        if groups.numel() == 0:
            continue
        flat_candidate = candidates[name].reshape(-1, candidates[name].shape[-1])
        flat_proposed = proposed_codes[name].reshape_as(flat_candidate)
        flat_candidate[groups, positions] = flat_proposed[groups, positions]
    return candidates, selected_groups, selected_positions, chosen_scores


@torch.no_grad()
def refit_selected_scales(
    target: torch.Tensor,
    codes: torch.Tensor,
    source_scales: torch.Tensor,
    group_indices: torch.Tensor,
) -> torch.Tensor:
    """Positive least-squares scales for selected groups, rounded to FP16."""
    if target.shape != codes.shape or source_scales.shape != codes.shape[:2]:
        raise ValueError("scale-refit shape mismatch")
    flat_target = target.reshape(-1, target.shape[-1]).float()
    flat_codes = codes.reshape(-1, codes.shape[-1]).float()
    selected_target = flat_target.index_select(0, group_indices)
    selected_codes = flat_codes.index_select(0, group_indices)
    numerator = (selected_target * selected_codes).sum(dim=-1)
    denominator = selected_codes.square().sum(dim=-1).clamp_min(1)
    fitted = numerator.div(denominator).clamp_min(1e-5).half().float()
    result = source_scales.clone().reshape(-1)
    result[group_indices] = fitted
    return result.view_as(source_scales)


@torch.no_grad()
def candidate_statistics(
    source_codes: torch.Tensor,
    source_scales: torch.Tensor,
    candidate_codes: torch.Tensor,
    candidate_scales: torch.Tensor,
    committed_mask: torch.Tensor,
    chosen_scores: torch.Tensor,
) -> dict:
    active_codes = committed_mask.unsqueeze(-1).expand_as(source_codes)
    changed_codes = (candidate_codes != source_codes) & active_codes
    changed_scales = (candidate_scales != source_scales) & committed_mask
    return {
        "changed_code_values": int(changed_codes.sum().item()),
        "changed_scale_groups": int(changed_scales.sum().item()),
        "code_churn": float(changed_codes.float().sum().div(active_codes.sum()).item()),
        "selected_first_order_score_sum": float(chosen_scores.sum().item()),
        "selected_first_order_score_min": float(chosen_scores.min().item()),
        "selected_first_order_score_max": float(chosen_scores.max().item()),
    }


def main() -> None:
    args = parse_args()
    budgets = tuple(
        int(value.strip()) for value in args.group_budgets.split(",") if value.strip()
    )
    scale_modes = tuple(
        value.strip() for value in args.scale_modes.split(",") if value.strip()
    )
    if not budgets or any(value < 1 for value in budgets):
        raise ValueError("group budgets must be positive")
    if not scale_modes or set(scale_modes) - set(SCALE_MODES):
        raise ValueError("unsupported scale mode")
    source_path = args.source_checkpoint.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = suite["calibration"][: args.calibration_sequences]
    if len(calibration) != args.calibration_sequences:
        raise ValueError("suite lacks requested calibration sequences")
    selection_gates = sliced_gates(
        suite["gates"], args.selection_start, args.selection_count
    )
    confirmation_gates = sliced_gates(
        suite["gates"], args.confirmation_start, args.confirmation_count
    )
    selection_range = set(
        range(args.selection_start, args.selection_start + args.selection_count)
    )
    confirmation_range = set(
        range(
            args.confirmation_start,
            args.confirmation_start + args.confirmation_count,
        )
    )
    if selection_range & confirmation_range:
        raise ValueError("selection and confirmation slices overlap")
    target_name = (
        f"model.layers.{args.target_layer}."
        f"{PROJECTION_MAP[args.target_projection]}"
    )
    if target_name not in payload["matrices"]:
        raise ValueError("target matrix is absent from source checkpoint")

    seed_everything(args.seed)
    started = time.time()
    model = load_model(model_path, args.device)
    original_weight = model.get_submodule(target_name).weight.detach().cpu().clone()
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    confirmation_baseline = evaluate_domains(model, confirmation_gates, args.device)
    matrices = install_checkpoint(model, payload, args.device)
    matrix = matrices[target_name]
    selection_source = evaluate_domains(model, selection_gates, args.device)
    confirmation_source = evaluate_domains(model, confirmation_gates, args.device)
    source_selection_ratios = ratios(selection_source, selection_baseline)
    source_confirmation_ratios = ratios(confirmation_source, confirmation_baseline)

    restore_original_committed_groups(matrix, original_weight)
    teacher = cache_teacher(model, calibration, args.target_layer, args.device, args)
    restore_source_state(model, matrices, payload, args.device)
    source_recheck = evaluate_domains(model, confirmation_gates, args.device)
    source_restoration_exact = all(
        source_recheck[domain]["nll"] == confirmation_source[domain]["nll"]
        for domain in confirmation_gates
    )
    if not source_restoration_exact:
        raise RuntimeError("source did not restore exactly after teacher cache")

    target = model.get_submodule(target_name)
    bias = target.bias
    probe = MaskedContinuousResidualLinear(matrix, bias).to(args.device)
    set_submodule(model, target_name, probe)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    probe.delta.requires_grad_(True)
    capture: dict[str, torch.Tensor] = {}
    handle = model.model.layers[args.target_layer].register_forward_hook(
        lambda _module, _inputs, output: capture.__setitem__(
            "block", tensor_output(output)
        )
    )
    accumulated_gradient = torch.zeros_like(probe.delta, dtype=torch.float32)
    gradient_history = []
    model.train()
    try:
        for index, chunk in enumerate(calibration):
            batch = chunk.unsqueeze(0).to(args.device)
            model.zero_grad(set_to_none=True)
            capture.clear()
            logits = model(input_ids=batch[:, :-1], use_cache=False).logits
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                batch[:, 1:].reshape(-1),
            )
            block_loss = normalized_mse(capture["block"], teacher[index]["block"])
            kd_loss = bucket_kl(logits, teacher[index], args)
            loss = (
                args.ce_weight * ce
                + args.block_weight * block_loss
                + args.kd_weight * kd_loss
            )
            loss.backward()
            accumulated_gradient.add_(probe.delta.grad.detach().float())
            if index == 0 or (index + 1) % 32 == 0 or index + 1 == len(calibration):
                item = {
                    "sequence": index + 1,
                    "ce": float(ce.item()),
                    "block_loss": float(block_loss.item()),
                    "kd_loss": float(kd_loss.item()),
                    "gradient_norm": float(probe.delta.grad.float().norm().item()),
                }
                gradient_history.append(item)
                print(
                    f"gradient {index + 1}/{len(calibration)} "
                    f"kd={item['kd_loss']:.6f} block={item['block_loss']:.6f}",
                    flush=True,
                )
    finally:
        handle.remove()
    accumulated_gradient.div_(len(calibration))
    set_submodule(
        model,
        target_name,
        TransactionalTernaryLinear(matrix, bias).to(args.device),
    )
    model.eval()

    source_codes = matrix.committed_codes.detach().clone()
    source_scales = matrix.group_scale.detach().clone()
    proposed_codes, _move_delta, move_scores = adjacent_move_proposals(
        source_codes,
        source_scales,
        accumulated_gradient,
        matrix.committed_mask,
        in_features=matrix.in_features,
    )
    original_grouped = F.pad(
        original_weight.to(args.device).float(), (0, matrix.padding)
    ).view_as(source_codes)
    candidates: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    candidate_results: dict[str, dict] = {}
    for budget in budgets:
        candidate_codes, group_indices, _positions, chosen_scores = apply_top_group_moves(
            source_codes, proposed_codes, move_scores, budget
        )
        for scale_mode in scale_modes:
            if scale_mode == "fixed":
                candidate_scales = source_scales.clone()
            else:
                candidate_scales = refit_selected_scales(
                    original_grouped,
                    candidate_codes,
                    source_scales,
                    group_indices,
                )
            candidate_scales = candidate_scales.half().float()
            key = f"groups{group_indices.numel()}_{scale_mode}"
            with torch.no_grad():
                matrix.committed_codes.copy_(candidate_codes)
                matrix.group_scale.copy_(candidate_scales)
            selection_metrics = evaluate_domains(model, selection_gates, args.device)
            selection_ratios = ratios(selection_metrics, selection_baseline)
            statistics = candidate_statistics(
                source_codes,
                source_scales,
                candidate_codes,
                candidate_scales,
                matrix.committed_mask,
                chosen_scores,
            )
            candidate_results[key] = {
                "group_budget": budget,
                "selected_groups": int(group_indices.numel()),
                "scale_mode": scale_mode,
                "selection_metrics": selection_metrics,
                "selection_ratios": selection_ratios,
                "selection_worst_ratio": max(selection_ratios.values()),
                "representation": statistics,
            }
            candidates[key] = (candidate_codes.cpu(), candidate_scales.cpu())
            print(key, selection_ratios, statistics, flush=True)
            with torch.no_grad():
                matrix.committed_codes.copy_(source_codes)
                matrix.group_scale.copy_(source_scales)

    chosen = min(
        candidates,
        key=lambda key: candidate_results[key]["selection_worst_ratio"],
    )
    chosen_codes, chosen_scales = candidates[chosen]
    with torch.no_grad():
        matrix.committed_codes.copy_(chosen_codes.to(args.device))
        matrix.group_scale.copy_(chosen_scales.to(args.device))
    confirmation_metrics = evaluate_domains(model, confirmation_gates, args.device)
    confirmation_ratios = ratios(confirmation_metrics, confirmation_baseline)
    selection_improvement = max(source_selection_ratios.values()) - max(
        candidate_results[chosen]["selection_ratios"].values()
    )
    confirmation_improvement = max(source_confirmation_ratios.values()) - max(
        confirmation_ratios.values()
    )
    code_churn = candidate_results[chosen]["representation"]["code_churn"]
    passed = (
        selection_improvement >= args.min_selection_improvement
        and confirmation_improvement >= args.min_confirmation_improvement
        and code_churn <= args.max_code_churn
    )
    artifact_path = WORKSPACE / f"wal2/artifacts/wal-tat-{args.tag}.pt"
    artifact_sha256 = None
    if passed:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "wal-tat-ternary-recode-v1",
                "source_checkpoint_sha256": sha256_file(source_path),
                "suite_sha256": sha256_file(suite_path),
                "target_name": target_name,
                "candidate": chosen,
                "recipe": "boundary_gradient_adjacent_recode",
                "residual_gain": 1.0,
                "committed_mask": matrix.committed_mask.cpu(),
                "ternary_codes_int8": chosen_codes.to(torch.int8),
                "scales_fp16": chosen_scales.half(),
                "selection_ratios": candidate_results[chosen]["selection_ratios"],
                "confirmation_ratios": confirmation_ratios,
                "representation": candidate_results[chosen]["representation"],
            },
            artifact_path,
        )
        artifact_sha256 = sha256_file(artifact_path)

    with torch.no_grad():
        matrix.committed_codes.copy_(source_codes)
        matrix.group_scale.copy_(source_scales)
    source_final = evaluate_domains(model, confirmation_gates, args.device)
    final_restoration_exact = all(
        source_final[domain]["nll"] == confirmation_source[domain]["nll"]
        for domain in confirmation_gates
    )
    result = {
        "schema": "wal-tat-boundary-sparse-recode-v1",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": "development/model selection; disjoint confirmation slice; not final audit",
        "target_name": target_name,
        "teacher_source": "target_counterfactual",
        "calibration_sequences": args.calibration_sequences,
        "selection_slice": [args.selection_start, args.selection_count],
        "confirmation_slice": [args.confirmation_start, args.confirmation_count],
        "group_budgets": list(budgets),
        "scale_modes": list(scale_modes),
        "loss_weights": {
            "ce": args.ce_weight,
            "block": args.block_weight,
            "kd": args.kd_weight,
        },
        "source_selection_ratios": source_selection_ratios,
        "source_confirmation_ratios": source_confirmation_ratios,
        "candidate_results": candidate_results,
        "chosen_candidate": chosen,
        "confirmation_metrics": confirmation_metrics,
        "confirmation_ratios": confirmation_ratios,
        "selection_worst_ratio_improvement": selection_improvement,
        "confirmation_worst_ratio_improvement": confirmation_improvement,
        "min_selection_improvement": args.min_selection_improvement,
        "min_confirmation_improvement": args.min_confirmation_improvement,
        "max_code_churn": args.max_code_churn,
        "passed": passed,
        "artifact": str(artifact_path) if passed else None,
        "artifact_sha256": artifact_sha256,
        "coverage_changed": False,
        "checkpoint_written": False,
        "source_restoration_exact": source_restoration_exact,
        "final_restoration_exact": final_restoration_exact,
        "gradient": {
            "mean_abs": float(accumulated_gradient.abs().mean().item()),
            "max_abs": float(accumulated_gradient.abs().max().item()),
            "l2": float(accumulated_gradient.norm().item()),
            "history": gradient_history,
        },
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else None
        ),
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output}; chosen={chosen} passed={passed} "
        f"selection_improvement={selection_improvement} "
        f"confirmation_improvement={confirmation_improvement}",
        flush=True,
    )
    if not final_restoration_exact:
        raise SystemExit(2)
    del model, matrices, probe, teacher, accumulated_gradient, candidates
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
