"""Recover a strict-ternary MLP overlay through its uncommitted BF16 groups.

Committed groups remain frozen hard g128 ternary codes.  Gradients are routed
only through the still-uncommitted sibling MLP groups, allowing them to absorb
the behavioral error before the next coverage-growth wave.  Selection and
confirmation are disjoint development slices.  A passing run emits a frozen
overlay/compensation artifact and never mutates the source checkpoint.
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
    sha256_file,
    tensor_output,
)
from counterfactual_bf16_restore_audit import (
    restore_original_committed_groups,
    restore_source_state,
)
from counterfactual_scale_recovery import slice_domain_gates
from matched_bf16_residual_recovery import cache_teacher, normalized_mse
from ternary_recode_artifact_audit import artifact_matrix_entries
from wal_tat.quantization import padded_grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--starting-artifact",
        type=Path,
        help=(
            "Optional frozen strict-ternary overlay. When omitted, derive the "
            "three target MLP entries exactly from the source checkpoint."
        ),
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-layer", type=int, default=24)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ce-weight", type=float, default=0.25)
    parser.add_argument("--block-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--delta-anchor-weight", type=float, default=0.01)
    parser.add_argument("--kd-topk", type=int, default=64)
    parser.add_argument("--kd-stride", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--selection-gate-start", type=int, default=0)
    parser.add_argument("--selection-gate-sequences", type=int, default=128)
    parser.add_argument("--confirmation-gate-start", type=int, default=128)
    parser.add_argument("--confirmation-gate-sequences", type=int, default=128)
    parser.add_argument("--selection-every", type=int, default=128)
    parser.add_argument("--min-selection-improvement", type=float, default=1e-3)
    parser.add_argument("--min-confirmation-improvement", type=float, default=1e-3)
    parser.add_argument("--max-relative-delta", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=277)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def validate_disjoint_slices(args: argparse.Namespace) -> None:
    selection = set(
        range(
            args.selection_gate_start,
            args.selection_gate_start + args.selection_gate_sequences,
        )
    )
    confirmation = set(
        range(
            args.confirmation_gate_start,
            args.confirmation_gate_start + args.confirmation_gate_sequences,
        )
    )
    if selection & confirmation:
        raise ValueError("selection and confirmation gate slices overlap")


def strict_mlp_artifact_from_checkpoint(
    source_payload: dict, source_hash: str, target_layer: int
) -> dict:
    """Build an exact in-memory strict-ternary bundle from a WAL-TAT checkpoint."""
    if source_payload.get("format") not in {
        "wal-tat-multi-g128-v1",
        "wal-tat-multi-g128-v2",
    }:
        raise ValueError("unsupported source checkpoint format")
    names = tuple(
        f"model.layers.{target_layer}.mlp.{projection}"
        for projection in ("up_proj", "gate_proj", "down_proj")
    )
    matrices = source_payload.get("matrices")
    if not isinstance(matrices, dict):
        raise ValueError("source checkpoint has no matrix entries")
    entries = {}
    for name in names:
        if name not in matrices:
            raise ValueError(f"source checkpoint is missing target matrix: {name}")
        source = matrices[name]
        shape = tuple(source.get("shape", ()))
        group_size = int(source.get("group_size", 0))
        mask = source.get("committed_mask")
        codes = source.get("ternary_codes_int8")
        scales = source.get("scales_fp16")
        if len(shape) != 2 or min(shape) <= 0 or group_size <= 0:
            raise ValueError(f"invalid matrix metadata for {name}")
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError(f"invalid committed mask for {name}")
        if not isinstance(codes, torch.Tensor) or codes.dtype != torch.int8:
            raise ValueError(f"invalid ternary codes for {name}")
        if not isinstance(scales, torch.Tensor) or scales.dtype != torch.float16:
            raise ValueError(f"invalid FP16 scales for {name}")
        if tuple(codes.shape) != shape:
            raise ValueError(f"ternary code shape mismatch for {name}")
        padding = (-shape[1]) % min(group_size, shape[1])
        grouped_codes = F.pad(codes, (0, padding)).view(
            shape[0], -1, min(group_size, shape[1])
        )
        expected_group_shape = grouped_codes.shape[:2]
        if tuple(mask.shape) != expected_group_shape:
            raise ValueError(f"committed mask shape mismatch for {name}")
        if tuple(scales.shape) != expected_group_shape:
            raise ValueError(f"scale shape mismatch for {name}")
        if not set(grouped_codes.unique().tolist()) <= {-1, 0, 1}:
            raise ValueError(f"non-ternary checkpoint codes for {name}")
        if not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError(f"invalid checkpoint scales for {name}")
        entries[name] = {
            "committed_mask": mask.detach().cpu().clone(),
            "ternary_codes_int8": grouped_codes.detach().cpu().clone(),
            "scales_fp16": scales.detach().cpu().clone(),
        }
    return {
        "format": "wal-tat-ternary-recode-bundle-v1",
        "source_checkpoint_sha256": source_hash,
        "target_names": list(names),
        "matrices": entries,
        "derived_from_source_checkpoint": True,
    }


@torch.no_grad()
def apply_strict_recode_artifact(
    artifact: dict, matrices: dict, checkpoint_hash: str, device: str
) -> tuple[str, ...]:
    if artifact.get("source_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("starting artifact source checkpoint mismatch")
    entries = artifact_matrix_entries(artifact)
    for name, entry in entries.items():
        if name not in matrices:
            raise ValueError(f"starting artifact matrix is absent: {name}")
        matrix = matrices[name]
        mask = entry["committed_mask"].to(device)
        codes = entry["ternary_codes_int8"].to(device, dtype=torch.int8)
        scales = entry["scales_fp16"].to(device).float()
        if not torch.equal(mask, matrix.committed_mask):
            raise ValueError("starting artifact committed mask mismatch")
        if codes.shape != matrix.committed_codes.shape:
            raise ValueError("starting artifact code shape mismatch")
        if scales.shape != matrix.group_scale.shape:
            raise ValueError("starting artifact scale shape mismatch")
        if not set(codes.unique().tolist()) <= {-1, 0, 1}:
            raise ValueError("starting artifact contains non-ternary codes")
        if not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("starting artifact contains invalid scales")
        if not torch.equal(codes[~mask], matrix.committed_codes[~mask]):
            raise ValueError("starting artifact changes uncommitted codes")
        if not torch.equal(scales[~mask], matrix.group_scale[~mask]):
            raise ValueError("starting artifact changes uncommitted scales")
        compensated_master = entry.get("fp_master_bf16")
        if compensated_master is not None:
            if not isinstance(compensated_master, torch.Tensor):
                raise ValueError("starting artifact master is not a tensor")
            if compensated_master.dtype != torch.bfloat16:
                raise ValueError("starting artifact master must be BF16")
            if compensated_master.shape != matrix.master_weight.shape:
                raise ValueError("starting artifact master shape mismatch")
            if not torch.isfinite(compensated_master).all():
                raise ValueError("starting artifact master contains non-finite values")
            candidate_master = compensated_master.to(device).float()
            candidate_grouped, _, _ = padded_grouped(
                candidate_master, matrix.group_size
            )
            current_grouped = grouped_master(matrix)
            if not torch.equal(candidate_grouped[mask], current_grouped[mask]):
                raise ValueError("starting artifact master changes committed groups")
            matrix.master_weight.copy_(candidate_master)
        matrix.committed_codes.copy_(codes)
        matrix.group_scale.copy_(scales)
    return tuple(entries)


@torch.no_grad()
def snapshot_masters(matrices: dict[str, object]) -> dict[str, torch.Tensor]:
    return {
        name: matrix.master_weight.detach().bfloat16().cpu().clone()
        for name, matrix in matrices.items()
    }


@torch.no_grad()
def restore_masters(matrices: dict[str, object], state: dict[str, torch.Tensor]) -> None:
    for name, matrix in matrices.items():
        matrix.master_weight.copy_(state[name].to(matrix.master_weight.device).float())


def grouped_master(matrix) -> torch.Tensor:
    grouped, _, _ = padded_grouped(matrix.master_weight, matrix.group_size)
    return grouped


@torch.no_grad()
def fallback_change_statistics(
    matrix,
    source_weight: torch.Tensor,
    candidate_weight: torch.Tensor,
) -> dict:
    source, _, _ = padded_grouped(source_weight.float(), matrix.group_size)
    candidate, _, _ = padded_grouped(candidate_weight.float(), matrix.group_size)
    fallback = ~matrix.committed_mask.cpu()
    committed = matrix.committed_mask.cpu()
    delta = candidate - source
    active_delta = delta[fallback]
    active_source = source[fallback]
    denominator = active_source.abs().mean().clamp_min(1e-12)
    committed_unchanged = torch.equal(candidate[committed], source[committed])
    return {
        "fallback_groups": int(fallback.sum().item()),
        "fallback_weights": int(active_delta.numel()),
        "changed_fallback_weights": int((active_delta != 0).sum().item()),
        "absolute_delta_mean": float(active_delta.abs().mean().item()),
        "absolute_delta_p95": float(
            torch.quantile(active_delta.abs().float(), 0.95).item()
        ),
        "absolute_delta_max": float(active_delta.abs().max().item()),
        "relative_delta_to_source_absmean": float(
            active_delta.abs().mean().div(denominator).item()
        ),
        "committed_master_values_unchanged": bool(committed_unchanged),
    }


def aggregate_relative_delta(statistics: dict[str, dict]) -> float:
    weighted_source_denominator = sum(
        item["fallback_weights"] for item in statistics.values()
    )
    # This is a weighted mean of each matrix-normalized relative delta.  Keep
    # it explicit because matrices have different native weight scales.
    return sum(
        item["relative_delta_to_source_absmean"] * item["fallback_weights"]
        for item in statistics.values()
    ) / max(weighted_source_denominator, 1)


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.selection_every < 1 or args.lr <= 0:
        raise ValueError("steps, selection interval and learning rate must be positive")
    if args.max_relative_delta < 0:
        raise ValueError("max-relative-delta must be non-negative")
    validate_disjoint_slices(args)
    source_path = args.source_checkpoint.resolve()
    source_hash = sha256_file(source_path)
    suite_path = args.suite.resolve()
    suite_hash = sha256_file(suite_path)
    model_path = (args.model_path or default_model_path()).resolve()
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = suite["calibration"]
    selection_gates = slice_domain_gates(
        suite["gates"],
        start=args.selection_gate_start,
        count=args.selection_gate_sequences,
        role="selection",
    )
    confirmation_gates = slice_domain_gates(
        suite["gates"],
        start=args.confirmation_gate_start,
        count=args.confirmation_gate_sequences,
        role="confirmation",
    )
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if args.starting_artifact is None:
        artifact_path = source_path
        artifact_hash = source_hash
        starting_artifact = strict_mlp_artifact_from_checkpoint(
            source_payload, source_hash, args.target_layer
        )
        starting_artifact_role = "derived_exactly_from_source_checkpoint"
    else:
        artifact_path = args.starting_artifact.resolve()
        artifact_hash = sha256_file(artifact_path)
        starting_artifact = torch.load(
            artifact_path, map_location="cpu", weights_only=False
        )
        starting_artifact_role = "external_frozen_artifact"
    artifact_entries = artifact_matrix_entries(starting_artifact)
    expected_prefix = f"model.layers.{args.target_layer}.mlp."
    target_names = tuple(
        name for name in artifact_entries if name.startswith(expected_prefix)
    )
    if set(target_names) != {
        f"{expected_prefix}up_proj",
        f"{expected_prefix}gate_proj",
        f"{expected_prefix}down_proj",
    }:
        raise ValueError("starting artifact must contain all three target MLP projections")

    seed_everything(args.seed)
    started = time.time()
    model = load_model(model_path, args.device)
    original_weights = {
        name: model.get_submodule(name).weight.detach().cpu().clone()
        for name in target_names
    }
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    confirmation_baseline = evaluate_domains(model, confirmation_gates, args.device)
    matrices = install_checkpoint(model, source_payload, args.device)
    artifact_names = apply_strict_recode_artifact(
        starting_artifact, matrices, source_hash, args.device
    )
    if set(artifact_names) != set(target_names):
        raise ValueError("starting artifact contains out-of-scope matrices")
    active = {name: matrices[name] for name in target_names}
    source_codes = {
        name: matrix.committed_codes.detach().clone() for name, matrix in active.items()
    }
    source_scales = {
        name: matrix.group_scale.detach().clone() for name, matrix in active.items()
    }
    source_masks = {
        name: matrix.committed_mask.detach().clone() for name, matrix in active.items()
    }
    source_masters = snapshot_masters(active)
    source_grouped = {
        name: grouped_master(matrix).detach().clone()
        for name, matrix in active.items()
    }
    selection_source = evaluate_domains(model, selection_gates, args.device)
    confirmation_source = evaluate_domains(model, confirmation_gates, args.device)
    source_selection_ratios = ratios(selection_source, selection_baseline)
    source_confirmation_ratios = ratios(confirmation_source, confirmation_baseline)

    # A local counterfactual teacher keeps the complete accepted frontier and
    # its existing BF16 compensation fixed, restoring only the committed MLP
    # groups whose error this stage is meant to absorb.
    for name, matrix in active.items():
        restore_original_committed_groups(matrix, original_weights[name])
    teacher_selection = evaluate_domains(model, selection_gates, args.device)
    teacher_confirmation = evaluate_domains(model, confirmation_gates, args.device)
    teacher = cache_teacher(model, calibration, args.target_layer, args.device, args)
    restore_source_state(model, matrices, source_payload, args.device)
    apply_strict_recode_artifact(starting_artifact, matrices, source_hash, args.device)
    source_recheck = evaluate_domains(model, confirmation_gates, args.device)
    source_restoration_exact = all(
        source_recheck[domain]["nll"] == confirmation_source[domain]["nll"]
        for domain in confirmation_gates
    )
    if not source_restoration_exact:
        raise RuntimeError("source did not restore exactly after teacher caching")

    opened = []
    try:
        for name, matrix in active.items():
            fallback_mask = ~matrix.committed_mask
            matrix.begin_continuous_compensation(fallback_mask)
            matrix.master_weight.requires_grad_(True)
            opened.append(name)
    except Exception:
        for name in reversed(opened):
            active[name].rollback_continuous_compensation()
        raise
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = []
    for matrix in active.values():
        matrix.master_weight.requires_grad_(True)
        parameters.append(matrix.master_weight)

    capture: dict[str, torch.Tensor] = {}
    handle = model.model.layers[args.target_layer].register_forward_hook(
        lambda _module, _inputs, output: capture.__setitem__(
            "block", tensor_output(output)
        )
    )
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    initial_state = snapshot_masters(active)
    initial_objective = max(source_selection_ratios.values())
    best_objective = initial_objective
    best_step = 0
    best_selection_ratios = source_selection_ratios
    best_state = initial_state
    history = []
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    try:
        for step in range(1, args.steps + 1):
            item = (step - 1) % len(calibration)
            batch = calibration[item].unsqueeze(0).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            capture.clear()
            logits = model(input_ids=batch[:, :-1], use_cache=False).logits
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                batch[:, 1:].reshape(-1),
            )
            block_loss = normalized_mse(capture["block"], teacher[item]["block"])
            kd_loss = bucket_kl(logits, teacher[item], args)
            anchor_terms = []
            for name, matrix in active.items():
                current = grouped_master(matrix)
                mask = ~matrix.committed_mask
                delta = current[mask] - source_grouped[name][mask]
                denominator = source_grouped[name][mask].square().mean().clamp_min(1e-12)
                anchor_terms.append(delta.square().mean() / denominator)
            anchor_loss = torch.stack(anchor_terms).mean()
            loss = (
                args.ce_weight * ce
                + args.block_weight * block_loss
                + args.kd_weight * kd_loss
                + args.delta_anchor_weight * anchor_loss
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            selection_metrics = None
            selection_ratios = None
            if step % args.selection_every == 0 or step == args.steps:
                selection_metrics = evaluate_domains(
                    model, selection_gates, args.device
                )
                selection_ratios = ratios(selection_metrics, selection_baseline)
                objective = max(selection_ratios.values())
                if objective < best_objective:
                    best_objective = objective
                    best_step = step
                    best_selection_ratios = selection_ratios
                    best_state = snapshot_masters(active)
            if (
                step == 1
                or step == args.steps
                or step % max(args.steps // 4, 1) == 0
            ):
                entry = {
                    "step": step,
                    "ce": float(ce.item()),
                    "block_loss": float(block_loss.item()),
                    "kd_loss": float(kd_loss.item()),
                    "anchor_loss": float(anchor_loss.item()),
                    "gradient_norm": float(gradient_norm),
                    "selection_ratios": selection_ratios,
                    "best_step": best_step,
                }
                history.append(entry)
                print(
                    f"step={step}/{args.steps} best={best_step} "
                    f"block={entry['block_loss']:.6f} kd={entry['kd_loss']:.6f} "
                    f"ratios={selection_ratios}",
                    flush=True,
                )
    finally:
        handle.remove()

    restore_masters(active, best_state)
    attempted_selection = evaluate_domains(model, selection_gates, args.device)
    attempted_confirmation = evaluate_domains(model, confirmation_gates, args.device)
    attempted_selection_ratios = ratios(attempted_selection, selection_baseline)
    attempted_confirmation_ratios = ratios(
        attempted_confirmation, confirmation_baseline
    )
    selection_improvement = max(source_selection_ratios.values()) - max(
        attempted_selection_ratios.values()
    )
    confirmation_improvement = max(source_confirmation_ratios.values()) - max(
        attempted_confirmation_ratios.values()
    )
    candidate_masters = snapshot_masters(active)
    change_statistics = {
        name: fallback_change_statistics(
            matrix, source_masters[name], candidate_masters[name]
        )
        for name, matrix in active.items()
    }
    relative_delta = aggregate_relative_delta(change_statistics)
    committed_master_unchanged = all(
        item["committed_master_values_unchanged"]
        for item in change_statistics.values()
    )
    codes_unchanged = all(
        torch.equal(matrix.committed_codes, source_codes[name])
        for name, matrix in active.items()
    )
    scales_unchanged = all(
        torch.equal(matrix.group_scale, source_scales[name])
        for name, matrix in active.items()
    )
    masks_unchanged = all(
        torch.equal(matrix.committed_mask, source_masks[name])
        for name, matrix in active.items()
    )
    passed = (
        selection_improvement >= args.min_selection_improvement
        and confirmation_improvement >= args.min_confirmation_improvement
        and relative_delta <= args.max_relative_delta
        and committed_master_unchanged
        and codes_unchanged
        and scales_unchanged
        and masks_unchanged
    )

    output_artifact = WORKSPACE / f"wal2/artifacts/wal-tat-{args.tag}.pt"
    output_artifact_sha256 = None
    if passed:
        output_artifact.parent.mkdir(parents=True, exist_ok=True)
        recode_entries = artifact_matrix_entries(starting_artifact)
        torch.save(
            {
                "format": "wal-tat-ternary-fallback-compensation-v1",
                "source_checkpoint_sha256": source_hash,
                "starting_artifact_sha256": artifact_hash,
                "starting_artifact_role": starting_artifact_role,
                "suite_sha256": suite_hash,
                "target_layer": args.target_layer,
                "target_names": list(target_names),
                "recipe": "frozen_ternary_mlp_uncommitted_bf16_qat",
                "matrices": {
                    name: {
                        "committed_mask": recode_entries[name]["committed_mask"],
                        "ternary_codes_int8": recode_entries[name]["ternary_codes_int8"],
                        "scales_fp16": recode_entries[name]["scales_fp16"],
                        "fp_master_bf16": candidate_masters[name],
                    }
                    for name in target_names
                },
                "selection_ratios": attempted_selection_ratios,
                "confirmation_ratios": attempted_confirmation_ratios,
                "fallback_change": change_statistics,
                "aggregate_relative_delta": relative_delta,
            },
            output_artifact,
        )
        output_artifact_sha256 = sha256_file(output_artifact)

    for name in reversed(opened):
        active[name].rollback_continuous_compensation()
        active[name].master_weight.requires_grad_(False)
    final_source = evaluate_domains(model, confirmation_gates, args.device)
    final_restoration_exact = all(
        final_source[domain]["nll"] == confirmation_source[domain]["nll"]
        for domain in confirmation_gates
    )
    result = {
        "schema": "wal-tat-mlp-fallback-compensation-recovery-v1",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_hash,
        "starting_artifact": str(artifact_path),
        "starting_artifact_sha256": artifact_hash,
        "starting_artifact_role": starting_artifact_role,
        "suite": str(suite_path),
        "suite_sha256": suite_hash,
        "suite_role": "development training, selection and disjoint confirmation; never audit",
        "target_layer": args.target_layer,
        "target_names": list(target_names),
        "teacher_source": "target_counterfactual_mlp",
        "teacher_selection_ratios": ratios(
            teacher_selection, selection_baseline
        ),
        "teacher_confirmation_ratios": ratios(
            teacher_confirmation, confirmation_baseline
        ),
        "source_restoration_exact": source_restoration_exact,
        "steps": args.steps,
        "lr": args.lr,
        "loss_weights": {
            "ce": args.ce_weight,
            "block": args.block_weight,
            "kd": args.kd_weight,
            "delta_anchor": args.delta_anchor_weight,
        },
        "selection_slice": [
            args.selection_gate_start,
            args.selection_gate_sequences,
        ],
        "confirmation_slice": [
            args.confirmation_gate_start,
            args.confirmation_gate_sequences,
        ],
        "source_selection_ratios": source_selection_ratios,
        "source_confirmation_ratios": source_confirmation_ratios,
        "best_step": best_step,
        "best_selection_ratios": best_selection_ratios,
        "attempted_selection_ratios": attempted_selection_ratios,
        "attempted_confirmation_ratios": attempted_confirmation_ratios,
        "selection_worst_ratio_improvement": selection_improvement,
        "confirmation_worst_ratio_improvement": confirmation_improvement,
        "min_selection_improvement": args.min_selection_improvement,
        "min_confirmation_improvement": args.min_confirmation_improvement,
        "max_relative_delta": args.max_relative_delta,
        "fallback_change": change_statistics,
        "aggregate_relative_delta": relative_delta,
        "committed_master_values_unchanged": committed_master_unchanged,
        "ternary_codes_unchanged": codes_unchanged,
        "scales_unchanged": scales_unchanged,
        "committed_masks_unchanged": masks_unchanged,
        "codes_strict_ternary": all(
            set(matrix.committed_codes.unique().tolist()) <= {-1, 0, 1}
            for matrix in active.values()
        ),
        "persistent_bf16_residual_on_committed_groups": False,
        "coverage_changed": False,
        "passed": passed,
        "artifact": str(output_artifact) if passed else None,
        "artifact_sha256": output_artifact_sha256,
        "checkpoint_written": False,
        "final_restoration_exact": final_restoration_exact,
        "history": history,
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
        f"wrote {output}; passed={passed} best_step={best_step} "
        f"selection_improvement={selection_improvement} "
        f"confirmation_improvement={confirmation_improvement} "
        f"relative_delta={relative_delta}",
        flush=True,
    )
    if not final_restoration_exact:
        raise SystemExit(2)
    del model, matrices, teacher, best_state, candidate_masters, original_weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
