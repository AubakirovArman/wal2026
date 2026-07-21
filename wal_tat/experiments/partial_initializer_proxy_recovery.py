"""Development-only hard-forward recovery of a frozen partial initializer.

Only groups named by the input artifact receive proxy-code and scale gradients.
Previously committed groups and all remaining BF16 weights are bitwise frozen.
The accepted frontier checkpoint is never modified; the output is another small
artifact that still requires fresh paired validation.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from compensation_window import (
    PROJECT,
    WORKSPACE,
    default_model_path,
    install_checkpoint,
    load_model,
    seed_everything,
    set_submodule,
    sha256_file,
    tensor_output,
)
from rht_proxy_recovery import (
    evaluate_domains,
    nll_ratios,
    normalized_mse,
    objective,
    token_kl,
)
from wal_tat import ProxyTernaryLinear, ProxyTernaryMatrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--calibration-sequences", type=int, default=96)
    parser.add_argument("--selection-sequences-per-domain", type=int, default=64)
    parser.add_argument("--selection-every", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--proxy-lr", type=float, default=1e-4)
    parser.add_argument("--scale-lr", type=float, default=5e-6)
    parser.add_argument("--initial-temperature", type=float, default=0.25)
    parser.add_argument("--final-temperature", type=float, default=0.08)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--block-weight", type=float, default=0.02)
    parser.add_argument("--proxy-anchor-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def layer_index(name: str) -> int:
    parts = name.split(".")
    if len(parts) < 4 or parts[:2] != ["model", "layers"]:
        raise ValueError("target name must start with model.layers.<index>")
    return int(parts[2])


def install_artifact(matrix, artifact, device: str):
    codes, scales = matrix.codes_scales()
    if matrix.padding:
        codes = F.pad(codes, (0, matrix.padding))
    codes = codes.view_as(matrix.committed_codes).to(device)
    scales = scales.to(device)
    candidate_mask = torch.zeros_like(matrix.committed_mask)
    flat_mask = candidate_mask.reshape(-1)
    flat_codes = codes.reshape(-1, matrix.group_size)
    flat_scales = scales.reshape(-1)
    for entry in artifact["entries"]:
        indices = entry["flat_group_indices"].to(device, dtype=torch.long)
        if torch.any(indices < 0) or torch.any(indices >= flat_mask.numel()):
            raise ValueError("artifact group index is out of bounds")
        if torch.any(matrix.committed_mask.reshape(-1).index_select(0, indices)):
            raise ValueError("artifact overlaps committed groups")
        if torch.any(flat_mask.index_select(0, indices)):
            raise ValueError("artifact entries overlap")
        entry_codes = entry["codes"].to(device, dtype=torch.int8)
        entry_scales = entry["scales"].to(device, dtype=flat_scales.dtype)
        if entry_codes.shape != (indices.numel(), matrix.group_size):
            raise ValueError("artifact code shape does not match")
        if entry_scales.shape != (indices.numel(),):
            raise ValueError("artifact scale shape does not match")
        flat_mask[indices] = True
        flat_codes[indices] = entry_codes
        flat_scales[indices] = entry_scales
    return codes, scales, candidate_mask


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.calibration_sequences < 1:
        raise ValueError("step, batch and calibration counts must be positive")
    checkpoint = args.checkpoint.resolve()
    artifact_path = args.artifact.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "wal-tat-partial-initializer-artifact-v1":
        raise ValueError("unsupported initializer artifact")
    if artifact.get("source_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("artifact source checkpoint hash does not match")
    if artifact.get("development_only") is not True:
        raise ValueError("artifact must be frozen on development")
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = suite["calibration"][: args.calibration_sequences]
    selection_gates = {
        domain: chunks[: args.selection_sequences_per_domain]
        for domain, chunks in suite["gates"].items()
    }
    full_gates = suite["gates"]
    if not calibration or any(not chunks for chunks in selection_gates.values()):
        raise ValueError("suite does not contain enough development data")

    seed_everything(args.seed)
    started = time.monotonic()
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # The teacher starts as raw BF16 for baseline measurement, then receives the
    # accepted checkpoint and remains the immutable current-frontier teacher.
    teacher = load_model(model_path, args.device)
    bf16_selection = evaluate_domains(teacher, selection_gates, args.device, args.batch_size)
    bf16_full = evaluate_domains(teacher, full_gates, args.device, args.batch_size)
    install_checkpoint(teacher, source_payload, args.device)
    frontier_selection = evaluate_domains(
        teacher, selection_gates, args.device, args.batch_size
    )
    frontier_full = evaluate_domains(teacher, full_gates, args.device, args.batch_size)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = load_model(model_path, args.device)
    matrices = install_checkpoint(student, source_payload, args.device)
    target_name = artifact["target_name"]
    if target_name not in matrices:
        raise ValueError("artifact target is absent from the checkpoint")
    matrix = matrices[target_name]
    if list(matrix.master_weight.shape) != artifact["target_shape"]:
        raise ValueError("artifact target shape does not match")
    if int(artifact["group_size"]) != matrix.group_size:
        raise ValueError("artifact group size does not match")
    codes, scales, candidate_mask = install_artifact(matrix, artifact, args.device)
    active_mask = matrix.committed_mask | candidate_mask
    target = student.get_submodule(target_name)
    proxy = ProxyTernaryMatrix(
        codes,
        scales,
        compute_dtype=matrix.compute_dtype,
        temperature=args.initial_temperature,
        committed_mask=active_mask,
        master_weight=matrix.master_weight,
    ).to(args.device)
    initial_codes = proxy.hard_codes().detach().clone()
    bias = target.bias
    set_submodule(student, target_name, ProxyTernaryLinear(proxy, bias).to(args.device))
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    proxy.proxy_code.requires_grad_(True)
    proxy.group_scale.requires_grad_(True)

    code_gradient_mask = candidate_mask.unsqueeze(-1).to(torch.float32)
    scale_gradient_mask = candidate_mask.to(torch.float32)
    gradient_hooks = [
        proxy.proxy_code.register_hook(lambda gradient: gradient * code_gradient_mask),
        proxy.group_scale.register_hook(lambda gradient: gradient * scale_gradient_mask),
    ]

    initial_selection = evaluate_domains(student, selection_gates, args.device, args.batch_size)
    initial_to_bf16 = nll_ratios(initial_selection, bf16_selection)
    initial_to_frontier = nll_ratios(initial_selection, frontier_selection)
    block_index = layer_index(target_name)
    teacher_capture: dict[str, torch.Tensor] = {}
    student_capture: dict[str, torch.Tensor] = {}
    handles = [
        teacher.model.layers[block_index].register_forward_hook(
            lambda _module, _inputs, output: teacher_capture.__setitem__(
                "block", tensor_output(output)
            )
        ),
        student.model.layers[block_index].register_forward_hook(
            lambda _module, _inputs, output: student_capture.__setitem__(
                "block", tensor_output(output)
            )
        ),
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": [proxy.proxy_code], "lr": args.proxy_lr},
            {"params": [proxy.group_scale], "lr": args.scale_lr},
        ],
        weight_decay=0.0,
    )

    def snapshot() -> dict:
        return {
            "codes": proxy.hard_codes().cpu(),
            "scales": proxy.group_scale.detach().cpu().clone(),
        }

    best_state = snapshot()
    best_step = 0
    best_metrics = initial_selection
    best_to_bf16 = initial_to_bf16
    best_to_frontier = initial_to_frontier
    best_objective = objective(best_to_bf16)
    history = []
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    try:
        for step in range(1, args.steps + 1):
            phase = step / args.steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * phase))
            proxy.temperature = args.final_temperature + (
                args.initial_temperature - args.final_temperature
            ) * cosine
            chunk = calibration[(step - 1) % len(calibration)]
            batch = chunk.unsqueeze(0).to(args.device)
            teacher_capture.clear()
            student_capture.clear()
            with torch.inference_mode():
                teacher_logits = teacher(input_ids=batch[:, :-1], use_cache=False).logits
            optimizer.zero_grad(set_to_none=True)
            student_logits = student(input_ids=batch[:, :-1], use_cache=False).logits
            ce = F.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]).float(),
                batch[:, 1:].reshape(-1),
            )
            kd = token_kl(student_logits, teacher_logits, args.kd_temperature)
            block_loss = normalized_mse(
                student_capture["block"], teacher_capture["block"]
            )
            anchor_loss = (
                proxy.proxy_code - initial_codes.float()
            ).square()[candidate_mask.unsqueeze(-1).expand_as(proxy.proxy_code)].mean()
            loss = (
                args.ce_weight * ce
                + args.kd_weight * kd
                + args.block_weight * block_loss
                + args.proxy_anchor_weight * anchor_loss
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [proxy.proxy_code, proxy.group_scale], 1.0
            )
            optimizer.step()
            proxy.constrain_()

            should_select = step % args.selection_every == 0 or step == args.steps
            selection_metrics = None
            ratios_to_bf16 = None
            ratios_to_frontier = None
            if should_select:
                student.eval()
                selection_metrics = evaluate_domains(
                    student, selection_gates, args.device, args.batch_size
                )
                ratios_to_bf16 = nll_ratios(selection_metrics, bf16_selection)
                ratios_to_frontier = nll_ratios(selection_metrics, frontier_selection)
                current_objective = objective(ratios_to_bf16)
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_step = step
                    best_state = snapshot()
                    best_metrics = selection_metrics
                    best_to_bf16 = ratios_to_bf16
                    best_to_frontier = ratios_to_frontier
                student.train()
            if step == 1 or should_select or step % max(args.steps // 4, 1) == 0:
                entry = {
                    "step": step,
                    "loss": float(loss.item()),
                    "ce": float(ce.item()),
                    "kd": float(kd.item()),
                    "block_loss": float(block_loss.item()),
                    "anchor_loss": float(anchor_loss.item()),
                    "temperature": proxy.temperature,
                    "gradient_norm": float(gradient_norm),
                    "candidate_code_churn": float(
                        (proxy.hard_codes() != initial_codes)[
                            candidate_mask.unsqueeze(-1).expand_as(initial_codes)
                        ].float().mean().item()
                    ),
                    "selection_metrics": selection_metrics,
                    "ratios_to_bf16": ratios_to_bf16,
                    "ratios_to_frontier": ratios_to_frontier,
                    "best_step": best_step,
                }
                history.append(entry)
                print(
                    f"[{time.strftime('%H:%M:%S')}] step={step}/{args.steps} "
                    f"loss={loss.item():.5f} kd={kd.item():.5f} best={best_step}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()
        for hook in gradient_hooks:
            hook.remove()

    with torch.no_grad():
        proxy.proxy_code.copy_(best_state["codes"].to(args.device).float())
        proxy.group_scale.copy_(best_state["scales"].to(args.device))
    student.eval()
    final_full = evaluate_domains(student, full_gates, args.device, args.batch_size)
    final_full_to_bf16 = nll_ratios(final_full, bf16_full)
    final_full_to_frontier = nll_ratios(final_full, frontier_full)
    hard_codes = proxy.hard_codes().cpu()
    hard_scales = proxy.group_scale.detach().abs().clamp_min(1e-5).half().cpu()
    initial_codes_cpu = initial_codes.cpu()
    candidate_mask_cpu = candidate_mask.cpu()

    recovered_entries = []
    flat_codes = hard_codes.reshape(-1, matrix.group_size)
    flat_scales = hard_scales.reshape(-1)
    for entry in artifact["entries"]:
        indices = entry["flat_group_indices"].long()
        recovered_entries.append(
            {
                **entry,
                "codes": flat_codes.index_select(0, indices).clone(),
                "scales": flat_scales.index_select(0, indices).clone(),
                "recovery_best_step": best_step,
            }
        )

    output_artifact_path = WORKSPACE / f"wal2/artifacts/wal-tat-{args.tag}.pt"
    output_artifact = {
        **artifact,
        "entries": recovered_entries,
        "development_suite": str(suite_path),
        "development_suite_sha256": sha256_file(suite_path),
        "parent_artifact": str(artifact_path),
        "parent_artifact_sha256": sha256_file(artifact_path),
        "recovery": {
            "mode": "hard-forward candidate-only proxy-code-and-scale",
            "best_step": best_step,
            "steps": args.steps,
            "norm_parameters_trained": 0,
            "previously_committed_groups_trainable": 0,
        },
    }
    torch.save(output_artifact, output_artifact_path)
    candidate_churn = float(
        (hard_codes != initial_codes_cpu)[
            candidate_mask_cpu.unsqueeze(-1).expand_as(hard_codes)
        ].float().mean().item()
    )
    result = {
        "schema": "wal-tat-partial-initializer-proxy-recovery-v1",
        "status": "development_only_recovered_artifact",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "input_artifact": str(artifact_path),
        "input_artifact_sha256": sha256_file(artifact_path),
        "output_artifact": str(output_artifact_path),
        "output_artifact_sha256": sha256_file(output_artifact_path),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "target_name": target_name,
        "candidate_groups": int(candidate_mask.sum().item()),
        "candidate_weights": int(candidate_mask.sum().item() * matrix.group_size),
        "steps": args.steps,
        "best_step": best_step,
        "hyperparameters": {
            "proxy_lr": args.proxy_lr,
            "scale_lr": args.scale_lr,
            "initial_temperature": args.initial_temperature,
            "final_temperature": args.final_temperature,
            "ce_weight": args.ce_weight,
            "kd_weight": args.kd_weight,
            "kd_temperature": args.kd_temperature,
            "block_weight": args.block_weight,
            "proxy_anchor_weight": args.proxy_anchor_weight,
        },
        "bf16_selection": bf16_selection,
        "frontier_selection": frontier_selection,
        "initial_selection": initial_selection,
        "initial_ratios_to_bf16": initial_to_bf16,
        "initial_ratios_to_frontier": initial_to_frontier,
        "best_selection": best_metrics,
        "best_selection_ratios_to_bf16": best_to_bf16,
        "best_selection_ratios_to_frontier": best_to_frontier,
        "final_full_development": final_full,
        "final_full_ratios_to_bf16": final_full_to_bf16,
        "final_full_ratios_to_frontier": final_full_to_frontier,
        "candidate_code_churn": candidate_churn,
        "norm_parameters_trained": 0,
        "previously_committed_groups_trainable": 0,
        "checkpoint_mutated": sha256_file(checkpoint) != checkpoint_hash,
        "accepted_coverage_changed": False,
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else 0
        ),
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))
    del teacher, student, matrices, proxy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
