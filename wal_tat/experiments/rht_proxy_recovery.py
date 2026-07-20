"""Development-only hard-forward recovery for one fixed RHT ternary matrix.

The source frontier is never mutated.  Hyperparameters and the RHT seed are
selected only on the supplied development suite.  The output is a small
standalone transform artifact that must pass fresh cumulative BF16 audits
before it can be considered for a checkpoint/artifact-format commit.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from compensation_window import (
    PROJECT,
    WORKSPACE,
    default_model_path,
    install_checkpoint,
    load_model,
    log,
    seed_everything,
    set_submodule,
    sha256_file,
    tensor_output,
)
from wal_tat import (
    FixedTernaryLinear,
    ProxyTernaryMatrix,
    TransformedProxyTernaryLinear,
    inverse_blockwise_randomized_hadamard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--target-name", default="model.layers.23.self_attn.q_proj")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--transform-seed", type=int, default=307)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--calibration-sequences", type=int, default=256)
    parser.add_argument("--selection-sequences-per-domain", type=int, default=128)
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
    parser.add_argument("--attention-weight", type=float, default=0.02)
    parser.add_argument("--proxy-anchor-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def target_layer_index(name: str) -> int:
    parts = name.split(".")
    if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
        raise ValueError("target-name must start with model.layers.<index>")
    return int(parts[2])


@torch.inference_mode()
def evaluate_chunks(model, chunks, device: str, batch_size: int) -> dict:
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    for start in range(0, len(chunks), batch_size):
        selected = chunks[start : start + batch_size]
        if len({int(chunk.numel()) for chunk in selected}) != 1:
            raise ValueError("evaluation chunks must have equal lengths")
        batch = torch.stack(selected).to(device)
        logits = model(input_ids=batch[:, :-1], use_cache=False).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            batch[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss.item())
        total_tokens += int(batch.shape[0] * (batch.shape[1] - 1))
    nll = total_nll / total_tokens
    return {"nll": nll, "ppl": math.exp(min(nll, 50.0)), "tokens": total_tokens}


def evaluate_domains(model, gates, device: str, batch_size: int) -> dict:
    return {
        domain: evaluate_chunks(model, chunks, device, batch_size)
        for domain, chunks in gates.items()
    }


def nll_ratios(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        domain: candidate[domain]["nll"] / baseline[domain]["nll"]
        for domain in baseline
    }


def objective(ratios: dict[str, float]) -> tuple[float, float]:
    return max(ratios.values()), sum(ratios.values()) / len(ratios)


def normalized_mse(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    target = teacher.detach().float()
    return F.mse_loss(student.float(), target) / target.square().mean().clamp_min(1e-8)


def token_kl(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    tau = max(float(temperature), 1e-4)
    student_log = F.log_softmax(student.float() / tau, dim=-1)
    teacher_prob = F.softmax(teacher.float() / tau, dim=-1)
    token_count = student.shape[0] * student.shape[1]
    return (
        F.kl_div(student_log, teacher_prob, reduction="sum")
        * (tau * tau)
        / token_count
    )


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    checkpoint = args.checkpoint.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = suite["calibration"][: args.calibration_sequences]
    if not calibration:
        raise ValueError("calibration set is empty")
    selection_gates = {
        domain: chunks[: args.selection_sequences_per_domain]
        for domain, chunks in suite["gates"].items()
    }
    full_gates = suite["gates"]
    if any(not chunks for chunks in selection_gates.values()):
        raise ValueError("selection suite contains an empty domain")

    seed_everything(args.seed)
    started = time.monotonic()
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    teacher = load_model(model_path, args.device)
    bf16_selection = evaluate_domains(
        teacher, selection_gates, args.device, args.batch_size
    )
    bf16_full = evaluate_domains(teacher, full_gates, args.device, args.batch_size)
    install_checkpoint(teacher, source_payload, args.device)
    frontier_selection = evaluate_domains(
        teacher, selection_gates, args.device, args.batch_size
    )
    frontier_full = evaluate_domains(teacher, full_gates, args.device, args.batch_size)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()

    student = load_model(model_path, args.device)
    install_checkpoint(student, source_payload, args.device)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    target = student.get_submodule(args.target_name)
    if not isinstance(target, nn.Linear):
        raise TypeError("target must be an untouched BF16 nn.Linear")
    reference_weight = target.weight.detach().clone()
    reference_bias = None if target.bias is None else target.bias.detach().clone()
    fixed = FixedTernaryLinear.from_weight(
        reference_weight,
        group_size=args.group_size,
        transform="rht",
        transform_seed=args.transform_seed,
        bias=reference_bias,
    )
    proxy = ProxyTernaryMatrix(
        fixed.ternary_codes,
        fixed.group_scales,
        compute_dtype=reference_weight.dtype,
        temperature=args.initial_temperature,
    ).to(args.device)
    initial_codes = proxy.hard_codes().cpu()
    set_submodule(
        student,
        args.target_name,
        TransformedProxyTernaryLinear(
            proxy,
            transform="rht",
            transform_seed=args.transform_seed,
            bias=reference_bias,
        ).to(args.device),
    )
    student.eval()

    initial_selection = evaluate_domains(
        student, selection_gates, args.device, args.batch_size
    )
    initial_to_bf16 = nll_ratios(initial_selection, bf16_selection)
    initial_to_frontier = nll_ratios(initial_selection, frontier_selection)

    layer_index = target_layer_index(args.target_name)
    teacher_capture: dict[str, torch.Tensor] = {}
    student_capture: dict[str, torch.Tensor] = {}
    handles = [
        teacher.model.layers[layer_index].register_forward_hook(
            lambda _module, _inputs, output: teacher_capture.__setitem__(
                "block", tensor_output(output)
            )
        ),
        teacher.model.layers[layer_index].self_attn.register_forward_hook(
            lambda _module, _inputs, output: teacher_capture.__setitem__(
                "attention", tensor_output(output)
            )
        ),
        student.model.layers[layer_index].register_forward_hook(
            lambda _module, _inputs, output: student_capture.__setitem__(
                "block", tensor_output(output)
            )
        ),
        student.model.layers[layer_index].self_attn.register_forward_hook(
            lambda _module, _inputs, output: student_capture.__setitem__(
                "attention", tensor_output(output)
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
                teacher_logits = teacher(
                    input_ids=batch[:, :-1], use_cache=False
                ).logits
            optimizer.zero_grad(set_to_none=True)
            student_logits = student(
                input_ids=batch[:, :-1], use_cache=False
            ).logits
            ce = F.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]).float(),
                batch[:, 1:].reshape(-1),
            )
            kd = token_kl(student_logits, teacher_logits, args.kd_temperature)
            block_loss = normalized_mse(
                student_capture["block"], teacher_capture["block"]
            )
            attention_loss = normalized_mse(
                student_capture["attention"], teacher_capture["attention"]
            )
            anchor_loss = proxy.proxy_anchor_loss()
            loss = (
                args.ce_weight * ce
                + args.kd_weight * kd
                + args.block_weight * block_loss
                + args.attention_weight * attention_loss
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
                selection_metrics = evaluate_domains(
                    student, selection_gates, args.device, args.batch_size
                )
                ratios_to_bf16 = nll_ratios(selection_metrics, bf16_selection)
                ratios_to_frontier = nll_ratios(
                    selection_metrics, frontier_selection
                )
                current_objective = objective(ratios_to_bf16)
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_step = step
                    best_state = snapshot()
                    best_metrics = selection_metrics
                    best_to_bf16 = ratios_to_bf16
                    best_to_frontier = ratios_to_frontier
            if step == 1 or should_select or step % max(args.steps // 4, 1) == 0:
                entry = {
                    "step": step,
                    "loss": float(loss.item()),
                    "ce": float(ce.item()),
                    "kd": float(kd.item()),
                    "block_loss": float(block_loss.item()),
                    "attention_loss": float(attention_loss.item()),
                    "anchor_loss": float(anchor_loss.item()),
                    "temperature": proxy.temperature,
                    "gradient_norm": float(gradient_norm),
                    "code_churn": proxy.code_churn(),
                    "selection_metrics": selection_metrics,
                    "ratios_to_bf16": ratios_to_bf16,
                    "ratios_to_frontier": ratios_to_frontier,
                    "best_step": best_step,
                }
                history.append(entry)
                log(
                    f"step={step}/{args.steps} loss={loss.item():.5f} "
                    f"ce={ce.item():.5f} kd={kd.item():.5f} "
                    f"churn={proxy.code_churn():.6f} best={best_step}"
                )
    finally:
        for handle in handles:
            handle.remove()

    with torch.no_grad():
        proxy.proxy_code.copy_(best_state["codes"].to(args.device).float())
        proxy.group_scale.copy_(best_state["scales"].to(args.device))
    student.eval()
    final_full = evaluate_domains(student, full_gates, args.device, args.batch_size)
    final_full_to_bf16 = nll_ratios(final_full, bf16_full)
    final_full_to_frontier = nll_ratios(final_full, frontier_full)
    hard_codes = proxy.hard_codes().cpu()
    hard_scales = proxy.group_scale.detach().abs().clamp_min(1e-5).cpu()
    transformed_weight = (
        hard_codes.float() * hard_scales.unsqueeze(-1)
    ).reshape(reference_weight.shape)
    effective_weight = inverse_blockwise_randomized_hadamard(
        transformed_weight,
        group_size=args.group_size,
        seed=args.transform_seed,
    )
    reference_cpu = reference_weight.float().cpu()
    weight_mse = float(F.mse_loss(effective_weight, reference_cpu).item())
    relative_weight_mse = weight_mse / float(
        reference_cpu.square().mean().clamp_min(1e-12).item()
    )
    values, counts = torch.unique(hard_codes, return_counts=True)
    histogram = {-1: 0, 0: 0, 1: 0}
    histogram.update({int(value): int(count) for value, count in zip(values, counts)})

    artifact_dir = WORKSPACE / "wal2/artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"wal-tat-{args.tag}.pt"
    artifact = {
        "schema": "wal-tat-rht-ternary-artifact-v1",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "model": str(model_path),
        "target_name": args.target_name,
        "target_shape": list(reference_weight.shape),
        "group_size": args.group_size,
        "transform": "rht",
        "transform_seed": args.transform_seed,
        "codes": hard_codes.to(torch.int8),
        "scales": hard_scales.half(),
        "compute_dtype": str(reference_weight.dtype),
        "bias": None if reference_bias is None else reference_bias.cpu(),
        "development_only": True,
        "best_step": best_step,
    }
    torch.save(artifact, artifact_path)

    result = {
        "schema": "wal-tat-rht-proxy-recovery-v1",
        "tag": args.tag,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": "development-only; hyperparameter and step selection",
        "model": str(model_path),
        "target_name": args.target_name,
        "group_size": args.group_size,
        "transform_seed": args.transform_seed,
        "steps": args.steps,
        "calibration_sequences": len(calibration),
        "selection_sequences_per_domain": args.selection_sequences_per_domain,
        "hyperparameters": {
            "proxy_lr": args.proxy_lr,
            "scale_lr": args.scale_lr,
            "initial_temperature": args.initial_temperature,
            "final_temperature": args.final_temperature,
            "ce_weight": args.ce_weight,
            "kd_weight": args.kd_weight,
            "kd_temperature": args.kd_temperature,
            "block_weight": args.block_weight,
            "attention_weight": args.attention_weight,
            "proxy_anchor_weight": args.proxy_anchor_weight,
        },
        "bf16_selection": bf16_selection,
        "frontier_selection": frontier_selection,
        "initial_selection": initial_selection,
        "initial_ratios_to_bf16": initial_to_bf16,
        "initial_ratios_to_frontier": initial_to_frontier,
        "best_step": best_step,
        "best_selection": best_metrics,
        "best_selection_ratios_to_bf16": best_to_bf16,
        "best_selection_ratios_to_frontier": best_to_frontier,
        "bf16_full_development": bf16_full,
        "frontier_full_development": frontier_full,
        "final_full_development": final_full,
        "final_full_ratios_to_bf16": final_full_to_bf16,
        "final_full_ratios_to_frontier": final_full_to_frontier,
        "initial_worst_ratio_to_bf16": objective(initial_to_bf16)[0],
        "final_worst_ratio_to_bf16": objective(final_full_to_bf16)[0],
        "initial_to_final_worst_improvement": (
            objective(initial_to_bf16)[0] - objective(final_full_to_bf16)[0]
        ),
        "code_churn": float((hard_codes != initial_codes).float().mean().item()),
        "code_histogram": {str(key): value for key, value in histogram.items()},
        "zero_fraction": histogram[0] / hard_codes.numel(),
        "weight_mse": weight_mse,
        "relative_weight_mse": relative_weight_mse,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "checkpoint_mutated": False,
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
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "history"},
            indent=2,
        )
    )

    del teacher, student, proxy, fixed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
