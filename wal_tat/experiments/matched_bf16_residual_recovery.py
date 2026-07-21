"""Matched continuous-residual control over committed ternary groups."""
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
    bucket_kl,
    default_model_path,
    evaluate_domains,
    install_checkpoint,
    load_model,
    ratios,
    seed_everything,
    set_submodule,
    sha256_file,
    teacher_bucket,
    tensor_output,
)
from counterfactual_bf16_restore_audit import (
    restore_original_committed_groups,
    restore_source_state,
)


PROJECTION_MAP = {
    "up_proj": "mlp.up_proj",
    "gate_proj": "mlp.gate_proj",
    "down_proj": "mlp.down_proj",
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-layer", type=int, default=24)
    parser.add_argument("--target-projection", choices=tuple(PROJECTION_MAP), default="up_proj")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--block-weight", type=float, default=0.1)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--kd-topk", type=int, default=64)
    parser.add_argument("--kd-stride", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--residual-anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher-source",
        choices=("raw_bf16", "target_counterfactual"),
        default="raw_bf16",
    )
    parser.add_argument("--selection-gate-sequences", type=int, default=256)
    parser.add_argument("--selection-every", type=int, default=128)
    parser.add_argument("--min-improvement", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=157)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


class MaskedContinuousResidualLinear(nn.Module):
    """Add a trainable FP32 residual only on already committed groups."""

    def __init__(self, matrix, bias=None):
        super().__init__()
        self.matrix = matrix
        for parameter in self.matrix.parameters():
            parameter.requires_grad_(False)
        mask = matrix.committed_mask.unsqueeze(-1).expand_as(matrix.committed_codes)
        mask = mask.reshape(matrix.out_features, -1)[:, : matrix.in_features]
        self.register_buffer("weight_mask", mask)
        self.delta = nn.Parameter(torch.zeros_like(matrix.master_weight))
        self.bias = None if bias is None else nn.Parameter(
            bias.detach().clone(), requires_grad=False
        )
        self.in_features = matrix.in_features
        self.out_features = matrix.out_features

    def effective_weight(self) -> torch.Tensor:
        delta = torch.where(self.weight_mask, self.delta, torch.zeros_like(self.delta))
        return self.matrix.effective_weight() + delta.to(self.matrix.compute_dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.effective_weight().to(value.dtype), self.bias)

    def masked_delta(self) -> torch.Tensor:
        return self.delta[self.weight_mask]


@torch.no_grad()
def cache_teacher(model, calibration, layer_index, device, args):
    capture = {}
    handle = model.model.layers[layer_index].register_forward_hook(
        lambda _module, _inputs, output: capture.__setitem__(
            "block", tensor_output(output)
        )
    )
    result = []
    try:
        for index, chunk in enumerate(calibration):
            capture.clear()
            batch = chunk.unsqueeze(0).to(device)
            output = model(input_ids=batch[:, :-1], use_cache=False)
            entry = {
                "block": capture["block"].squeeze(0).half().cpu(),
                **teacher_bucket(output.logits, args),
            }
            result.append(entry)
            if (index + 1) % 64 == 0 or index + 1 == len(calibration):
                print(f"teacher {index + 1}/{len(calibration)}", flush=True)
    finally:
        handle.remove()
    return result


def normalized_mse(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    target = teacher.to(student.device).float().unsqueeze(0)
    return F.mse_loss(student.float(), target) / target.square().mean().clamp_min(1e-8)


@torch.no_grad()
def residual_statistics(residual, original_weight: torch.Tensor) -> dict:
    matrix = residual.matrix
    original = F.pad(
        original_weight.to(matrix.master_weight.device).float(), (0, matrix.padding)
    ).view_as(matrix.committed_codes)
    hard = matrix.committed_codes.float() * matrix.group_scale.detach().unsqueeze(-1)
    oracle = (original - hard)[matrix.committed_mask]
    learned = residual.delta.reshape(matrix.out_features, -1)
    if matrix.padding:
        learned = F.pad(learned, (0, matrix.padding))
    learned = learned.view_as(matrix.committed_codes)[matrix.committed_mask]
    learned_flat = learned.float().flatten()
    oracle_flat = oracle.float().flatten()
    learned_norm = learned_flat.norm()
    oracle_norm = oracle_flat.norm().clamp_min(1e-12)
    cosine = F.cosine_similarity(learned_flat, oracle_flat, dim=0)
    return {
        "active_weights": learned_flat.numel(),
        "learned_l2": float(learned_norm.item()),
        "oracle_l2": float(oracle_norm.item()),
        "learned_to_oracle_l2_ratio": float((learned_norm / oracle_norm).item()),
        "cosine_to_original_bf16_correction": float(cosine.item()),
        "mean_abs": float(learned_flat.abs().mean().item()),
        "max_abs": float(learned_flat.abs().max().item()),
        "nonzero_fraction": float((learned_flat != 0).float().mean().item()),
    }


def main() -> None:
    args = parse_args()
    source_checkpoint = args.source_checkpoint.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = suite["calibration"]
    gates = suite["gates"]
    selection_gates = {
        domain: chunks[: args.selection_gate_sequences]
        for domain, chunks in gates.items()
    }
    if any(len(chunks) != args.selection_gate_sequences for chunks in selection_gates.values()):
        raise ValueError("suite lacks the requested selection gate sequences")
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    target_name = (
        f"model.layers.{args.target_layer}."
        f"{PROJECTION_MAP[args.target_projection]}"
    )
    if target_name not in payload["matrices"]:
        raise ValueError(f"target matrix is absent from checkpoint: {target_name}")

    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    original_weight = model.get_submodule(target_name).weight.detach().cpu().clone()
    baseline = evaluate_domains(model, gates, args.device)
    selection_baseline = (
        baseline
        if all(len(selection_gates[name]) == len(gates[name]) for name in gates)
        else evaluate_domains(model, selection_gates, args.device)
    )
    if args.teacher_source == "raw_bf16":
        teacher_metrics = baseline
        teacher = cache_teacher(
            model, calibration, args.target_layer, args.device, args
        )
        matrices = install_checkpoint(model, payload, args.device)
        source = evaluate_domains(model, gates, args.device)
        source_restoration_exact = None
    else:
        matrices = install_checkpoint(model, payload, args.device)
        source = evaluate_domains(model, gates, args.device)
        restore_original_committed_groups(matrices[target_name], original_weight)
        teacher_metrics = evaluate_domains(model, gates, args.device)
        teacher = cache_teacher(
            model, calibration, args.target_layer, args.device, args
        )
        restore_source_state(model, matrices, payload, args.device)
        source_recheck = evaluate_domains(model, gates, args.device)
        source_restoration_exact = all(
            source_recheck[domain]["nll"] == source[domain]["nll"]
            for domain in gates
        )
        if not source_restoration_exact:
            raise RuntimeError(
                "source state did not restore exactly after counterfactual teacher cache"
            )
    target = model.get_submodule(target_name)
    residual = MaskedContinuousResidualLinear(
        matrices[target_name], target.bias
    ).to(args.device)
    set_submodule(model, target_name, residual)
    initial_selection = evaluate_domains(model, selection_gates, args.device)
    best_ratios = ratios(initial_selection, selection_baseline)
    best_objective = max(best_ratios.values())
    best_step = 0
    best_delta = residual.delta.detach().cpu().clone()

    capture = {}
    handle = model.model.layers[args.target_layer].register_forward_hook(
        lambda _module, _inputs, output: capture.__setitem__(
            "block", tensor_output(output)
        )
    )
    optimizer = torch.optim.AdamW([residual.delta], lr=args.lr, weight_decay=0.0)
    history = []
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model.train()
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
            active_delta = residual.masked_delta()
            anchor_loss = active_delta.float().square().mean()
            loss = (
                args.ce_weight * ce
                + args.block_weight * block_loss
                + args.kd_weight * kd_loss
                + args.residual_anchor_weight * anchor_loss
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_([residual.delta], 1.0)
            optimizer.step()
            with torch.no_grad():
                residual.delta.masked_fill_(~residual.weight_mask, 0)

            selection_metrics = None
            selection_ratios = None
            if step % args.selection_every == 0 or step == args.steps:
                model.eval()
                selection_metrics = evaluate_domains(
                    model, selection_gates, args.device
                )
                selection_ratios = ratios(selection_metrics, selection_baseline)
                objective = max(selection_ratios.values())
                if objective < best_objective:
                    best_objective = objective
                    best_ratios = selection_ratios
                    best_step = step
                    best_delta = residual.delta.detach().cpu().clone()
                model.train()
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
                    "residual": residual_statistics(residual, original_weight),
                }
                history.append(entry)
                print(
                    f"step={step}/{args.steps} ce={entry['ce']:.4f} "
                    f"best={best_step} ratios={selection_ratios}",
                    flush=True,
                )
    finally:
        handle.remove()
    model.eval()
    with torch.no_grad():
        residual.delta.copy_(best_delta.to(args.device))
    attempted = evaluate_domains(model, gates, args.device)
    source_ratios = ratios(source, baseline)
    attempted_ratios = ratios(attempted, baseline)
    improvement = max(source_ratios.values()) - max(attempted_ratios.values())
    passed = improvement >= args.min_improvement
    statistics = residual_statistics(residual, original_weight)

    artifact_path = WORKSPACE / f"wal2/artifacts/wal-tat-{args.tag}.pt"
    artifact_sha256 = None
    if passed:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "wal-tat-matched-bf16-residual-v1",
                "source_checkpoint_sha256": sha256_file(source_checkpoint),
                "suite_sha256": sha256_file(suite_path),
                "target_name": target_name,
                "teacher_source": args.teacher_source,
                "committed_mask": matrices[target_name].committed_mask.cpu(),
                "delta_bf16": residual.delta.detach().bfloat16().cpu(),
                "best_step": best_step,
                "metrics": attempted,
                "ratios": attempted_ratios,
                "statistics": statistics,
            },
            artifact_path,
        )
        artifact_sha256 = sha256_file(artifact_path)

    result = {
        "schema": "wal-tat-matched-bf16-residual-recovery-v1",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "target_name": target_name,
        "teacher_source": args.teacher_source,
        "teacher_metrics": teacher_metrics,
        "teacher_ratios": ratios(teacher_metrics, baseline),
        "source_restoration_exact": source_restoration_exact,
        "steps": args.steps,
        "lr": args.lr,
        "ce_weight": args.ce_weight,
        "block_weight": args.block_weight,
        "kd_weight": args.kd_weight,
        "residual_anchor_weight": args.residual_anchor_weight,
        "selection_every": args.selection_every,
        "selection_gate_sequences": args.selection_gate_sequences,
        "baseline": baseline,
        "selection_baseline": selection_baseline,
        "source": source,
        "source_ratios": source_ratios,
        "best_step": best_step,
        "best_selection_ratios": best_ratios,
        "attempted": attempted,
        "attempted_ratios": attempted_ratios,
        "worst_ratio_improvement": improvement,
        "min_improvement": args.min_improvement,
        "passed": passed,
        "residual_statistics": statistics,
        "artifact": str(artifact_path) if passed else None,
        "artifact_sha256": artifact_sha256,
        "coverage_changed": False,
        "checkpoint_written": False,
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
        f"ratios={attempted_ratios}",
        flush=True,
    )
    del model, matrices, residual, teacher, best_delta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
