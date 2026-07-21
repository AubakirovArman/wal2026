"""Direct fixed-code scale QAT from a target-local counterfactual teacher."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

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
    tensor_output,
)
from counterfactual_bf16_restore_audit import (
    restore_original_committed_groups,
    restore_source_state,
)
from matched_bf16_residual_recovery import cache_teacher, normalized_mse


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
    parser.add_argument(
        "--target-projection", choices=tuple(PROJECTION_MAP), default="up_proj"
    )
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--block-weight", type=float, default=0.2)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-topk", type=int, default=64)
    parser.add_argument("--kd-stride", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--scale-anchor-weight", type=float, default=0.0)
    parser.add_argument("--max-abs-log-scale-delta", type=float, default=0.25)
    parser.add_argument(
        "--fake-fp16-scale",
        action="store_true",
        help="Use FP16-rounded scales in forward with an STE gradient.",
    )
    parser.add_argument("--selection-gate-sequences", type=int, default=256)
    parser.add_argument("--selection-gate-start", type=int, default=0)
    parser.add_argument("--confirmation-gate-sequences", type=int)
    parser.add_argument("--confirmation-gate-start", type=int)
    parser.add_argument("--selection-every", type=int, default=128)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--min-confirmation-improvement", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=197)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def slice_domain_gates(
    gates: dict[str, list[torch.Tensor]],
    *,
    start: int,
    count: int,
    role: str,
) -> dict[str, list[torch.Tensor]]:
    if start < 0:
        raise ValueError(f"{role} gate start must be non-negative")
    if count <= 0:
        raise ValueError(f"{role} gate count must be positive")
    stop = start + count
    selected = {domain: chunks[start:stop] for domain, chunks in gates.items()}
    missing = {
        domain: len(chunks)
        for domain, chunks in selected.items()
        if len(chunks) != count
    }
    if missing:
        raise ValueError(
            f"suite lacks the requested {role} gate slice "
            f"[{start}, {stop}): {missing}"
        )
    return selected


class FixedCodeScaleLinear(nn.Module):
    """Train positive g128 scales while codes and all fallback weights stay fixed."""

    def __init__(
        self,
        matrix,
        max_abs_log_scale_delta: float,
        bias=None,
        fake_fp16_scale: bool = False,
    ):
        super().__init__()
        if max_abs_log_scale_delta <= 0:
            raise ValueError("max_abs_log_scale_delta must be positive")
        self.matrix = matrix
        for parameter in self.matrix.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("base_scale", matrix.group_scale.detach().clone())
        self.register_buffer("active_mask", matrix.committed_mask.detach().clone())
        self.log_scale_delta = nn.Parameter(torch.zeros_like(self.base_scale))
        self.max_abs_log_scale_delta = float(max_abs_log_scale_delta)
        self.fake_fp16_scale = bool(fake_fp16_scale)
        self.bias = None if bias is None else nn.Parameter(
            bias.detach().clone(), requires_grad=False
        )
        self.in_features = matrix.in_features
        self.out_features = matrix.out_features

    def effective_scales(self) -> torch.Tensor:
        bounded = self.log_scale_delta.clamp(
            -self.max_abs_log_scale_delta, self.max_abs_log_scale_delta
        )
        multiplier = torch.where(
            self.active_mask, bounded.exp(), torch.ones_like(bounded)
        )
        scales = self.base_scale * multiplier
        if self.fake_fp16_scale:
            rounded = scales.half().float()
            scales = scales + (rounded - scales).detach()
        return scales

    def effective_weight(self) -> torch.Tensor:
        scales = self.effective_scales()
        scale_delta = scales - self.base_scale
        grouped_delta = self.matrix.committed_codes.float() * scale_delta.unsqueeze(-1)
        grouped_delta = torch.where(
            self.active_mask.unsqueeze(-1),
            grouped_delta,
            torch.zeros_like(grouped_delta),
        )
        delta = grouped_delta.reshape(self.out_features, -1)[:, : self.in_features]
        return self.matrix.effective_weight() + delta.to(self.matrix.compute_dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.effective_weight().to(value.dtype), self.bias)

    def constrain_(self) -> None:
        with torch.no_grad():
            self.log_scale_delta.clamp_(
                -self.max_abs_log_scale_delta, self.max_abs_log_scale_delta
            )
            self.log_scale_delta.masked_fill_(~self.active_mask, 0)

    def set_scales_(self, scales: torch.Tensor) -> None:
        if scales.shape != self.base_scale.shape:
            raise ValueError("scale shape mismatch")
        ratio = scales.float().div(self.base_scale.clamp_min(1e-12))
        with torch.no_grad():
            self.log_scale_delta.copy_(ratio.log())
        self.constrain_()


@torch.no_grad()
def scale_statistics(layer: FixedCodeScaleLinear, original_weight: torch.Tensor) -> dict:
    matrix = layer.matrix
    scales = layer.effective_scales()
    mask = layer.active_mask
    relative = scales[mask].div(layer.base_scale[mask].clamp_min(1e-12)) - 1
    original = F.pad(
        original_weight.to(scales.device).float(), (0, matrix.padding)
    ).view_as(matrix.committed_codes)
    codes = matrix.committed_codes.float()
    numerator = (original * codes).sum(dim=-1)
    denominator = codes.square().sum(dim=-1).clamp_min(1)
    oracle = numerator.div(denominator).abs().clamp_min(1e-5)
    base_distance = (layer.base_scale[mask] - oracle[mask]).float().square().sum()
    candidate_distance = (scales[mask] - oracle[mask]).float().square().sum()
    return {
        "active_groups": int(mask.sum().item()),
        "active_weights": int(mask.sum().item() * matrix.group_size),
        "relative_change_mean": float(relative.mean().item()),
        "relative_change_abs_mean": float(relative.abs().mean().item()),
        "relative_change_min": float(relative.min().item()),
        "relative_change_max": float(relative.max().item()),
        "log_delta_abs_mean": float(layer.log_scale_delta[mask].abs().mean().item()),
        "clamp_fraction": float(
            (
                layer.log_scale_delta[mask].abs()
                >= layer.max_abs_log_scale_delta - 1e-7
            )
            .float()
            .mean()
            .item()
        ),
        "oracle_scale_distance_reduction_fraction": float(
            (1 - candidate_distance / base_distance.clamp_min(1e-12)).item()
        ),
    }


def main() -> None:
    args = parse_args()
    source_checkpoint = args.source_checkpoint.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = suite["calibration"]
    gates = suite["gates"]
    selection_gates = slice_domain_gates(
        gates,
        start=args.selection_gate_start,
        count=args.selection_gate_sequences,
        role="selection",
    )
    confirmation_gates = None
    if (args.confirmation_gate_start is None) != (
        args.confirmation_gate_sequences is None
    ):
        raise ValueError(
            "confirmation gate start and count must be provided together"
        )
    if args.confirmation_gate_start is not None:
        confirmation_gates = slice_domain_gates(
            gates,
            start=args.confirmation_gate_start,
            count=args.confirmation_gate_sequences,
            role="confirmation",
        )
        selection_range = set(
            range(
                args.selection_gate_start,
                args.selection_gate_start + args.selection_gate_sequences,
            )
        )
        confirmation_range = set(
            range(
                args.confirmation_gate_start,
                args.confirmation_gate_start
                + args.confirmation_gate_sequences,
            )
        )
        if selection_range & confirmation_range:
            raise ValueError("selection and confirmation gate slices overlap")
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
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    confirmation_baseline = (
        evaluate_domains(model, confirmation_gates, args.device)
        if confirmation_gates is not None
        else None
    )
    matrices = install_checkpoint(model, payload, args.device)
    source = evaluate_domains(model, gates, args.device)
    source_selection = evaluate_domains(model, selection_gates, args.device)
    source_confirmation = (
        evaluate_domains(model, confirmation_gates, args.device)
        if confirmation_gates is not None
        else None
    )
    restore_original_committed_groups(matrices[target_name], original_weight)
    teacher_metrics = evaluate_domains(model, gates, args.device)
    teacher = cache_teacher(model, calibration, args.target_layer, args.device, args)
    restore_source_state(model, matrices, payload, args.device)
    source_recheck = evaluate_domains(model, gates, args.device)
    source_restoration_exact = all(
        source_recheck[domain]["nll"] == source[domain]["nll"] for domain in gates
    )
    if not source_restoration_exact:
        raise RuntimeError("source did not restore exactly after teacher cache")

    target = model.get_submodule(target_name)
    scale_layer = FixedCodeScaleLinear(
        matrices[target_name],
        args.max_abs_log_scale_delta,
        target.bias,
        fake_fp16_scale=args.fake_fp16_scale,
    ).to(args.device)
    set_submodule(model, target_name, scale_layer)
    initial_selection = evaluate_domains(model, selection_gates, args.device)
    best_ratios = ratios(initial_selection, selection_baseline)
    best_objective = max(best_ratios.values())
    best_step = 0
    best_log_delta = scale_layer.log_scale_delta.detach().cpu().clone()

    capture = {}
    handle = model.model.layers[args.target_layer].register_forward_hook(
        lambda _module, _inputs, output: capture.__setitem__(
            "block", tensor_output(output)
        )
    )
    optimizer = torch.optim.AdamW(
        [scale_layer.log_scale_delta], lr=args.lr, weight_decay=0.0
    )
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
            anchor_loss = scale_layer.log_scale_delta[
                scale_layer.active_mask
            ].float().square().mean()
            loss = (
                args.ce_weight * ce
                + args.block_weight * block_loss
                + args.kd_weight * kd_loss
                + args.scale_anchor_weight * anchor_loss
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [scale_layer.log_scale_delta], 1.0
            )
            optimizer.step()
            scale_layer.constrain_()

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
                    best_log_delta = (
                        scale_layer.log_scale_delta.detach().cpu().clone()
                    )
                model.train()
            if step == 1 or step == args.steps or step % max(args.steps // 4, 1) == 0:
                entry = {
                    "step": step,
                    "ce": float(ce.item()),
                    "block_loss": float(block_loss.item()),
                    "kd_loss": float(kd_loss.item()),
                    "anchor_loss": float(anchor_loss.item()),
                    "gradient_norm": float(gradient_norm),
                    "selection_ratios": selection_ratios,
                    "best_step": best_step,
                    "scales": scale_statistics(scale_layer, original_weight),
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
        scale_layer.log_scale_delta.copy_(best_log_delta.to(args.device))
    deploy_scales = scale_layer.effective_scales().half().float()
    scale_layer.set_scales_(deploy_scales)
    attempted = evaluate_domains(model, gates, args.device)
    attempted_selection = evaluate_domains(model, selection_gates, args.device)
    attempted_confirmation = (
        evaluate_domains(model, confirmation_gates, args.device)
        if confirmation_gates is not None
        else None
    )
    source_ratios = ratios(source, baseline)
    attempted_ratios = ratios(attempted, baseline)
    improvement = max(source_ratios.values()) - max(attempted_ratios.values())
    source_selection_ratios = ratios(source_selection, selection_baseline)
    attempted_selection_ratios = ratios(attempted_selection, selection_baseline)
    selection_improvement = max(source_selection_ratios.values()) - max(
        attempted_selection_ratios.values()
    )
    source_confirmation_ratios = (
        ratios(source_confirmation, confirmation_baseline)
        if source_confirmation is not None
        else None
    )
    attempted_confirmation_ratios = (
        ratios(attempted_confirmation, confirmation_baseline)
        if attempted_confirmation is not None
        else None
    )
    confirmation_improvement = (
        max(source_confirmation_ratios.values())
        - max(attempted_confirmation_ratios.values())
        if source_confirmation_ratios is not None
        else None
    )
    passed = selection_improvement >= args.min_improvement and (
        confirmation_improvement is None
        or confirmation_improvement >= args.min_confirmation_improvement
    )
    statistics = scale_statistics(scale_layer, original_weight)
    codes_unchanged = torch.equal(
        matrices[target_name].committed_codes,
        payload["matrices"][target_name]["ternary_codes_int8"]
        .to(args.device)
        .view_as(matrices[target_name].committed_codes),
    )
    mask_unchanged = torch.equal(
        matrices[target_name].committed_mask,
        payload["matrices"][target_name]["committed_mask"].to(args.device),
    )

    artifact_path = WORKSPACE / f"wal2/artifacts/wal-tat-{args.tag}.pt"
    artifact_sha256 = None
    if passed:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "wal-tat-direct-scale-recovery-v1",
                "source_checkpoint_sha256": sha256_file(source_checkpoint),
                "suite_sha256": sha256_file(suite_path),
                "target_name": target_name,
                "teacher_source": "target_counterfactual",
                "committed_mask": matrices[target_name].committed_mask.cpu(),
                "ternary_codes_int8": matrices[
                    target_name
                ].committed_codes.detach().cpu(),
                "scales_fp16": deploy_scales.half().cpu(),
                "best_step": best_step,
                "metrics": attempted,
                "ratios": attempted_ratios,
                "selection_metrics": attempted_selection,
                "selection_ratios": attempted_selection_ratios,
                "confirmation_metrics": attempted_confirmation,
                "confirmation_ratios": attempted_confirmation_ratios,
                "statistics": statistics,
            },
            artifact_path,
        )
        artifact_sha256 = sha256_file(artifact_path)

    result = {
        "schema": "wal-tat-counterfactual-scale-recovery-v1",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "target_name": target_name,
        "teacher_source": "target_counterfactual",
        "teacher_metrics": teacher_metrics,
        "teacher_ratios": ratios(teacher_metrics, baseline),
        "source_restoration_exact": source_restoration_exact,
        "steps": args.steps,
        "lr": args.lr,
        "ce_weight": args.ce_weight,
        "block_weight": args.block_weight,
        "kd_weight": args.kd_weight,
        "scale_anchor_weight": args.scale_anchor_weight,
        "max_abs_log_scale_delta": args.max_abs_log_scale_delta,
        "fake_fp16_scale": args.fake_fp16_scale,
        "selection_every": args.selection_every,
        "selection_gate_start": args.selection_gate_start,
        "selection_gate_sequences": args.selection_gate_sequences,
        "confirmation_gate_start": args.confirmation_gate_start,
        "confirmation_gate_sequences": args.confirmation_gate_sequences,
        "baseline": baseline,
        "selection_baseline": selection_baseline,
        "confirmation_baseline": confirmation_baseline,
        "source": source,
        "source_ratios": source_ratios,
        "source_selection": source_selection,
        "source_selection_ratios": source_selection_ratios,
        "source_confirmation": source_confirmation,
        "source_confirmation_ratios": source_confirmation_ratios,
        "best_step": best_step,
        "best_selection_ratios": best_ratios,
        "attempted": attempted,
        "attempted_ratios": attempted_ratios,
        "attempted_selection": attempted_selection,
        "attempted_selection_ratios": attempted_selection_ratios,
        "attempted_confirmation": attempted_confirmation,
        "attempted_confirmation_ratios": attempted_confirmation_ratios,
        "worst_ratio_improvement": improvement,
        "selection_worst_ratio_improvement": selection_improvement,
        "confirmation_worst_ratio_improvement": confirmation_improvement,
        "min_improvement": args.min_improvement,
        "min_confirmation_improvement": args.min_confirmation_improvement,
        "passed": passed,
        "scale_statistics": statistics,
        "codes_unchanged": codes_unchanged,
        "committed_mask_unchanged": mask_unchanged,
        "persistent_bf16_residual": False,
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
    del model, matrices, scale_layer, teacher, best_log_delta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
