"""Recover a mixed frontier by training only fixed-code Q4 group scales."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from compensation_window import (
    PROJECT,
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
from wal_tat import install_mixed_q2_q4_artifact


PROJECTION_MAP = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "up_proj": "mlp.up_proj",
    "gate_proj": "mlp.gate_proj",
    "down_proj": "mlp.down_proj",
}


class FixedCodeMixedScaleLinear(nn.Module):
    """Keep Q2/Q4 codes fixed and train positive Q4 scales only."""

    def __init__(
        self,
        entry: dict,
        *,
        max_abs_log_scale_delta: float,
        bias: torch.Tensor | None,
    ):
        super().__init__()
        if max_abs_log_scale_delta <= 0:
            raise ValueError("max_abs_log_scale_delta must be positive")
        rows, columns = map(int, entry["shape"])
        self.rows = rows
        self.columns = columns
        self.register_buffer("q2_codes", entry["q2_codes_int8"].float())
        self.register_buffer("q2_scales", entry["q2_scales_fp16"].float())
        self.register_buffer("q4_codes", entry["q4_codes_int8"].float())
        self.register_buffer("base_q4_scales", entry["q4_scales_fp16"].float())
        self.register_buffer("q4_mask", entry["q4_mask"].bool())
        self.log_scale_delta = nn.Parameter(torch.zeros_like(self.base_q4_scales))
        self.max_abs_log_scale_delta = float(max_abs_log_scale_delta)
        self.bias = None if bias is None else nn.Parameter(
            bias.detach().clone(), requires_grad=False
        )
        self.in_features = columns
        self.out_features = rows

    def effective_q4_scales(self) -> torch.Tensor:
        bounded = self.log_scale_delta.clamp(
            -self.max_abs_log_scale_delta, self.max_abs_log_scale_delta
        )
        scales = self.base_q4_scales * bounded.exp()
        rounded = scales.half().float()
        return scales + (rounded - scales).detach()

    def effective_weight(self) -> torch.Tensor:
        q2 = self.q2_codes * self.q2_scales.unsqueeze(-1)
        q4 = self.q4_codes * self.effective_q4_scales().unsqueeze(-1)
        grouped = torch.where(self.q4_mask.unsqueeze(-1), q4, q2)
        return grouped.reshape(self.rows, -1)[:, : self.columns]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.effective_weight().to(value.dtype), self.bias)

    @torch.no_grad()
    def constrain_(self) -> None:
        self.log_scale_delta.clamp_(
            -self.max_abs_log_scale_delta, self.max_abs_log_scale_delta
        )
        self.log_scale_delta.masked_fill_(~self.q4_mask, 0)

    @torch.no_grad()
    def deploy_scales(self) -> torch.Tensor:
        return self.effective_q4_scales().half().cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-layer", type=int, required=True)
    parser.add_argument(
        "--target-projections",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj",
    )
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--train-sequences", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--block-weight", type=float, default=0.1)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-topk", type=int, default=64)
    parser.add_argument("--kd-stride", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--max-abs-log-scale-delta", type=float, default=0.2)
    parser.add_argument("--selection-gate-sequences", type=int, default=64)
    parser.add_argument("--selection-every", type=int, default=64)
    parser.add_argument("--gate-ratio", type=float, default=1.02)
    parser.add_argument("--incremental-gate-ratio", type=float, default=1.005)
    parser.add_argument("--min-selection-improvement", type=float, default=1e-4)
    parser.add_argument("--write-artifact", action="store_true")
    parser.add_argument("--seed", type=int, default=727)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


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
            output = model(
                input_ids=chunk.unsqueeze(0).to(device)[:, :-1], use_cache=False
            )
            result.append(
                {
                    "block": capture["block"].squeeze(0).half().cpu(),
                    **teacher_bucket(output.logits, args),
                }
            )
            if (index + 1) % 64 == 0 or index + 1 == len(calibration):
                print(f"teacher {index + 1}/{len(calibration)}", flush=True)
    finally:
        handle.remove()
    return result


def normalized_mse(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    target = teacher.to(student.device).float().unsqueeze(0)
    return F.mse_loss(student.float(), target) / target.square().mean().clamp_min(1e-8)


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.train_sequences < 1:
        raise ValueError("steps and train sequences must be positive")
    source_path = args.source_checkpoint.resolve()
    parent_path = args.parent_artifact.resolve()
    candidate_path = args.candidate_artifact.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    source_hash = sha256_file(source_path)
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    parent_artifact = torch.load(parent_path, map_location="cpu", weights_only=False)
    candidate_artifact = torch.load(
        candidate_path, map_location="cpu", weights_only=False
    )
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = suite["calibration"][: args.train_sequences]
    gates = suite["gates"]
    if len(calibration) != args.train_sequences:
        raise ValueError("suite lacks requested train sequences")
    selection_gates = {
        domain: chunks[: args.selection_gate_sequences]
        for domain, chunks in gates.items()
    }
    if any(
        len(chunks) != args.selection_gate_sequences
        for chunks in selection_gates.values()
    ):
        raise ValueError("suite lacks requested selection sequences")
    requested = tuple(
        item.strip() for item in args.target_projections.split(",") if item.strip()
    )
    if not requested or any(item not in PROJECTION_MAP for item in requested):
        raise ValueError("invalid target projections")
    names = tuple(
        f"model.layers.{args.target_layer}.{PROJECTION_MAP[item]}"
        for item in requested
    )
    if any(name not in candidate_artifact["matrices"] for name in names):
        raise ValueError("candidate artifact lacks a target matrix")

    seed_everything(args.seed)
    started = time.time()
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    teacher = cache_teacher(
        model, calibration, args.target_layer, args.device, args
    )
    install_checkpoint(model, source_payload, args.device)
    install_mixed_q2_q4_artifact(
        model,
        parent_artifact,
        source_payload,
        device=args.device,
        expected_source_sha256=source_hash,
    )
    parent = evaluate_domains(model, gates, args.device)
    parent_selection = evaluate_domains(model, selection_gates, args.device)
    install_mixed_q2_q4_artifact(
        model,
        candidate_artifact,
        source_payload,
        device=args.device,
        expected_source_sha256=source_hash,
    )
    candidate = evaluate_domains(model, gates, args.device)
    candidate_selection = evaluate_domains(model, selection_gates, args.device)

    layers = {}
    for name in names:
        target = model.get_submodule(name)
        layer = FixedCodeMixedScaleLinear(
            candidate_artifact["matrices"][name],
            max_abs_log_scale_delta=args.max_abs_log_scale_delta,
            bias=target.bias,
        ).to(args.device)
        set_submodule(model, name, layer)
        layers[name] = layer

    parameters = [layer.log_scale_delta for layer in layers.values()]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    capture = {}
    handle = model.model.layers[args.target_layer].register_forward_hook(
        lambda _module, _inputs, output: capture.__setitem__(
            "block", tensor_output(output)
        )
    )
    candidate_selection_ratios = ratios(candidate_selection, selection_baseline)
    best_objective = max(candidate_selection_ratios.values())
    best_step = 0
    best_deltas = {
        name: layer.log_scale_delta.detach().cpu().clone()
        for name, layer in layers.items()
    }
    history = []
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
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
            kd = bucket_kl(logits, teacher[item], args)
            block = normalized_mse(capture["block"], teacher[item]["block"])
            loss = args.ce_weight * ce + args.kd_weight * kd + args.block_weight * block
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            for layer in layers.values():
                layer.constrain_()

            selection_ratios = None
            if step % args.selection_every == 0 or step == args.steps:
                model.eval()
                selection = evaluate_domains(model, selection_gates, args.device)
                selection_ratios = ratios(selection, selection_baseline)
                objective = max(selection_ratios.values())
                if objective < best_objective:
                    best_objective = objective
                    best_step = step
                    best_deltas = {
                        name: layer.log_scale_delta.detach().cpu().clone()
                        for name, layer in layers.items()
                    }
                model.train()
            if step == 1 or step % args.selection_every == 0 or step == args.steps:
                entry = {
                    "step": step,
                    "ce": float(ce.item()),
                    "kd": float(kd.item()),
                    "block": float(block.item()),
                    "gradient_norm": float(gradient_norm),
                    "selection_ratios": selection_ratios,
                    "best_step": best_step,
                }
                history.append(entry)
                print(
                    f"step={step}/{args.steps} kd={entry['kd']:.6f} "
                    f"block={entry['block']:.6f} best={best_step} "
                    f"ratios={selection_ratios}",
                    flush=True,
                )
    finally:
        handle.remove()

    model.eval()
    with torch.no_grad():
        for name, layer in layers.items():
            layer.log_scale_delta.copy_(best_deltas[name].to(args.device))
    attempted = evaluate_domains(model, gates, args.device)
    attempted_selection = evaluate_domains(model, selection_gates, args.device)
    attempted_ratios = ratios(attempted, baseline)
    attempted_incremental = ratios(attempted, parent)
    selection_improvement = max(candidate_selection_ratios.values()) - max(
        ratios(attempted_selection, selection_baseline).values()
    )
    passed = (
        selection_improvement >= args.min_selection_improvement
        and all(value <= args.gate_ratio for value in attempted_ratios.values())
        and all(
            value <= args.incremental_gate_ratio
            for value in attempted_incremental.values()
        )
    )

    artifact_path = None
    artifact_sha256 = None
    if args.write_artifact and passed:
        recovered = copy.deepcopy(candidate_artifact)
        for name, layer in layers.items():
            recovered["matrices"][name]["q4_scales_fp16"] = layer.deploy_scales()
        recovered["parent_artifact_sha256"] = sha256_file(parent_path)
        recovered["scale_recovery"] = {
            "source_candidate_sha256": sha256_file(candidate_path),
            "best_step": best_step,
            "development_ratios": attempted_ratios,
            "development_incremental_ratios_vs_parent": attempted_incremental,
        }
        artifact_path = (
            source_path.parents[1]
            / "artifacts"
            / f"wal-tat-{args.tag}-mixed-q2-q4.pt"
        )
        torch.save(recovered, artifact_path)
        artifact_sha256 = sha256_file(artifact_path)

    result = {
        "schema": "wal-tat-mixed-q4-scale-recovery-v1",
        "tag": args.tag,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_hash,
        "parent_artifact": str(parent_path),
        "parent_artifact_sha256": sha256_file(parent_path),
        "candidate_artifact": str(candidate_path),
        "candidate_artifact_sha256": sha256_file(candidate_path),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "target_layer": args.target_layer,
        "target_projections": requested,
        "steps": args.steps,
        "train_sequences": args.train_sequences,
        "lr": args.lr,
        "weights": {
            "ce": args.ce_weight,
            "kd": args.kd_weight,
            "block": args.block_weight,
        },
        "baseline": baseline,
        "parent": parent,
        "parent_ratios": ratios(parent, baseline),
        "candidate": candidate,
        "candidate_ratios": ratios(candidate, baseline),
        "candidate_incremental_ratios_vs_parent": ratios(candidate, parent),
        "attempted": attempted,
        "attempted_ratios": attempted_ratios,
        "attempted_incremental_ratios_vs_parent": attempted_incremental,
        "candidate_selection_ratios": candidate_selection_ratios,
        "attempted_selection_ratios": ratios(
            attempted_selection, selection_baseline
        ),
        "selection_improvement": selection_improvement,
        "best_step": best_step,
        "history": history,
        "gate_ratio": args.gate_ratio,
        "incremental_gate_ratio": args.incremental_gate_ratio,
        "passed": passed,
        "artifact": str(artifact_path) if artifact_path is not None else None,
        "artifact_sha256": artifact_sha256,
        "artifact_written": artifact_path is not None,
        "checkpoint_written": False,
        "sealed_audit_opened": False,
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
    del model, teacher, layers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
