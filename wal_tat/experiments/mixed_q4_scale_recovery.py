"""Recover a mixed frontier with hard-forward Q4 scales and code proxies."""
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

from block_proxy_recovery import collect_input_second_moments
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
from wal_tat import (
    install_mixed_q2_q4_artifact,
    valid_group_weight_count,
    weighted_symmetric_odd_level_project,
)


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
    """Keep Q2 fixed while training positive Q4 scales and optional code proxies."""

    def __init__(
        self,
        entry: dict,
        *,
        max_abs_log_scale_delta: float,
        bias: torch.Tensor | None,
        train_codes: bool = False,
        progressive_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        if max_abs_log_scale_delta <= 0:
            raise ValueError("max_abs_log_scale_delta must be positive")
        rows, columns = map(int, entry["shape"])
        self.rows = rows
        self.columns = columns
        self.register_buffer("q2_codes", entry["q2_codes_int8"].float())
        self.register_buffer("q2_scales", entry["q2_scales_fp16"].float())
        self.register_buffer("base_q4_codes", entry["q4_codes_int8"].float())
        self.proxy_code = nn.Parameter(
            self.base_q4_codes.clone(), requires_grad=bool(train_codes)
        )
        self.code_lower = -8
        self.code_upper = 7
        self.register_buffer("base_q4_scales", entry["q4_scales_fp16"].float())
        self.register_buffer("q4_mask", entry["q4_mask"].bool())
        if progressive_mask is None:
            progressive_mask = self.q4_mask.clone()
        progressive_mask = progressive_mask.bool()
        if progressive_mask.shape != self.q4_mask.shape:
            raise ValueError("progressive mask shape does not match Q4 mask")
        if torch.logical_and(progressive_mask, ~self.q4_mask).any():
            raise ValueError("progressive mask must be a subset of Q4 groups")
        self.register_buffer("progressive_mask", progressive_mask)
        self.register_buffer(
            "q8_codes",
            entry.get(
                "q8_codes_int8", torch.zeros_like(entry["q4_codes_int8"])
            ).float(),
        )
        self.register_buffer(
            "q8_scales",
            entry.get(
                "q8_scales_fp16", torch.ones_like(entry["q4_scales_fp16"])
            ).float(),
        )
        self.register_buffer(
            "q8_mask",
            entry.get("q8_mask", torch.zeros_like(entry["q4_mask"])).bool(),
        )
        if torch.logical_and(self.q4_mask, self.q8_mask).any():
            raise ValueError("Q4 and Q8 masks overlap")
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
        hard = self.proxy_code.round().clamp(self.code_lower, self.code_upper)
        codes = hard.detach() + self.proxy_code - self.proxy_code.detach()
        progressive_q4 = codes * self.effective_q4_scales().unsqueeze(-1)
        fallback_q4 = self.base_q4_codes * self.base_q4_scales.unsqueeze(-1)
        q4 = torch.where(
            self.progressive_mask.unsqueeze(-1), progressive_q4, fallback_q4
        )
        grouped = torch.where(self.q4_mask.unsqueeze(-1), q4, q2)
        q8 = self.q8_codes * self.q8_scales.unsqueeze(-1)
        grouped = torch.where(self.q8_mask.unsqueeze(-1), q8, grouped)
        return grouped.reshape(self.rows, -1)[:, : self.columns]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.effective_weight().to(value.dtype), self.bias)

    @torch.no_grad()
    def constrain_(self) -> None:
        self.log_scale_delta.clamp_(
            -self.max_abs_log_scale_delta, self.max_abs_log_scale_delta
        )
        self.log_scale_delta.masked_fill_(~self.progressive_mask, 0)
        active = self.progressive_mask.unsqueeze(-1).expand_as(self.proxy_code)
        self.proxy_code[active] = self.proxy_code[active].clamp(
            self.code_lower, self.code_upper
        )
        self.proxy_code[~active] = self.base_q4_codes[~active]

    @torch.no_grad()
    def deploy_scales(self) -> torch.Tensor:
        return self.effective_q4_scales().half().cpu()

    @torch.no_grad()
    def deploy_codes(self) -> torch.Tensor:
        active = self.progressive_mask.unsqueeze(-1).expand_as(self.proxy_code)
        result = self.base_q4_codes.clone()
        result[active] = self.proxy_code[active].round().clamp(
            self.code_lower, self.code_upper
        )
        return result.to(torch.int8).cpu()

    @torch.no_grad()
    def reset_codebook_(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        *,
        levels: int,
    ) -> None:
        if levels < 3 or levels % 2 == 0:
            raise ValueError("levels must be an odd integer of at least three")
        if codes.shape != self.base_q4_codes.shape:
            raise ValueError("codebook code shape mismatch")
        if scales.shape != self.base_q4_scales.shape:
            raise ValueError("codebook scale shape mismatch")
        radius = levels // 2
        active = self.progressive_mask.unsqueeze(-1).expand_as(codes)
        if active.any() and (
            int(codes[active].min()) < -radius
            or int(codes[active].max()) > radius
        ):
            raise ValueError("codes fall outside requested codebook")
        self.code_lower = -radius
        self.code_upper = radius
        self.base_q4_codes[active] = codes.to(self.base_q4_codes)[active]
        self.proxy_code[active] = codes.to(self.proxy_code)[active]
        self.base_q4_scales[self.progressive_mask] = scales.to(
            self.base_q4_scales
        )[self.progressive_mask]
        self.log_scale_delta.zero_()
        self.constrain_()

    @torch.no_grad()
    def code_churn(self) -> float:
        active = self.progressive_mask.unsqueeze(-1).expand_as(self.base_q4_codes)
        if not active.any():
            return 0.0
        changed = self.deploy_codes().to(self.base_q4_codes.device).ne(
            self.base_q4_codes.to(torch.int8)
        )
        return float(changed[active].float().mean())

    def code_anchor_loss(self) -> torch.Tensor:
        active = self.progressive_mask.unsqueeze(-1).expand_as(self.proxy_code)
        if not active.any():
            return self.proxy_code.sum() * 0
        return (self.proxy_code[active] - self.base_q4_codes[active]).float().square().mean()

    @torch.no_grad()
    def proxy_statistics(self) -> dict[str, float]:
        active = self.progressive_mask.unsqueeze(-1).expand_as(self.proxy_code)
        if not active.any():
            return {
                "mean_abs_displacement": 0.0,
                "max_abs_displacement": 0.0,
                "near_boundary_fraction": 0.0,
            }
        displacement = (self.proxy_code - self.base_q4_codes).abs()[active]
        boundary_distance = (self.proxy_code - self.proxy_code.round()).abs()[active]
        return {
            "mean_abs_displacement": float(displacement.mean()),
            "max_abs_displacement": float(displacement.max()),
            "near_boundary_fraction": float((boundary_distance >= 0.45).float().mean()),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument(
        "--teacher-artifact",
        type=Path,
        help="optional accepted mixed artifact used instead of raw BF16 as teacher",
    )
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
    parser.add_argument(
        "--proxy-lr",
        type=float,
        default=0.0,
        help="positive value enables hard-forward Q4 code proxy updates",
    )
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--block-weight", type=float, default=0.1)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-topk", type=int, default=64)
    parser.add_argument("--kd-stride", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--max-abs-log-scale-delta", type=float, default=0.2)
    parser.add_argument("--code-anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--progressive-levels",
        default="",
        help="optional odd hard codebook schedule, for example 7,5,3",
    )
    parser.add_argument(
        "--progressive-fraction",
        type=float,
        default=1.0,
        help="lowest-damage fraction of Q4 groups allowed to collapse",
    )
    parser.add_argument(
        "--moment-sequences",
        type=int,
        default=64,
        help="calibration sequences for activation-weighted stage projection",
    )
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


def parse_progressive_levels(value: str) -> tuple[int, ...]:
    levels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not levels:
        return ()
    if any(item < 3 or item > 15 or item % 2 == 0 for item in levels):
        raise ValueError("progressive levels must be odd integers in [3, 15]")
    if any(right >= left for left, right in zip(levels, levels[1:])):
        raise ValueError("progressive levels must be strictly decreasing")
    return levels


def lowest_damage_progressive_masks(
    damage: dict[str, torch.Tensor],
    eligible: dict[str, torch.Tensor],
    fraction: float,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Select one global lowest-damage subset without touching Q4 fallback."""
    if not 0 < fraction <= 1:
        raise ValueError("progressive fraction must be in (0, 1]")
    if not damage or set(damage) != set(eligible):
        raise ValueError("damage and eligible maps must have the same keys")
    names = tuple(damage)
    values = []
    locations = []
    total = 0
    for matrix_index, name in enumerate(names):
        if damage[name].shape != eligible[name].shape:
            raise ValueError(f"shape mismatch for {name}")
        flat_indices = torch.where(eligible[name].reshape(-1))[0]
        total += int(flat_indices.numel())
        values.append(damage[name].reshape(-1).index_select(0, flat_indices))
        locations.extend(
            (matrix_index, int(flat_index)) for flat_index in flat_indices.tolist()
        )
    if total == 0:
        raise ValueError("target contains no eligible Q4 groups")
    selected = max(1, min(total, round(total * fraction)))
    joined = torch.cat(values)
    chosen = torch.topk(joined, selected, largest=False, sorted=False).indices
    masks = {name: torch.zeros_like(eligible[name]) for name in names}
    for joined_index in chosen.tolist():
        matrix_index, flat_index = locations[joined_index]
        masks[names[matrix_index]].view(-1)[flat_index] = True
    return masks, selected, total


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
    progressive_levels = parse_progressive_levels(args.progressive_levels)
    if args.steps < 1 or args.train_sequences < 1:
        raise ValueError("steps and train sequences must be positive")
    if not 1 <= args.moment_sequences <= args.train_sequences:
        raise ValueError("moment sequences must be within train sequences")
    if progressive_levels and args.proxy_lr <= 0:
        raise ValueError("progressive collapse requires a positive proxy lr")
    if not 0 < args.progressive_fraction <= 1:
        raise ValueError("progressive fraction must be in (0, 1]")
    if not progressive_levels and args.progressive_fraction != 1.0:
        raise ValueError("progressive fraction requires progressive levels")
    source_path = args.source_checkpoint.resolve()
    parent_path = args.parent_artifact.resolve()
    candidate_path = args.candidate_artifact.resolve()
    teacher_path = args.teacher_artifact.resolve() if args.teacher_artifact else None
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
    if teacher_path is not None:
        teacher_artifact = torch.load(
            teacher_path, map_location="cpu", weights_only=False
        )
        install_checkpoint(model, source_payload, args.device)
        install_mixed_q2_q4_artifact(
            model,
            teacher_artifact,
            source_payload,
            device=args.device,
            expected_source_sha256=source_hash,
        )
    teacher = cache_teacher(
        model, calibration, args.target_layer, args.device, args
    )
    if teacher_path is not None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = load_model(model_path, args.device)
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
    moments = None
    progressive_weights = None
    progressive_masks = None
    progressive_selected_groups = 0
    progressive_eligible_groups = 0
    progressive_selected_weights = 0
    if progressive_levels:
        moments = collect_input_second_moments(
            model,
            names,
            calibration,
            args.device,
            args.moment_sequences,
        )
        progressive_weights = {
            name: model.get_submodule(name).weight.detach().float().clone()
            for name in names
        }
        final_damage = {}
        eligible = {}
        for name in names:
            _codes, _scales, error = weighted_symmetric_odd_level_project(
                progressive_weights[name],
                moments[name],
                levels=progressive_levels[-1],
                group_size=128,
            )
            final_damage[name] = error.cpu()
            eligible[name] = candidate_artifact["matrices"][name][
                "q4_mask"
            ].bool().cpu()
        (
            progressive_masks,
            progressive_selected_groups,
            progressive_eligible_groups,
        ) = lowest_damage_progressive_masks(
            final_damage, eligible, args.progressive_fraction
        )
        progressive_selected_weights = sum(
            valid_group_weight_count(
                progressive_masks[name],
                int(candidate_artifact["matrices"][name]["shape"][1]),
                128,
            )
            for name in names
        )
        print(
            "progressive selection "
            f"groups={progressive_selected_groups}/{progressive_eligible_groups} "
            f"weights={progressive_selected_weights}",
            flush=True,
        )

    layers = {}
    for name in names:
        target = model.get_submodule(name)
        layer = FixedCodeMixedScaleLinear(
            candidate_artifact["matrices"][name],
            max_abs_log_scale_delta=args.max_abs_log_scale_delta,
            bias=target.bias,
            train_codes=args.proxy_lr > 0,
            progressive_mask=(
                progressive_masks[name] if progressive_masks is not None else None
            ),
        ).to(args.device)
        if progressive_levels:
            codes, scales, _error = weighted_symmetric_odd_level_project(
                progressive_weights[name],
                moments[name],
                levels=progressive_levels[0],
                group_size=128,
            )
            layer.reset_codebook_(
                codes,
                scales,
                levels=progressive_levels[0],
            )
        set_submodule(model, name, layer)
        layers[name] = layer

    capture = {}
    handle = model.model.layers[args.target_layer].register_forward_hook(
        lambda _module, _inputs, output: capture.__setitem__(
            "block", tensor_output(output)
        )
    )
    candidate_selection_ratios = ratios(candidate_selection, selection_baseline)
    best_step = 0
    history = []
    stage_summaries = []
    stage_levels = progressive_levels or (None,)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        for stage_index, levels in enumerate(stage_levels):
            if stage_index > 0:
                for name, layer in layers.items():
                    codes, scales, _error = weighted_symmetric_odd_level_project(
                        layer.effective_weight().detach().float(),
                        moments[name],
                        levels=levels,
                        group_size=128,
                    )
                    layer.reset_codebook_(codes, scales, levels=levels)

            model.eval()
            stage_initial = evaluate_domains(model, selection_gates, args.device)
            stage_initial_ratios = ratios(stage_initial, selection_baseline)
            stage_best_objective = max(stage_initial_ratios.values())
            stage_best_step = 0
            stage_best_deltas = {
                name: layer.log_scale_delta.detach().cpu().clone()
                for name, layer in layers.items()
            }
            stage_best_proxies = {
                name: layer.proxy_code.detach().cpu().clone()
                for name, layer in layers.items()
            }

            scale_parameters = [layer.log_scale_delta for layer in layers.values()]
            proxy_parameters = (
                [layer.proxy_code for layer in layers.values()]
                if args.proxy_lr > 0
                else []
            )
            parameters = scale_parameters + proxy_parameters
            parameter_groups = [{"params": scale_parameters, "lr": args.lr}]
            if proxy_parameters:
                parameter_groups.append(
                    {"params": proxy_parameters, "lr": args.proxy_lr}
                )
            optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)
            model.train()

            for stage_step in range(1, args.steps + 1):
                global_step = stage_index * args.steps + stage_step
                item = (global_step - 1) % len(calibration)
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
                code_anchor = torch.stack(
                    [layer.code_anchor_loss() for layer in layers.values()]
                ).mean()
                loss = (
                    args.ce_weight * ce
                    + args.kd_weight * kd
                    + args.block_weight * block
                    + args.code_anchor_weight * code_anchor
                )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                for layer in layers.values():
                    layer.constrain_()

                selection_ratios = None
                if (
                    stage_step % args.selection_every == 0
                    or stage_step == args.steps
                ):
                    model.eval()
                    selection = evaluate_domains(
                        model, selection_gates, args.device
                    )
                    selection_ratios = ratios(selection, selection_baseline)
                    objective = max(selection_ratios.values())
                    if objective < stage_best_objective:
                        stage_best_objective = objective
                        stage_best_step = stage_step
                        stage_best_deltas = {
                            name: layer.log_scale_delta.detach().cpu().clone()
                            for name, layer in layers.items()
                        }
                        stage_best_proxies = {
                            name: layer.proxy_code.detach().cpu().clone()
                            for name, layer in layers.items()
                        }
                    model.train()
                if (
                    stage_step == 1
                    or stage_step % args.selection_every == 0
                    or stage_step == args.steps
                ):
                    entry = {
                        "step": global_step,
                        "stage": stage_index,
                        "levels": levels,
                        "stage_step": stage_step,
                        "ce": float(ce.item()),
                        "kd": float(kd.item()),
                        "block": float(block.item()),
                        "gradient_norm": float(gradient_norm),
                        "code_anchor": float(code_anchor.item()),
                        "code_churn": {
                            name: layer.code_churn()
                            for name, layer in layers.items()
                        },
                        "proxy_statistics": {
                            name: layer.proxy_statistics()
                            for name, layer in layers.items()
                        },
                        "selection_ratios": selection_ratios,
                        "stage_best_step": stage_best_step,
                    }
                    history.append(entry)
                    print(
                        f"stage={stage_index} levels={levels} "
                        f"step={stage_step}/{args.steps} kd={entry['kd']:.6f} "
                        f"block={entry['block']:.6f} best={stage_best_step} "
                        f"ratios={selection_ratios}",
                        flush=True,
                    )

            model.eval()
            with torch.no_grad():
                for name, layer in layers.items():
                    layer.log_scale_delta.copy_(
                        stage_best_deltas[name].to(args.device)
                    )
                    layer.proxy_code.copy_(
                        stage_best_proxies[name].to(args.device)
                    )
            best_step = stage_index * args.steps + stage_best_step
            stage_summaries.append(
                {
                    "stage": stage_index,
                    "levels": levels,
                    "initial_selection_ratios": stage_initial_ratios,
                    "best_step": stage_best_step,
                    "best_objective": stage_best_objective,
                }
            )
    finally:
        handle.remove()

    model.eval()
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
            matrix_entry = recovered["matrices"][name]
            deployed_scales = layer.deploy_scales()
            deployed_codes = layer.deploy_codes()
            if progressive_levels and progressive_levels[-1] == 3:
                collapse_mask = layer.progressive_mask.detach().cpu().bool()
                active = collapse_mask.unsqueeze(-1).expand_as(deployed_codes)
                if active.any() and int(deployed_codes[active].abs().max()) > 1:
                    raise RuntimeError("final three-level stage is not ternary")
                matrix_entry["q2_codes_int8"] = matrix_entry[
                    "q2_codes_int8"
                ].clone()
                matrix_entry["q2_scales_fp16"] = matrix_entry[
                    "q2_scales_fp16"
                ].clone()
                matrix_entry["q2_codes_int8"][collapse_mask] = deployed_codes[
                    collapse_mask
                ]
                matrix_entry["q2_scales_fp16"][collapse_mask] = deployed_scales[
                    collapse_mask
                ]
                matrix_entry["q4_mask"] = matrix_entry["q4_mask"].bool().clone()
                matrix_entry["q4_mask"][collapse_mask] = False
            else:
                matrix_entry["q4_scales_fp16"] = deployed_scales
                matrix_entry["q4_codes_int8"] = deployed_codes
        recovered["parent_artifact_sha256"] = sha256_file(parent_path)
        recovered["scale_recovery"] = {
            "source_candidate_sha256": sha256_file(candidate_path),
            "best_step": best_step,
            "proxy_lr": args.proxy_lr,
            "progressive_levels": progressive_levels,
            "progressive_fraction": args.progressive_fraction,
            "progressive_selected_groups": progressive_selected_groups,
            "progressive_selected_weights": progressive_selected_weights,
            "development_ratios": attempted_ratios,
            "development_incremental_ratios_vs_parent": attempted_incremental,
        }
        suffix = (
            "mixed-q2-q4-q8.pt"
            if recovered.get("format") == "wal-tat-mixed-q2-q4-q8-v1"
            else "mixed-q2-q4.pt"
        )
        artifact_path = source_path.parents[1] / "artifacts" / f"wal-tat-{args.tag}-{suffix}"
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
        "teacher_artifact": str(teacher_path) if teacher_path is not None else None,
        "teacher_artifact_sha256": (
            sha256_file(teacher_path) if teacher_path is not None else None
        ),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "target_layer": args.target_layer,
        "target_projections": requested,
        "steps": args.steps,
        "train_sequences": args.train_sequences,
        "lr": args.lr,
        "proxy_lr": args.proxy_lr,
        "progressive_levels": progressive_levels,
        "progressive_fraction": args.progressive_fraction,
        "progressive_selected_groups": progressive_selected_groups,
        "progressive_eligible_groups": progressive_eligible_groups,
        "progressive_selected_weights": progressive_selected_weights,
        "moment_sequences": args.moment_sequences,
        "weights": {
            "ce": args.ce_weight,
            "kd": args.kd_weight,
            "block": args.block_weight,
            "code_anchor": args.code_anchor_weight,
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
        "stage_summaries": stage_summaries,
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
