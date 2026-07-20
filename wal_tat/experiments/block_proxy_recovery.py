"""Hard-forward proxy-code recovery for full or partially ternary blocks."""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from wal_tat import (
    HashChainWAL,
    ProxyTernaryLinear,
    ProxyTernaryMatrix,
    soft_ternary_proxy_derivative,
)
from wal_tat.scoring import diagonal_ternary_search, exact_diagonal_ternary_project

from boundary_sparse_recode import (
    adjacent_move_proposals,
    apply_global_top_group_moves,
    refit_selected_scales,
)

from compensation_window import (
    PROJECT,
    WORKSPACE,
    bucket_kl,
    checkpoint_payload,
    default_model_path,
    ensure_candidate_matrix,
    evaluate_domains,
    fresh_verify,
    install_checkpoint,
    load_model,
    log,
    ratios,
    seed_everything,
    set_submodule,
    sha256_file,
    tensor_output,
    teacher_bucket,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--selection-suite",
        type=Path,
        help=(
            "optional suite whose gates are used only for checkpoint selection; "
            "keeps recovery/dev data separate from model selection"
        ),
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", default="block_proxy_recovery_v1")
    parser.add_argument("--target-layer", type=int)
    parser.add_argument(
        "--target-projections",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="comma-separated projection suffixes within attention/MLP",
    )
    parser.add_argument(
        "--add-target-layer",
        action="store_true",
        help="initialize and add all seven projections of target-layer",
    )
    parser.add_argument(
        "--complete-target-projections",
        action="store_true",
        help=(
            "strict-ternarize every still-uncommitted group in the requested "
            "existing projections before joint QAT; previously committed "
            "groups remain gradient-frozen"
        ),
    )
    parser.add_argument(
        "--completion-initializer",
        choices=("threshold_ls", "exact_support"),
        default="threshold_ls",
        help="strict ternary initializer for newly completed groups",
    )
    parser.add_argument(
        "--completion-moment-sequences",
        type=int,
        default=0,
        help=(
            "number of development calibration sequences used to estimate "
            "per-input activation second moments for completion initialization"
        ),
    )
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--proxy-lr", type=float, default=7.5e-4)
    parser.add_argument("--scale-lr", type=float, default=5e-7)
    parser.add_argument("--proxy-start-step", type=int, default=1)
    parser.add_argument(
        "--proxy-freeze-step",
        type=int,
        default=0,
        help="last step with proxy updates; zero keeps proxies trainable throughout",
    )
    parser.add_argument("--scale-start-step", type=int, default=1)
    parser.add_argument("--compensation-scale-lr", type=float, default=1e-6)
    parser.add_argument("--norm-lr", type=float, default=0.0)
    parser.add_argument(
        "--compensate-source-scales",
        action="store_true",
        help="reopen every source matrix scale while freezing its ternary codes",
    )
    parser.add_argument("--block-weight", type=float, default=0.02)
    parser.add_argument("--attention-weight", type=float, default=0.02)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--proxy-anchor-weight", type=float, default=0.0)
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument("--kd-topk", type=int, default=64)
    parser.add_argument("--kd-stride", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--initial-temperature", type=float, default=0.35)
    parser.add_argument("--final-temperature", type=float, default=0.10)
    parser.add_argument("--gate-ratio", type=float, default=1.03)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument(
        "--allow-within-gate",
        action="store_true",
        help="accept a recovery state inside the gate even if it does not beat its source dev objective",
    )
    parser.add_argument("--selection-gate-sequences", type=int, default=32)
    parser.add_argument("--selection-every", type=int, default=384)
    parser.add_argument(
        "--discrete-repair-rounds",
        type=int,
        default=0,
        help="targeted first-order adjacent-code repair rounds before Adam QAT",
    )
    parser.add_argument("--discrete-repair-sequences", type=int, default=64)
    parser.add_argument(
        "--discrete-repair-domain",
        default="all",
        help="all or an exact gate-domain name used to select calibration sequences",
    )
    parser.add_argument(
        "--discrete-repair-group-budgets", default="256,1024,4096"
    )
    parser.add_argument(
        "--discrete-repair-scale-modes", default="fixed,original_ls"
    )
    parser.add_argument("--discrete-repair-ce-weight", type=float, default=0.0)
    parser.add_argument("--discrete-repair-block-weight", type=float, default=0.2)
    parser.add_argument("--discrete-repair-kd-weight", type=float, default=1.0)
    parser.add_argument(
        "--discrete-repair-min-improvement", type=float, default=1e-5
    )
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def cache_hidden_teacher(model, calibration, layer_index, device, args):
    captures = {}
    result = []
    handles = [
        model.model.layers[layer_index].register_forward_hook(
            lambda _module, _inputs, output: captures.__setitem__(
                "block", tensor_output(output)
            )
        ),
        model.model.layers[layer_index].self_attn.register_forward_hook(
            lambda _module, _inputs, output: captures.__setitem__(
                "attention", tensor_output(output)
            )
        ),
    ]
    try:
        for index, chunk in enumerate(calibration):
            captures.clear()
            batch = chunk.unsqueeze(0).to(device)
            output = model(input_ids=batch[:, :-1], use_cache=False)
            entry = {
                "block": captures["block"].squeeze(0).half().cpu(),
                "attention": captures["attention"].squeeze(0).half().cpu(),
            }
            if args.kd_weight > 0 or args.discrete_repair_kd_weight > 0:
                entry.update(teacher_bucket(output.logits, args))
            result.append(entry)
            if (index + 1) % 64 == 0 or index + 1 == len(calibration):
                log(f"teacher hidden {index + 1}/{len(calibration)}")
    finally:
        for handle in handles:
            handle.remove()
    return result


def normalized_mse(student, teacher):
    target = teacher.to(student.device).float().unsqueeze(0)
    return F.mse_loss(student.float(), target) / target.square().mean().clamp_min(1e-8)


@torch.no_grad()
def collect_input_second_moments(model, names, calibration, device, count):
    """Collect diagonal input covariance for several linear operators at once."""
    if count < 1 or count > len(calibration):
        raise ValueError("completion moment sequence count is outside calibration")
    sums = {}
    samples = {name: 0 for name in names}
    handles = []

    def capture(name, _module, inputs):
        value = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
        contribution = value.square().sum(0)
        if name not in sums:
            sums[name] = contribution
        else:
            sums[name].add_(contribution)
        samples[name] += value.shape[0]

    for name in names:
        handles.append(
            model.get_submodule(name).register_forward_pre_hook(
                lambda module, inputs, target=name: capture(target, module, inputs)
            )
        )
    model.eval()
    try:
        for index, chunk in enumerate(calibration[:count]):
            model(input_ids=chunk.unsqueeze(0).to(device)[:, :-1], use_cache=False)
            if (index + 1) % 64 == 0 or index + 1 == count:
                log(f"completion moments {index + 1}/{count}")
    finally:
        for handle in handles:
            handle.remove()
    return {
        name: sums[name].div(max(samples[name], 1)).clamp_min(0)
        for name in names
    }


def comma_separated_positive_integers(value: str, role: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 1 for item in parsed):
        raise ValueError(f"{role} must contain positive integers")
    return parsed


def calibration_indices_for_domain(suite: dict, domain: str) -> tuple[int, ...]:
    """Recover calibration indices for one domain from the suite build policy."""
    calibration = suite["calibration"]
    if domain == "all":
        return tuple(range(len(calibration)))
    domains = tuple(suite["gates"])
    if len(domains) != 3:
        raise ValueError("domain calibration routing requires exactly three domains")
    if domain not in domains:
        raise ValueError(f"unknown discrete repair domain: {domain}")
    base_count = int(suite["calibration_per_domain"])
    repeats = {
        domains[0]: int(suite["c4_calibration_repeat"]),
        domains[1]: int(suite["squad_calibration_repeat"]),
        domains[2]: int(suite["code_calibration_repeat"]),
    }
    if suite.get("interleave_calibration", False):
        cycle = sum(repeats.values())
        offset = sum(repeats[name] for name in domains[: domains.index(domain)])
        indices = tuple(
            base * cycle + offset + repeat
            for base in range(base_count)
            for repeat in range(repeats[domain])
        )
    else:
        offset = sum(base_count * repeats[name] for name in domains[: domains.index(domain)])
        indices = tuple(range(offset, offset + base_count * repeats[domain]))
    if any(index >= len(calibration) for index in indices):
        raise ValueError("suite calibration metadata is inconsistent")
    return indices


def corrected_proxy_weight_gradient(proxy: ProxyTernaryMatrix) -> torch.Tensor:
    """Undo the smooth-surrogate Jacobian to estimate d(loss)/d(weight)."""
    if proxy.proxy_code.grad is None:
        raise RuntimeError("proxy gradient is unavailable")
    derivative = soft_ternary_proxy_derivative(
        proxy.proxy_code.detach(), proxy.temperature
    ).clamp_min(1e-5)
    scale = proxy.group_scale.detach().abs().clamp_min(1e-5).unsqueeze(-1)
    grouped = proxy.proxy_code.grad.detach().float() / derivative / scale
    return grouped.reshape(proxy.out_features, -1)[:, : proxy.in_features]


def main() -> None:
    args = parse_args()
    if args.discrete_repair_rounds < 0:
        raise ValueError("discrete-repair-rounds must be non-negative")
    repair_budgets = comma_separated_positive_integers(
        args.discrete_repair_group_budgets, "discrete repair group budgets"
    )
    repair_scale_modes = tuple(
        item.strip()
        for item in args.discrete_repair_scale_modes.split(",")
        if item.strip()
    )
    if not repair_scale_modes or set(repair_scale_modes) - {"fixed", "original_ls"}:
        raise ValueError("unsupported discrete repair scale mode")
    if args.add_target_layer and args.complete_target_projections:
        raise ValueError(
            "add-target-layer and complete-target-projections are mutually exclusive"
        )
    if args.discrete_repair_rounds and not args.complete_target_projections:
        raise ValueError(
            "targeted discrete repair currently requires complete-target-projections"
        )
    if not 1 <= args.proxy_start_step <= args.steps + 1:
        raise ValueError("proxy-start-step must be in [1, steps + 1]")
    if args.proxy_freeze_step and not (
        args.proxy_start_step <= args.proxy_freeze_step <= args.steps
    ):
        raise ValueError(
            "proxy-freeze-step must be zero or in [proxy-start-step, steps]"
        )
    if not 1 <= args.scale_start_step <= args.steps + 1:
        raise ValueError("scale-start-step must be in [1, steps + 1]")
    model_path = (args.model_path or default_model_path()).resolve()
    source_hash = sha256_file(args.source_checkpoint)
    suite_hash = sha256_file(args.suite)
    source_payload = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    suite = torch.load(args.suite, map_location="cpu", weights_only=False)
    calibration, gates = suite["calibration"], suite["gates"]
    repair_calibration_indices = calibration_indices_for_domain(
        suite, args.discrete_repair_domain
    )
    if args.discrete_repair_rounds and not (
        1 <= args.discrete_repair_sequences <= len(repair_calibration_indices)
    ):
        raise ValueError("discrete repair sequence count is outside its domain")
    selection_suite_path = args.selection_suite or args.suite
    selection_suite = torch.load(
        selection_suite_path, map_location="cpu", weights_only=False
    )
    selection_suite_hash = sha256_file(selection_suite_path)
    if "gates" not in selection_suite:
        raise ValueError("selection suite must contain gates")
    source_layer_indices = {
        int(name.split(".")[2]) for name in source_payload["matrices"]
    }
    if args.target_layer is None:
        if args.add_target_layer or len(source_layer_indices) != 1:
            raise ValueError("target-layer is required when adding or using a multi-block source")
        layer_index = next(iter(source_layer_indices))
    else:
        layer_index = args.target_layer
    if not 0 <= layer_index < 28:
        raise ValueError("target-layer must be in [0, 27]")

    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    selection_gates = {
        name: chunks[: args.selection_gate_sequences]
        for name, chunks in selection_suite["gates"].items()
    }
    if any(not chunks for chunks in selection_gates.values()):
        raise ValueError("selection suite has an empty gate domain")
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    teacher = cache_hidden_teacher(model, calibration, layer_index, args.device, args)
    matrices = install_checkpoint(model, source_payload, args.device)
    starting = evaluate_domains(model, gates, args.device)
    starting_ratios = ratios(starting, baseline)

    projection_map = {
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "gate_proj": "mlp.gate_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
    }
    requested = tuple(
        value.strip() for value in args.target_projections.split(",") if value.strip()
    )
    if not requested or any(value not in projection_map for value in requested):
        raise ValueError(f"invalid target-projections: {requested}")
    projection_names = tuple(projection_map[value] for value in requested)
    active_names = tuple(
        f"model.layers.{layer_index}.{projection}" for projection in projection_names
    )
    completion_moments = {}
    if args.complete_target_projections and args.completion_moment_sequences:
        completion_moments = collect_input_second_moments(
            model,
            active_names,
            calibration,
            args.device,
            args.completion_moment_sequences,
        )
    completion_masks = {}
    completion_targets = {}
    if args.add_target_layer:
        overlap = set(active_names) & set(matrices)
        if overlap:
            raise ValueError(f"target projections are already present: {sorted(overlap)}")
        for name in active_names:
            matrix = ensure_candidate_matrix(model, matrices, name, args.device)
            mask = torch.ones_like(matrix.committed_mask)
            matrix.begin(mask, transaction_id=f"{args.tag}-initial-hard")
            matrix.set_candidate_state(1.0, 0.0)
            matrix.commit()
        initial = evaluate_domains(model, gates, args.device)
        initial_ratios = ratios(initial, baseline)
    else:
        missing = set(active_names) - set(matrices)
        if missing:
            raise ValueError(
                f"target layer {layer_index} is missing {sorted(missing)}"
            )
        if args.complete_target_projections:
            for name in active_names:
                matrix = matrices[name]
                candidate_mask = ~matrix.committed_mask
                if not candidate_mask.any():
                    raise ValueError(f"target projection is already complete: {name}")
                input_moment = completion_moments.get(name)
                if input_moment is None:
                    input_moment = torch.ones(
                        matrix.in_features,
                        dtype=torch.float32,
                        device=matrix.master_weight.device,
                    )
                projector = (
                    exact_diagonal_ternary_project
                    if args.completion_initializer == "exact_support"
                    else diagonal_ternary_search
                )
                codes, scales, _ = projector(
                    matrix.master_weight, input_moment, group_size=matrix.group_size
                )
                completion_targets[name] = F.pad(
                    matrix.master_weight.detach().float(), (0, matrix.padding)
                ).view_as(matrix.committed_codes).clone()
                matrix.begin(
                    candidate_mask,
                    transaction_id=(
                        f"{args.tag}-complete-{name.rsplit('.', 1)[-1]}"
                    ),
                )
                matrix.set_candidate_codes(codes, scales)
                matrix.set_candidate_state(1.0, 0.0)
                matrix.commit()
                completion_masks[name] = candidate_mask.detach().clone()
            initial = evaluate_domains(model, gates, args.device)
            initial_ratios = ratios(initial, baseline)
        else:
            initial = starting
            initial_ratios = starting_ratios

    proxies = {}
    proxy_gradient_hooks = []
    for name in active_names:
        matrix = matrices[name]
        proxy = ProxyTernaryMatrix(
            matrix.committed_codes,
            matrix.group_scale,
            compute_dtype=matrix.compute_dtype,
            temperature=args.initial_temperature,
            committed_mask=matrix.committed_mask,
            master_weight=matrix.master_weight,
        ).to(args.device)
        target = model.get_submodule(name)
        set_submodule(model, name, ProxyTernaryLinear(proxy, target.bias).to(args.device))
        proxies[name] = proxy
        if args.complete_target_projections:
            trainable_mask = completion_masks[name]
            code_gradient_mask = trainable_mask.unsqueeze(-1).to(torch.float32)
            scale_gradient_mask = trainable_mask.to(torch.float32)
            proxy_gradient_hooks.extend(
                [
                    proxy.proxy_code.register_hook(
                        lambda gradient, mask=code_gradient_mask: gradient * mask
                    ),
                    proxy.group_scale.register_hook(
                        lambda gradient, mask=scale_gradient_mask: gradient * mask
                    ),
                ]
            )

    compensation_proxies = {}
    if args.compensate_source_scales:
        for name, matrix in matrices.items():
            if name in proxies:
                continue
            proxy = ProxyTernaryMatrix(
                matrix.committed_codes,
                matrix.group_scale,
                compute_dtype=matrix.compute_dtype,
                temperature=args.initial_temperature,
                committed_mask=matrix.committed_mask,
                master_weight=matrix.master_weight,
            ).to(args.device)
            proxy.proxy_code.requires_grad_(False)
            target = model.get_submodule(name)
            set_submodule(
                model, name, ProxyTernaryLinear(proxy, target.bias).to(args.device)
            )
            compensation_proxies[name] = proxy
    all_proxies = {**compensation_proxies, **proxies}

    captures = {}
    handles = [
        model.model.layers[layer_index].register_forward_hook(
            lambda _module, _inputs, output: captures.__setitem__(
                "block", tensor_output(output)
            )
        ),
        model.model.layers[layer_index].self_attn.register_forward_hook(
            lambda _module, _inputs, output: captures.__setitem__(
                "attention", tensor_output(output)
            )
        ),
    ]
    pre_repair_selection = evaluate_domains(model, selection_gates, args.device)
    pre_repair_selection_ratios = ratios(
        pre_repair_selection, selection_baseline
    )
    discrete_repair_history = []
    if args.discrete_repair_rounds:
        current_selection = pre_repair_selection
        current_selection_ratios = pre_repair_selection_ratios
        for repair_round in range(1, args.discrete_repair_rounds + 1):
            round_source_selection_ratios = current_selection_ratios
            accumulated_gradients = {
                name: torch.zeros(
                    (proxy.out_features, proxy.in_features),
                    dtype=torch.float32,
                    device=proxy.proxy_code.device,
                )
                for name, proxy in proxies.items()
            }
            gradient_loss_history = []
            model.train()
            selected_repair_indices = repair_calibration_indices[
                : args.discrete_repair_sequences
            ]
            for local_item, item in enumerate(selected_repair_indices):
                chunk = calibration[item]
                model.zero_grad(set_to_none=True)
                captures.clear()
                batch = chunk.unsqueeze(0).to(args.device)
                logits = model(input_ids=batch[:, :-1], use_cache=False).logits
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    batch[:, 1:].reshape(-1),
                )
                block_loss = normalized_mse(
                    captures["block"], teacher[item]["block"]
                )
                kd_loss = (
                    bucket_kl(logits, teacher[item], args)
                    if args.discrete_repair_kd_weight > 0
                    else logits.new_zeros(())
                )
                loss = (
                    args.discrete_repair_ce_weight * ce
                    + args.discrete_repair_block_weight * block_loss
                    + args.discrete_repair_kd_weight * kd_loss
                )
                loss.backward()
                for name, proxy in proxies.items():
                    accumulated_gradients[name].add_(
                        corrected_proxy_weight_gradient(proxy)
                    )
                if (
                    local_item == 0
                    or (local_item + 1) % 32 == 0
                    or local_item + 1 == args.discrete_repair_sequences
                ):
                    gradient_loss_history.append(
                        {
                            "sequence": local_item + 1,
                            "calibration_index": item,
                            "ce": float(ce.item()),
                            "block_loss": float(block_loss.item()),
                            "kd_loss": float(kd_loss.item()),
                        }
                    )
                    log(
                        f"discrete repair round={repair_round} gradient "
                        f"{local_item + 1}/{args.discrete_repair_sequences}"
                    )
            for gradient in accumulated_gradients.values():
                gradient.div_(args.discrete_repair_sequences)
            model.eval()

            source_codes = {
                name: proxy.hard_codes() for name, proxy in proxies.items()
            }
            source_scales = {
                name: proxy.group_scale.detach().clone()
                for name, proxy in proxies.items()
            }
            proposed_codes = {}
            proposal_scores = {}
            gradient_statistics = {}
            for name, proxy in proxies.items():
                proposed, _delta, scores = adjacent_move_proposals(
                    source_codes[name],
                    source_scales[name],
                    accumulated_gradients[name],
                    completion_masks[name],
                    in_features=proxy.in_features,
                )
                proposed_codes[name] = proposed
                proposal_scores[name] = scores
                active_gradient = accumulated_gradients[name]
                gradient_statistics[name] = {
                    "mean_abs": float(active_gradient.abs().mean().item()),
                    "max_abs": float(active_gradient.abs().max().item()),
                    "l2": float(active_gradient.norm().item()),
                }

            candidate_states = {}
            candidate_results = {}
            for budget in repair_budgets:
                try:
                    (
                        candidate_codes,
                        selected_groups,
                        _selected_positions,
                        chosen_scores,
                    ) = apply_global_top_group_moves(
                        source_codes,
                        proposed_codes,
                        proposal_scores,
                        budget,
                    )
                except ValueError:
                    continue
                for scale_mode in repair_scale_modes:
                    candidate_scales = {
                        name: value.clone()
                        for name, value in source_scales.items()
                    }
                    if scale_mode == "original_ls":
                        for name in proxies:
                            if selected_groups[name].numel():
                                candidate_scales[name] = refit_selected_scales(
                                    completion_targets[name],
                                    candidate_codes[name],
                                    source_scales[name],
                                    selected_groups[name],
                                )
                    candidate_scales = {
                        name: value.half().float()
                        for name, value in candidate_scales.items()
                    }
                    with torch.no_grad():
                        for name, proxy in proxies.items():
                            proxy.proxy_code.copy_(candidate_codes[name].float())
                            proxy.group_scale.copy_(candidate_scales[name])
                    selection_metrics = evaluate_domains(
                        model, selection_gates, args.device
                    )
                    selection_ratios = ratios(
                        selection_metrics, selection_baseline
                    )
                    key = (
                        f"budget{budget}_groups{int(chosen_scores.numel())}_"
                        f"{scale_mode}"
                    )
                    changed = {
                        name: int(
                            (
                                (candidate_codes[name] != source_codes[name])
                                & completion_masks[name].unsqueeze(-1)
                            ).sum().item()
                        )
                        for name in proxies
                    }
                    candidate_results[key] = {
                        "requested_group_budget": budget,
                        "selected_groups": {
                            name: int(groups.numel())
                            for name, groups in selected_groups.items()
                        },
                        "changed_code_values": changed,
                        "scale_mode": scale_mode,
                        "score_sum": float(chosen_scores.sum().item()),
                        "score_min": float(chosen_scores.min().item()),
                        "selection_metrics": selection_metrics,
                        "selection_ratios": selection_ratios,
                        "selection_worst_ratio": max(selection_ratios.values()),
                    }
                    candidate_states[key] = {
                        "codes": {
                            name: value.cpu()
                            for name, value in candidate_codes.items()
                        },
                        "scales": {
                            name: value.cpu()
                            for name, value in candidate_scales.items()
                        },
                    }
                    with torch.no_grad():
                        for name, proxy in proxies.items():
                            proxy.proxy_code.copy_(source_codes[name].float())
                            proxy.group_scale.copy_(source_scales[name])

            if not candidate_states:
                discrete_repair_history.append(
                    {
                        "round": repair_round,
                        "accepted": False,
                        "reason": "no_positive_first_order_moves",
                        "gradient_loss_history": gradient_loss_history,
                        "gradient_statistics": gradient_statistics,
                    }
                )
                break
            chosen_key = min(
                candidate_states,
                key=lambda key: candidate_results[key]["selection_worst_ratio"],
            )
            chosen_result = candidate_results[chosen_key]
            improvement = max(current_selection_ratios.values()) - chosen_result[
                "selection_worst_ratio"
            ]
            accepted = improvement >= args.discrete_repair_min_improvement
            if accepted:
                chosen_state = candidate_states[chosen_key]
                with torch.no_grad():
                    for name, proxy in proxies.items():
                        proxy.proxy_code.copy_(
                            chosen_state["codes"][name].to(args.device).float()
                        )
                        proxy.group_scale.copy_(
                            chosen_state["scales"][name].to(args.device)
                        )
                current_selection = chosen_result["selection_metrics"]
                current_selection_ratios = chosen_result["selection_ratios"]
            discrete_repair_history.append(
                {
                    "round": repair_round,
                    "source_selection_ratios": round_source_selection_ratios,
                    "gradient_loss_history": gradient_loss_history,
                    "gradient_statistics": gradient_statistics,
                    "candidate_results": candidate_results,
                    "chosen_candidate": chosen_key,
                    "selection_worst_ratio_improvement": improvement,
                    "accepted": accepted,
                    "accepted_selection_ratios": (
                        current_selection_ratios if accepted else None
                    ),
                }
            )
            log(
                f"discrete repair round={repair_round} chosen={chosen_key} "
                f"improvement={improvement:.8f} accepted={accepted}"
            )
            if not accepted:
                break
        initial_selection = current_selection
    else:
        initial_selection = pre_repair_selection

    proxy_optimizer_group = {
        "params": [proxy.proxy_code for proxy in proxies.values()],
        "lr": args.proxy_lr,
    }
    scale_optimizer_group = {
        "params": [proxy.group_scale for proxy in proxies.values()],
        "lr": args.scale_lr,
    }
    optimizer_groups = [proxy_optimizer_group, scale_optimizer_group]
    if compensation_proxies:
        optimizer_groups.append(
            {
                "params": [
                    proxy.group_scale for proxy in compensation_proxies.values()
                ],
                "lr": args.compensation_scale_lr,
            }
        )
    norm_parameters = []
    if args.norm_lr > 0:
        norm_parameters = list(model.model.layers[layer_index].input_layernorm.parameters())
        norm_parameters += list(
            model.model.layers[layer_index].post_attention_layernorm.parameters()
        )
        for parameter in norm_parameters:
            parameter.requires_grad_(True)
        optimizer_groups.append({"params": norm_parameters, "lr": args.norm_lr})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.0)

    def snapshot_state():
        return {
            "codes": {
                name: proxy.hard_codes().cpu() for name, proxy in all_proxies.items()
            },
            "scales": {
                name: proxy.group_scale.detach().cpu().clone()
                for name, proxy in all_proxies.items()
            },
            "norms": [parameter.detach().cpu().clone() for parameter in norm_parameters],
        }

    best_selection_ratios = ratios(initial_selection, selection_baseline)
    best_selection_step = 0
    best_state = snapshot_state()
    wal_path = PROJECT / f"results/{args.tag}.wal.jsonl"
    if wal_path.exists():
        wal_path.unlink()
    wal = HashChainWAL(wal_path)
    transaction_id = f"{args.tag}-proxy-codes"
    wal.append(
        "begin",
        transaction_id,
        {
            "source_sha256": source_hash,
            "source_layers": sorted(source_layer_indices),
            "target_layer": layer_index,
            "target_projections": requested,
            "add_target_layer": args.add_target_layer,
            "compensate_source_scales": args.compensate_source_scales,
            "matrices": list(proxies),
            "committed_groups": {
                name: int(proxy.committed_mask.sum().item())
                for name, proxy in proxies.items()
            },
            "partial_proxy": any(
                not bool(proxy.committed_mask.all()) for proxy in proxies.values()
            ),
            "scale_compensation_matrices": list(compensation_proxies),
            "proxy_lr": args.proxy_lr,
            "scale_lr": args.scale_lr,
            "proxy_start_step": args.proxy_start_step,
            "proxy_freeze_step": args.proxy_freeze_step,
            "scale_start_step": args.scale_start_step,
            "compensation_scale_lr": args.compensation_scale_lr,
            "norm_lr": args.norm_lr,
            "block_weight": args.block_weight,
            "attention_weight": args.attention_weight,
            "ce_weight": args.ce_weight,
            "proxy_anchor_weight": args.proxy_anchor_weight,
            "kd_weight": args.kd_weight,
            "kd_topk": args.kd_topk,
            "kd_stride": args.kd_stride,
            "kd_temperature": args.kd_temperature,
            "selection_gate_sequences": args.selection_gate_sequences,
            "selection_suite_sha256": selection_suite_hash,
            "initial_selection_ratios": best_selection_ratios,
            "discrete_repair": {
                "rounds": args.discrete_repair_rounds,
                "sequences": args.discrete_repair_sequences,
                "domain": args.discrete_repair_domain,
                "group_budgets": repair_budgets,
                "scale_modes": repair_scale_modes,
                "ce_weight": args.discrete_repair_ce_weight,
                "block_weight": args.discrete_repair_block_weight,
                "kd_weight": args.discrete_repair_kd_weight,
                "min_improvement": args.discrete_repair_min_improvement,
            },
        },
    )
    history = []
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model.train()
    try:
        for step in range(1, args.steps + 1):
            proxy_active = step >= args.proxy_start_step and (
                args.proxy_freeze_step == 0 or step <= args.proxy_freeze_step
            )
            scale_active = step >= args.scale_start_step
            optimizer.param_groups[0]["lr"] = args.proxy_lr if proxy_active else 0.0
            optimizer.param_groups[1]["lr"] = args.scale_lr if scale_active else 0.0
            phase = step / max(args.steps, 1)
            cosine = 0.5 * (1 + math.cos(math.pi * phase))
            temperature = args.final_temperature + (
                args.initial_temperature - args.final_temperature
            ) * cosine
            for proxy in proxies.values():
                proxy.temperature = temperature
            item = (step - 1) % len(calibration)
            batch = calibration[item].unsqueeze(0).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            captures.clear()
            logits = model(input_ids=batch[:, :-1], use_cache=False).logits
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                batch[:, 1:].reshape(-1),
            )
            block_loss = normalized_mse(captures["block"], teacher[item]["block"])
            attention_loss = normalized_mse(
                captures["attention"], teacher[item]["attention"]
            )
            anchor_loss = torch.stack(
                [
                    proxy.proxy_anchor_loss()
                    for proxy in proxies.values()
                ]
            ).mean()
            kd_loss = (
                bucket_kl(logits, teacher[item], args)
                if args.kd_weight > 0
                else logits.new_zeros(())
            )
            loss = (
                args.ce_weight * ce
                + args.block_weight * block_loss
                + args.attention_weight * attention_loss
                + args.proxy_anchor_weight * anchor_loss
                + args.kd_weight * kd_loss
            )
            loss.backward()
            parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            for proxy in proxies.values():
                proxy.constrain_()
            report_step = (
                step == 1
                or step == args.steps
                or step % max(args.steps // 4, 1) == 0
            )
            selection_metrics = None
            selection_ratios = None
            if step % max(args.selection_every, 1) == 0 or step == args.steps:
                model.eval()
                selection_metrics = evaluate_domains(
                    model, selection_gates, args.device
                )
                selection_ratios = ratios(selection_metrics, selection_baseline)
                selection_objective = max(selection_ratios.values())
                if selection_objective < max(best_selection_ratios.values()):
                    best_selection_ratios = selection_ratios
                    best_selection_step = step
                    best_state = snapshot_state()
                model.train()
            if report_step:
                entry = {
                    "step": step,
                    "ce": float(ce.item()),
                    "block_loss": float(block_loss.item()),
                    "attention_loss": float(attention_loss.item()),
                    "anchor_loss": float(anchor_loss.item()),
                    "kd_loss": float(kd_loss.item()),
                    "temperature": temperature,
                    "proxy_active": proxy_active,
                    "scale_active": scale_active,
                    "gradient_norm": float(gradient_norm),
                    "code_churn": {
                        name: proxy.code_churn() for name, proxy in proxies.items()
                    },
                    "deployment_statistics": {
                        name: proxy.deployment_statistics()
                        for name, proxy in proxies.items()
                    },
                    "selection_metrics": selection_metrics,
                    "selection_ratios": selection_ratios,
                    "best_selection_step": best_selection_step,
                }
                history.append(entry)
                wal.append("progress", transaction_id, entry)
                log(
                    f"step={step}/{args.steps} ce={ce.item():.4f} "
                    f"block={block_loss.item():.4f} attn={attention_loss.item():.4f}"
                )
    finally:
        for handle in handles:
            handle.remove()
        for hook in proxy_gradient_hooks:
            hook.remove()
    model.eval()

    with torch.no_grad():
        for name, proxy in all_proxies.items():
            proxy.proxy_code.copy_(best_state["codes"][name].to(args.device).float())
            proxy.group_scale.copy_(best_state["scales"][name].to(args.device))
        for parameter, value in zip(norm_parameters, best_state["norms"]):
            parameter.copy_(value.to(parameter.device, parameter.dtype))

    churn = {name: proxy.code_churn() for name, proxy in proxies.items()}
    with torch.no_grad():
        for name in all_proxies:
            matrix = matrices[name]
            proxy = all_proxies[name]
            codes = proxy.hard_codes()
            scales = proxy.group_scale.detach().abs().clamp_min(1e-5)
            committed = matrix.committed_mask
            matrix.committed_codes[committed] = codes[committed]
            matrix.group_scale[committed] = scales[committed]
            grouped_hard = codes.float() * scales.unsqueeze(-1)
            padded = F.pad(
                matrix.master_weight.detach(), (0, matrix.padding)
            ).view_as(matrix.committed_codes).clone()
            padded[committed] = grouped_hard[committed]
            matrix.master_weight.copy_(
                padded.reshape(matrix.out_features, -1)[:, : matrix.in_features]
            )
            target = model.get_submodule(name)
            from wal_tat import TransactionalTernaryLinear

            set_submodule(
                model, name, TransactionalTernaryLinear(matrix, target.bias).to(args.device)
            )

    attempted = evaluate_domains(model, gates, args.device)
    attempted_ratios = ratios(attempted, baseline)
    starting_objective = max(initial_ratios.values())
    attempted_objective = max(attempted_ratios.values())
    passed = (
        (
            args.allow_within_gate
            or attempted_objective <= starting_objective - args.min_improvement
        )
        and all(value <= args.gate_ratio for value in attempted_ratios.values())
    )
    wal.append(
        "commit" if passed else "rollback",
        transaction_id,
        {
            "source_frontier_ratios": starting_ratios,
            "initial_hard_target_ratios": initial_ratios,
            "attempted_ratios": attempted_ratios,
            "code_churn": churn,
        },
    )

    checkpoint_path = WORKSPACE / f"wal2/checkpoints/wal-tat-{args.tag}-proxy-codes.pt"
    fresh = None
    if passed:
        payload = checkpoint_payload(
            model,
            matrices,
            source_hash,
            attempted,
            attempted_ratios,
            {
                "experiment": args.tag,
                "mode": "hard-forward-soft-backward-proxy-code-recovery",
                "target_layer": layer_index,
                "target_projections": requested,
                "added_target_layer": args.add_target_layer,
                "completed_target_projections": args.complete_target_projections,
                "completion_initializer": args.completion_initializer,
                "completion_moment_sequences": args.completion_moment_sequences,
                "discrete_repair_rounds": args.discrete_repair_rounds,
                "new_groups": {
                    name: int(mask.sum().item())
                    for name, mask in completion_masks.items()
                },
                "steps": args.steps,
                "proxy_lr": args.proxy_lr,
                "scale_lr": args.scale_lr,
                "proxy_start_step": args.proxy_start_step,
                "proxy_freeze_step": args.proxy_freeze_step,
                "scale_start_step": args.scale_start_step,
                "ce_weight": args.ce_weight,
                "kd_weight": args.kd_weight,
                "kd_topk": args.kd_topk,
                "kd_stride": args.kd_stride,
                "kd_temperature": args.kd_temperature,
            },
        )
        torch.save(payload, checkpoint_path)
        fresh = fresh_verify(checkpoint_path, model_path, suite, args)
        if not fresh["passed"]:
            checkpoint_path.unlink()
            passed = False

    result = {
        "schema": "wal-tat-block-proxy-recovery-v1",
        "tag": args.tag,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_sha256": source_hash,
        "suite": str(args.suite.resolve()),
        "suite_sha256": suite_hash,
        "selection_suite": str(selection_suite_path.resolve()),
        "selection_suite_sha256": selection_suite_hash,
        "steps": args.steps,
        "proxy_lr": args.proxy_lr,
        "scale_lr": args.scale_lr,
        "proxy_start_step": args.proxy_start_step,
        "proxy_freeze_step": args.proxy_freeze_step,
        "scale_start_step": args.scale_start_step,
        "compensation_scale_lr": args.compensation_scale_lr,
        "norm_lr": args.norm_lr,
        "block_weight": args.block_weight,
        "attention_weight": args.attention_weight,
        "ce_weight": args.ce_weight,
        "allow_within_gate": args.allow_within_gate,
        "proxy_anchor_weight": args.proxy_anchor_weight,
        "kd_weight": args.kd_weight,
        "kd_topk": args.kd_topk,
        "kd_stride": args.kd_stride,
        "kd_temperature": args.kd_temperature,
        "selection_gate_sequences": args.selection_gate_sequences,
        "selection_baseline": selection_baseline,
        "pre_repair_selection": pre_repair_selection,
        "pre_repair_selection_ratios": pre_repair_selection_ratios,
        "initial_selection": initial_selection,
        "best_selection_step": best_selection_step,
        "best_selection_ratios": best_selection_ratios,
        "target_layer": layer_index,
        "target_projections": requested,
        "target_committed_coverage": {
            name: float(proxy.committed_mask.float().mean().item())
            for name, proxy in proxies.items()
        },
        "partial_proxy": any(
            not bool(proxy.committed_mask.all()) for proxy in proxies.values()
        ),
        "added_target_layer": args.add_target_layer,
        "completed_target_projections": args.complete_target_projections,
        "completion_initializer": args.completion_initializer,
        "completion_moment_sequences": args.completion_moment_sequences,
        "discrete_repair": {
            "rounds_requested": args.discrete_repair_rounds,
            "sequences": args.discrete_repair_sequences,
            "domain": args.discrete_repair_domain,
            "calibration_indices": list(
                repair_calibration_indices[: args.discrete_repair_sequences]
            ),
            "group_budgets": list(repair_budgets),
            "scale_modes": list(repair_scale_modes),
            "ce_weight": args.discrete_repair_ce_weight,
            "block_weight": args.discrete_repair_block_weight,
            "kd_weight": args.discrete_repair_kd_weight,
            "min_improvement": args.discrete_repair_min_improvement,
            "history": discrete_repair_history,
        },
        "new_groups": {
            name: int(mask.sum().item()) for name, mask in completion_masks.items()
        },
        "new_weights": {
            name: int(mask.sum().item() * matrices[name].group_size)
            for name, mask in completion_masks.items()
        },
        "compensate_source_scales": args.compensate_source_scales,
        "scale_compensation_matrices": list(compensation_proxies),
        "source_frontier": starting,
        "source_frontier_ratios": starting_ratios,
        "initial_hard_target": initial,
        "initial_hard_target_ratios": initial_ratios,
        "attempted": attempted,
        "attempted_ratios": attempted_ratios,
        "code_churn": churn,
        "deployment_statistics": {
            name: proxy.deployment_statistics() for name, proxy in proxies.items()
        },
        "history": history,
        "passed": passed,
        "fresh_verify": fresh,
        "accepted_checkpoint": str(checkpoint_path) if passed else None,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else None
        ),
        "wal": str(wal_path),
    }
    result_path = PROJECT / f"results/{args.tag}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {result_path}; passed={passed} ratios={attempted_ratios} churn={churn}")
    del teacher, model, matrices, proxies
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
