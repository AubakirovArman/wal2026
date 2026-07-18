"""Hard-forward proxy-code recovery for a fully ternary transformer block."""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from wal_tat import HashChainWAL, ProxyTernaryLinear, ProxyTernaryMatrix

from compensation_window import (
    PROJECT,
    WORKSPACE,
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
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
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--proxy-lr", type=float, default=7.5e-4)
    parser.add_argument("--scale-lr", type=float, default=5e-7)
    parser.add_argument("--compensation-scale-lr", type=float, default=1e-6)
    parser.add_argument("--norm-lr", type=float, default=0.0)
    parser.add_argument(
        "--compensate-source-scales",
        action="store_true",
        help="reopen every source matrix scale while freezing its ternary codes",
    )
    parser.add_argument("--block-weight", type=float, default=0.02)
    parser.add_argument("--attention-weight", type=float, default=0.02)
    parser.add_argument("--proxy-anchor-weight", type=float, default=0.0)
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
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def cache_hidden_teacher(model, calibration, layer_index, device):
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
            model(input_ids=batch[:, :-1], use_cache=False)
            result.append(
                {
                    "block": captures["block"].squeeze(0).half().cpu(),
                    "attention": captures["attention"].squeeze(0).half().cpu(),
                }
            )
            if (index + 1) % 64 == 0 or index + 1 == len(calibration):
                log(f"teacher hidden {index + 1}/{len(calibration)}")
    finally:
        for handle in handles:
            handle.remove()
    return result


def normalized_mse(student, teacher):
    target = teacher.to(student.device).float().unsqueeze(0)
    return F.mse_loss(student.float(), target) / target.square().mean().clamp_min(1e-8)


def main() -> None:
    args = parse_args()
    model_path = (args.model_path or default_model_path()).resolve()
    source_hash = sha256_file(args.source_checkpoint)
    suite_hash = sha256_file(args.suite)
    source_payload = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    suite = torch.load(args.suite, map_location="cpu", weights_only=False)
    calibration, gates = suite["calibration"], suite["gates"]
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
        name: chunks[: args.selection_gate_sequences] for name, chunks in gates.items()
    }
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    teacher = cache_hidden_teacher(model, calibration, layer_index, args.device)
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
    if args.add_target_layer:
        active_names = tuple(
            f"model.layers.{layer_index}.{projection}" for projection in projection_names
        )
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
        active_names = tuple(
            f"model.layers.{layer_index}.{projection}" for projection in projection_names
        )
        missing = set(active_names) - set(matrices)
        if missing:
            raise ValueError(
                f"target layer {layer_index} is missing {sorted(missing)}"
            )
        initial = starting
        initial_ratios = starting_ratios

    proxies = {}
    for name in active_names:
        matrix = matrices[name]
        proxy = ProxyTernaryMatrix(
            matrix.committed_codes,
            matrix.group_scale,
            compute_dtype=matrix.compute_dtype,
            temperature=args.initial_temperature,
        ).to(args.device)
        target = model.get_submodule(name)
        set_submodule(model, name, ProxyTernaryLinear(proxy, target.bias).to(args.device))
        proxies[name] = proxy

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
    optimizer_groups = [
            {"params": [proxy.proxy_code for proxy in proxies.values()], "lr": args.proxy_lr},
            {"params": [proxy.group_scale for proxy in proxies.values()], "lr": args.scale_lr},
    ]
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

    initial_selection = evaluate_domains(model, selection_gates, args.device)
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
            "scale_compensation_matrices": list(compensation_proxies),
            "proxy_lr": args.proxy_lr,
            "scale_lr": args.scale_lr,
            "compensation_scale_lr": args.compensation_scale_lr,
            "norm_lr": args.norm_lr,
            "block_weight": args.block_weight,
            "attention_weight": args.attention_weight,
            "proxy_anchor_weight": args.proxy_anchor_weight,
            "selection_gate_sequences": args.selection_gate_sequences,
            "initial_selection_ratios": best_selection_ratios,
        },
    )
    history = []
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model.train()
    try:
        for step in range(1, args.steps + 1):
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
                    (proxy.proxy_code - proxy.initial_codes.float()).square().mean()
                    for proxy in proxies.values()
                ]
            ).mean()
            loss = (
                ce
                + args.block_weight * block_loss
                + args.attention_weight * attention_loss
                + args.proxy_anchor_weight * anchor_loss
            )
            loss.backward()
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
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
                    "temperature": temperature,
                    "gradient_norm": float(gradient_norm),
                    "code_churn": {
                        name: proxy.code_churn() for name, proxy in proxies.items()
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
            matrix.committed_codes.copy_(codes)
            matrix.group_scale.copy_(scales)
            grouped_hard = codes.float() * scales.unsqueeze(-1)
            matrix.master_weight.copy_(
                grouped_hard.reshape(matrix.out_features, matrix.in_features)
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
                "steps": args.steps,
                "proxy_lr": args.proxy_lr,
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
        "steps": args.steps,
        "proxy_lr": args.proxy_lr,
        "scale_lr": args.scale_lr,
        "compensation_scale_lr": args.compensation_scale_lr,
        "norm_lr": args.norm_lr,
        "block_weight": args.block_weight,
        "attention_weight": args.attention_weight,
        "allow_within_gate": args.allow_within_gate,
        "proxy_anchor_weight": args.proxy_anchor_weight,
        "selection_gate_sequences": args.selection_gate_sequences,
        "selection_baseline": selection_baseline,
        "initial_selection": initial_selection,
        "best_selection_step": best_selection_step,
        "best_selection_ratios": best_selection_ratios,
        "target_layer": layer_index,
        "target_projections": requested,
        "added_target_layer": args.add_target_layer,
        "compensate_source_scales": args.compensate_source_scales,
        "scale_compensation_matrices": list(compensation_proxies),
        "source_frontier": starting,
        "source_frontier_ratios": starting_ratios,
        "initial_hard_target": initial,
        "initial_hard_target_ratios": initial_ratios,
        "attempted": attempted,
        "attempted_ratios": attempted_ratios,
        "code_churn": churn,
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
