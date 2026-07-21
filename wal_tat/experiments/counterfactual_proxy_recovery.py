"""Target-local hard-forward/soft-backward ternary code recovery.

The source checkpoint is immutable.  A counterfactual teacher restores only
the committed groups of one target matrix to the original BF16 weights.  The
student always executes strict {-1, 0, +1} codes and FP16-rounded g128 scales,
while a smooth proxy supplies gradients.  Model selection and confirmation
use disjoint development slices; a passing run writes an auditable recode
artifact, never a checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
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
from counterfactual_scale_recovery import slice_domain_gates
from matched_bf16_residual_recovery import cache_teacher, normalized_mse
from wal_tat import ProxyTernaryLinear, ProxyTernaryMatrix


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
    parser.add_argument(
        "--starting-artifact",
        type=Path,
        help="optional frozen strict-ternary recode used as the development start",
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-layer", type=int, default=24)
    parser.add_argument(
        "--target-projection", choices=tuple(PROJECTION_MAP), default="up_proj"
    )
    parser.add_argument("--steps", type=int, default=1536)
    parser.add_argument("--proxy-lr", type=float, default=1.5e-3)
    parser.add_argument("--scale-lr", type=float, default=7.5e-6)
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--block-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--proxy-anchor-weight", type=float, default=0.05)
    parser.add_argument("--kd-topk", type=int, default=64)
    parser.add_argument("--kd-stride", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--initial-temperature", type=float, default=0.50)
    parser.add_argument("--final-temperature", type=float, default=0.08)
    parser.add_argument("--selection-gate-start", type=int, default=0)
    parser.add_argument("--selection-gate-sequences", type=int, default=128)
    parser.add_argument("--confirmation-gate-start", type=int, default=128)
    parser.add_argument("--confirmation-gate-sequences", type=int, default=128)
    parser.add_argument("--selection-every", type=int, default=256)
    parser.add_argument("--min-selection-improvement", type=float, default=1e-3)
    parser.add_argument("--min-confirmation-improvement", type=float, default=0.0)
    parser.add_argument("--max-code-churn", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=233)
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


@torch.no_grad()
def snapshot_proxy(proxy: ProxyTernaryMatrix) -> dict[str, torch.Tensor]:
    return {
        "proxy_code": proxy.proxy_code.detach().cpu().clone(),
        "group_scale": proxy.group_scale.detach().cpu().clone(),
    }


@torch.no_grad()
def restore_proxy(proxy: ProxyTernaryMatrix, state: dict[str, torch.Tensor]) -> None:
    proxy.proxy_code.copy_(state["proxy_code"].to(proxy.proxy_code.device))
    proxy.group_scale.copy_(state["group_scale"].to(proxy.group_scale.device))


def load_starting_artifact(
    path: Path,
    *,
    checkpoint_hash: str,
    target_name: str,
    matrix,
    device: str,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("format") != "wal-tat-ternary-recode-v1":
        raise ValueError("unsupported starting artifact format")
    if artifact.get("source_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("starting artifact source checkpoint mismatch")
    if artifact.get("target_name") != target_name:
        raise ValueError("starting artifact target mismatch")
    mask = artifact["committed_mask"].to(device)
    codes = artifact["ternary_codes_int8"].to(device, dtype=torch.int8)
    scales = artifact["scales_fp16"].to(device).float()
    if not torch.equal(mask, matrix.committed_mask):
        raise ValueError("starting artifact committed mask mismatch")
    if codes.shape != matrix.committed_codes.shape:
        raise ValueError("starting artifact code shape mismatch")
    if scales.shape != matrix.group_scale.shape:
        raise ValueError("starting artifact scale shape mismatch")
    if not set(codes.unique().tolist()) <= {-1, 0, 1}:
        raise ValueError("starting artifact contains non-ternary codes")
    if not torch.equal(codes[~mask], matrix.committed_codes[~mask]):
        raise ValueError("starting artifact changes codes outside committed mask")
    if not torch.equal(scales[~mask], matrix.group_scale[~mask]):
        raise ValueError("starting artifact changes scales outside committed mask")
    return artifact, codes, scales


@torch.no_grad()
def change_statistics(
    reference_codes: torch.Tensor,
    reference_scales: torch.Tensor,
    candidate_codes: torch.Tensor,
    candidate_scales: torch.Tensor,
    committed_mask: torch.Tensor,
) -> dict:
    active_codes = committed_mask.unsqueeze(-1).expand_as(reference_codes)
    changed_codes = (candidate_codes != reference_codes) & active_codes
    changed_scales = (candidate_scales.float() != reference_scales) & committed_mask
    return {
        "changed_code_values": int(changed_codes.sum().item()),
        "changed_scale_groups": int(changed_scales.sum().item()),
        "code_churn": float(changed_codes.sum().div(active_codes.sum()).item()),
    }


@torch.no_grad()
def deployed_representation(
    proxy: ProxyTernaryMatrix,
    source_codes: torch.Tensor,
    source_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    codes = proxy.hard_codes()
    scales = proxy.group_scale.detach().abs().clamp_min(1e-5).half()
    statistics = {
        **change_statistics(
            source_codes,
            source_scales,
            codes,
            scales,
            proxy.committed_mask,
        ),
        "deployment": proxy.deployment_statistics(),
    }
    return codes, scales, statistics


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.selection_every < 1:
        raise ValueError("steps and selection interval must be positive")
    if args.initial_temperature <= 0 or args.final_temperature <= 0:
        raise ValueError("temperatures must be positive")
    validate_disjoint_slices(args)
    source_path = args.source_checkpoint.resolve()
    source_hash = sha256_file(source_path)
    suite_path = args.suite.resolve()
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
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    target_name = (
        f"model.layers.{args.target_layer}."
        f"{PROJECTION_MAP[args.target_projection]}"
    )
    if target_name not in payload["matrices"]:
        raise ValueError(f"target matrix is absent from checkpoint: {target_name}")

    seed_everything(args.seed)
    started = time.time()
    model = load_model(model_path, args.device)
    original_weight = model.get_submodule(target_name).weight.detach().cpu().clone()
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    confirmation_baseline = evaluate_domains(model, confirmation_gates, args.device)
    matrices = install_checkpoint(model, payload, args.device)
    matrix = matrices[target_name]
    checkpoint_codes = matrix.committed_codes.detach().clone()
    checkpoint_scales = matrix.group_scale.detach().clone()
    starting_artifact_path = (
        args.starting_artifact.resolve() if args.starting_artifact else None
    )
    starting_artifact = None
    starting_artifact_hash = None
    if starting_artifact_path is not None:
        starting_artifact, starting_codes, starting_scales = load_starting_artifact(
            starting_artifact_path,
            checkpoint_hash=source_hash,
            target_name=target_name,
            matrix=matrix,
            device=args.device,
        )
        with torch.no_grad():
            matrix.committed_codes.copy_(starting_codes)
            matrix.group_scale.copy_(starting_scales)
        starting_artifact_hash = sha256_file(starting_artifact_path)
    selection_source = evaluate_domains(model, selection_gates, args.device)
    confirmation_source = evaluate_domains(model, confirmation_gates, args.device)
    source_selection_ratios = ratios(selection_source, selection_baseline)
    source_confirmation_ratios = ratios(confirmation_source, confirmation_baseline)

    restore_original_committed_groups(matrix, original_weight)
    teacher_selection = evaluate_domains(model, selection_gates, args.device)
    teacher_confirmation = evaluate_domains(model, confirmation_gates, args.device)
    teacher = cache_teacher(model, calibration, args.target_layer, args.device, args)
    restore_source_state(model, matrices, payload, args.device)
    if starting_artifact is not None:
        with torch.no_grad():
            matrix.committed_codes.copy_(starting_codes)
            matrix.group_scale.copy_(starting_scales)
    source_recheck = evaluate_domains(model, confirmation_gates, args.device)
    source_restoration_exact = all(
        source_recheck[domain]["nll"] == confirmation_source[domain]["nll"]
        for domain in confirmation_gates
    )
    if not source_restoration_exact:
        raise RuntimeError("source did not restore exactly after teacher cache")

    source_codes = matrix.committed_codes.detach().clone()
    source_scales = matrix.group_scale.detach().clone()
    target = model.get_submodule(target_name)
    proxy = ProxyTernaryMatrix(
        source_codes,
        source_scales,
        compute_dtype=matrix.compute_dtype,
        temperature=args.initial_temperature,
        committed_mask=matrix.committed_mask,
        master_weight=matrix.master_weight,
        fake_fp16_scale=True,
    ).to(args.device)
    set_submodule(model, target_name, ProxyTernaryLinear(proxy, target.bias).to(args.device))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    proxy.proxy_code.requires_grad_(True)
    proxy.group_scale.requires_grad_(True)

    initial_selection = evaluate_domains(model, selection_gates, args.device)
    initial_selection_ratios = ratios(initial_selection, selection_baseline)
    best_objective = max(initial_selection_ratios.values())
    best_selection_ratios = initial_selection_ratios
    best_step = 0
    best_state = snapshot_proxy(proxy)
    capture: dict[str, torch.Tensor] = {}
    handle = model.model.layers[args.target_layer].register_forward_hook(
        lambda _module, _inputs, output: capture.__setitem__(
            "block", tensor_output(output)
        )
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [proxy.proxy_code], "lr": args.proxy_lr},
            {"params": [proxy.group_scale], "lr": args.scale_lr},
        ],
        weight_decay=0.0,
    )
    history = []
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model.train()
    try:
        for step in range(1, args.steps + 1):
            phase = step / args.steps
            cosine = 0.5 * (1 + math.cos(math.pi * phase))
            proxy.temperature = args.final_temperature + (
                args.initial_temperature - args.final_temperature
            ) * cosine
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
            anchor_loss = proxy.proxy_anchor_loss()
            loss = (
                args.ce_weight * ce
                + args.block_weight * block_loss
                + args.kd_weight * kd_loss
                + args.proxy_anchor_weight * anchor_loss
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [proxy.proxy_code, proxy.group_scale], 1.0
            )
            optimizer.step()
            proxy.constrain_()

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
                    best_selection_ratios = selection_ratios
                    best_step = step
                    best_state = snapshot_proxy(proxy)
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
                    "temperature": proxy.temperature,
                    "gradient_norm": float(gradient_norm),
                    "selection_ratios": selection_ratios,
                    "best_step": best_step,
                    "code_churn": proxy.code_churn(),
                    "deployment": proxy.deployment_statistics(),
                }
                history.append(entry)
                print(
                    f"step={step}/{args.steps} best={best_step} "
                    f"churn={entry['code_churn']:.6f} ratios={selection_ratios}",
                    flush=True,
                )
    finally:
        handle.remove()

    model.eval()
    restore_proxy(proxy, best_state)
    candidate_codes, candidate_scales, representation = deployed_representation(
        proxy, source_codes, source_scales
    )
    if starting_artifact is not None:
        representation["incremental_from_starting_artifact"] = {
            key: representation[key]
            for key in ("changed_code_values", "changed_scale_groups", "code_churn")
        }
        total = change_statistics(
            checkpoint_codes,
            checkpoint_scales,
            candidate_codes,
            candidate_scales,
            proxy.committed_mask,
        )
        representation.update(total)
    with torch.no_grad():
        proxy.proxy_code.copy_(candidate_codes.float())
        proxy.group_scale.copy_(candidate_scales.float())
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
    passed = (
        selection_improvement >= args.min_selection_improvement
        and confirmation_improvement >= args.min_confirmation_improvement
        and representation["code_churn"] <= args.max_code_churn
    )

    artifact_path = WORKSPACE / f"wal2/artifacts/wal-tat-{args.tag}.pt"
    artifact_sha256 = None
    if passed:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "wal-tat-ternary-recode-v1",
                "source_checkpoint_sha256": sha256_file(source_path),
                "starting_artifact_sha256": starting_artifact_hash,
                "suite_sha256": sha256_file(suite_path),
                "target_name": target_name,
                "candidate": f"proxy_step_{best_step}",
                "recipe": "target_counterfactual_hard_forward_soft_backward",
                "residual_gain": 1.0,
                "committed_mask": proxy.committed_mask.cpu(),
                "ternary_codes_int8": candidate_codes.cpu().to(torch.int8),
                "scales_fp16": candidate_scales.cpu().half(),
                "selection_ratios": attempted_selection_ratios,
                "confirmation_ratios": attempted_confirmation_ratios,
                "representation": representation,
            },
            artifact_path,
        )
        artifact_sha256 = sha256_file(artifact_path)

    restore_proxy(
        proxy,
        {"proxy_code": source_codes.float().cpu(), "group_scale": source_scales.cpu()},
    )
    final_source = evaluate_domains(model, confirmation_gates, args.device)
    final_restoration_exact = all(
        final_source[domain]["nll"] == confirmation_source[domain]["nll"]
        for domain in confirmation_gates
    )
    result = {
        "schema": "wal-tat-counterfactual-proxy-recovery-v1",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "starting_artifact": (
            str(starting_artifact_path) if starting_artifact_path else None
        ),
        "starting_artifact_sha256": starting_artifact_hash,
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": "development selection and disjoint confirmation; not final audit",
        "target_name": target_name,
        "teacher_source": "target_counterfactual",
        "teacher_selection_ratios": ratios(teacher_selection, selection_baseline),
        "teacher_confirmation_ratios": ratios(
            teacher_confirmation, confirmation_baseline
        ),
        "steps": args.steps,
        "proxy_lr": args.proxy_lr,
        "scale_lr": args.scale_lr,
        "loss_weights": {
            "ce": args.ce_weight,
            "block": args.block_weight,
            "kd": args.kd_weight,
            "proxy_anchor": args.proxy_anchor_weight,
        },
        "temperatures": {
            "initial": args.initial_temperature,
            "final": args.final_temperature,
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
        "initial_selection_ratios": initial_selection_ratios,
        "best_step": best_step,
        "best_selection_ratios": best_selection_ratios,
        "attempted_selection_ratios": attempted_selection_ratios,
        "attempted_confirmation_ratios": attempted_confirmation_ratios,
        "selection_worst_ratio_improvement": selection_improvement,
        "confirmation_worst_ratio_improvement": confirmation_improvement,
        "min_selection_improvement": args.min_selection_improvement,
        "min_confirmation_improvement": args.min_confirmation_improvement,
        "max_code_churn": args.max_code_churn,
        "representation": representation,
        "codes_strict_ternary": bool(
            torch.all((candidate_codes >= -1) & (candidate_codes <= 1)).item()
        ),
        "committed_mask_unchanged": torch.equal(
            proxy.committed_mask,
            payload["matrices"][target_name]["committed_mask"].to(args.device),
        ),
        "persistent_bf16_residual": False,
        "coverage_changed": False,
        "passed": passed,
        "artifact": str(artifact_path) if passed else None,
        "artifact_sha256": artifact_sha256,
        "checkpoint_written": False,
        "source_restoration_exact": source_restoration_exact,
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
        f"churn={representation['code_churn']}",
        flush=True,
    )
    if not final_restoration_exact:
        raise SystemExit(2)
    del model, matrices, proxy, teacher, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
