"""Recover gate margin by optimizing fixed-code scales of one ternary MLP."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from wal_tat import AtomicTernaryTransaction, HashChainWAL

from compensation_window import (
    DOWN_NAME,
    PROJECT,
    UP_NAME,
    WORKSPACE,
    bucket_kl,
    cache_teacher,
    checkpoint_payload,
    default_model_path,
    evaluate_domains,
    fresh_verify,
    install_checkpoint,
    load_model,
    log,
    ratios,
    seed_everything,
    sha256_file,
    tensor_output,
)


GATE_NAME = "model.layers.27.mlp.gate_proj"
MATRIX_NAMES = (DOWN_NAME, UP_NAME, GATE_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--suite",
        type=Path,
        default=WORKSPACE / "wal2/cache/wal-tat-suite-v1-l128-wc8-cc8-g256.pt",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", default="mlp_scale_recovery_v1")
    parser.add_argument(
        "--all-checkpoint-matrices",
        action="store_true",
        help="recover every matrix stored in the checkpoint instead of only the MLP",
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gate-ratio", type=float, default=1.02)
    parser.add_argument("--objective-domain", default="code")
    parser.add_argument("--min-improvement", type=float, default=1e-5)
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument("--kd-topk", type=int, default=64)
    parser.add_argument("--kd-stride", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--hidden-kd-weight", type=float, default=0.0)
    parser.add_argument("--hidden-module", default="model.layers.27.mlp")
    parser.add_argument(
        "--teacher-source", choices=("frontier", "bf16"), default="bf16"
    )
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def cache_optional_teacher(model, calibration, args):
    if args.kd_weight <= 0 and args.hidden_kd_weight <= 0:
        return None
    return cache_teacher(model, calibration, args)


def train_fixed_scales(
    model, atomic, matrices, matrix_names, calibration, teacher, args, wal
):
    parameters = [matrices[name].group_scale for name in matrix_names]
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    student_hidden = {}
    handle = None
    if args.hidden_kd_weight > 0:
        handle = model.get_submodule(args.hidden_module).register_forward_hook(
            lambda _module, _inputs, output: student_hidden.__setitem__(
                "value", tensor_output(output)
            )
        )
    history = []
    model.train()
    try:
        for step in range(1, args.steps + 1):
            item = (step - 1) % len(calibration)
            batch = calibration[item].unsqueeze(0).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            student_hidden.clear()
            logits = model(input_ids=batch[:, :-1], use_cache=False).logits
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), batch[:, 1:].reshape(-1)
            )
            kd = torch.zeros((), device=logits.device)
            hidden_kd = torch.zeros((), device=logits.device)
            if teacher is not None and args.kd_weight > 0:
                kd = bucket_kl(logits, teacher[item], args)
            if teacher is not None and args.hidden_kd_weight > 0:
                student = student_hidden["value"].float()
                target = teacher[item]["hidden"].to(student.device).float().unsqueeze(0)
                hidden_kd = F.mse_loss(student, target) / target.square().mean().clamp_min(1e-8)
            loss = ce + args.kd_weight * kd + args.hidden_kd_weight * hidden_kd
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step == 1 or step == args.steps or step % max(args.steps // 4, 1) == 0:
                entry = {
                    "step": step,
                    "loss": float(loss.item()),
                    "ce": float(ce.item()),
                    "kd": float(kd.item()),
                    "hidden_kd": float(hidden_kd.item()),
                    "gradient_norm": float(gradient_norm),
                    "code_churn": atomic.current_code_churn(),
                }
                history.append(entry)
                wal.append("progress", atomic.transaction_id, entry)
                log(
                    f"step={step}/{args.steps} ce={ce.item():.4f} "
                    f"kd={kd.item():.4f} hidden={hidden_kd.item():.4f}"
                )
    finally:
        if handle is not None:
            handle.remove()
        for parameter in parameters:
            parameter.requires_grad_(False)
    model.eval()
    return history


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.min_improvement < 0:
        raise ValueError("steps must be positive and min-improvement non-negative")
    model_path = (args.model_path or default_model_path()).resolve()
    source_hash = sha256_file(args.source_checkpoint)
    suite_hash = sha256_file(args.suite)
    source_payload = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    suite = torch.load(args.suite, map_location="cpu", weights_only=False)
    calibration, gates = suite["calibration"], suite["gates"]
    if args.objective_domain != "max" and args.objective_domain not in gates:
        raise ValueError(f"unknown objective domain: {args.objective_domain}")

    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    teacher = (
        cache_optional_teacher(model, calibration, args)
        if args.teacher_source == "bf16"
        else None
    )
    matrices = install_checkpoint(model, source_payload, args.device)
    matrix_names = (
        tuple(source_payload["matrices"])
        if args.all_checkpoint_matrices
        else MATRIX_NAMES
    )
    if any(name not in matrices for name in matrix_names):
        raise ValueError("checkpoint does not contain every requested recovery matrix")
    starting = evaluate_domains(model, gates, args.device)
    starting_ratios = ratios(starting, baseline)
    if teacher is None and args.teacher_source == "frontier":
        teacher = cache_optional_teacher(model, calibration, args)

    active = {name: matrices[name] for name in matrix_names}
    masks = {name: matrix.committed_mask.detach().clone() for name, matrix in active.items()}
    frozen_codes = {
        name: matrix.committed_codes.detach().clone() for name, matrix in active.items()
    }
    frozen_scales = {
        name: matrix.group_scale.detach().clone() for name, matrix in active.items()
    }
    transaction_id = f"{args.tag}-fixed-scales"
    atomic = AtomicTernaryTransaction(active)
    atomic.begin(masks, reopen=set(active), transaction_id=transaction_id)
    for name, matrix in active.items():
        matrix.set_candidate_codes(frozen_codes[name], frozen_scales[name])
    atomic.set_candidate_state(1.0, 0.0)

    wal_path = PROJECT / f"results/{args.tag}.wal.jsonl"
    if wal_path.exists():
        wal_path.unlink()
    wal = HashChainWAL(wal_path)
    wal.append(
        "begin",
        transaction_id,
        {
            "source_sha256": source_hash,
            "fixed_code_groups": {name: int(mask.sum()) for name, mask in masks.items()},
            "teacher_source": args.teacher_source,
            "kd_weight": args.kd_weight,
            "hidden_kd_weight": args.hidden_kd_weight,
        },
    )
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    history = train_fixed_scales(
        model, atomic, matrices, matrix_names, calibration, teacher, args, wal
    )
    attempted = evaluate_domains(model, gates, args.device)
    attempted_ratios = ratios(attempted, baseline)
    starting_objective = (
        max(starting_ratios.values())
        if args.objective_domain == "max"
        else starting_ratios[args.objective_domain]
    )
    attempted_objective = (
        max(attempted_ratios.values())
        if args.objective_domain == "max"
        else attempted_ratios[args.objective_domain]
    )
    improved = attempted_objective <= starting_objective - args.min_improvement
    passed = improved and all(value <= args.gate_ratio for value in attempted_ratios.values())
    wal.append(
        "commit_intent" if passed else "rollback_intent",
        transaction_id,
        {
            "starting_ratios": starting_ratios,
            "attempted_ratios": attempted_ratios,
            "starting_objective": starting_objective,
            "attempted_objective": attempted_objective,
            "improved": improved,
        },
    )
    churn = atomic.current_code_churn()
    if passed:
        atomic_result = atomic.commit()
        terminal = "commit"
    else:
        atomic_result = atomic.rollback()
        terminal = "rollback"
    deployed = evaluate_domains(model, gates, args.device)
    deployed_ratios = ratios(deployed, baseline)
    wal.append(
        terminal,
        transaction_id,
        {"ratios": deployed_ratios, "code_churn": churn, "result": atomic_result},
    )

    checkpoint_path = WORKSPACE / f"wal2/checkpoints/wal-tat-{args.tag}-fixed-scales.pt"
    fresh = None
    if passed:
        payload = checkpoint_payload(
            model,
            matrices,
            source_hash,
            deployed,
            deployed_ratios,
            {
                "experiment": args.tag,
                "mode": "fixed-code-scale-recovery",
                "matrix_names": list(matrix_names),
                "teacher_source": args.teacher_source,
                "steps": args.steps,
                "lr": args.lr,
            },
        )
        torch.save(payload, checkpoint_path)
        fresh = fresh_verify(checkpoint_path, model_path, suite, args)
        if not fresh["passed"]:
            checkpoint_path.unlink()

    result = {
        "schema": "wal-tat-fixed-scale-recovery-v1",
        "tag": args.tag,
        "model": str(model_path),
        "source_checkpoint": str(args.source_checkpoint),
        "source_sha256": source_hash,
        "suite": str(args.suite),
        "suite_sha256": suite_hash,
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "teacher_source": args.teacher_source,
        "kd_weight": args.kd_weight,
        "hidden_kd_weight": args.hidden_kd_weight,
        "matrix_names": list(matrix_names),
        "objective_domain": args.objective_domain,
        "min_improvement": args.min_improvement,
        "starting_objective": starting_objective,
        "attempted_objective": attempted_objective,
        "baseline": baseline,
        "starting": starting,
        "starting_ratios": starting_ratios,
        "attempted": attempted,
        "attempted_ratios": attempted_ratios,
        "passed": passed,
        "deployed": deployed,
        "deployed_ratios": deployed_ratios,
        "code_churn": churn,
        "coverage": {
            name: float(matrix.committed_mask.float().mean())
            for name, matrix in matrices.items()
        },
        "history": history,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else None
        ),
        "fresh_verify": fresh,
        "accepted_checkpoint": (
            str(checkpoint_path) if fresh is not None and fresh["passed"] else None
        ),
        "wal": str(wal_path),
    }
    result_path = PROJECT / f"results/{args.tag}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(
        f"wrote {result_path}; passed={passed} fresh="
        f"{None if fresh is None else fresh['passed']} ratios={attempted_ratios}"
    )
    del teacher, model, matrices
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
