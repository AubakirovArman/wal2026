"""Reopen and recover every ternary code in one completed transformer block."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from wal_tat import AtomicTernaryTransaction, HashChainWAL

from attention_vo_window import attention_extras
from compensation_window import (
    PROJECT,
    WORKSPACE,
    cache_teacher,
    checkpoint_payload,
    compensation_parameters,
    default_model_path,
    evaluate_domains,
    fresh_verify,
    install_checkpoint,
    load_model,
    log,
    ratios,
    seed_everything,
    sha256_file,
    train_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", default="block_code_recovery_v1")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--gate-ratio", type=float, default=1.06)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument("--kd-topk", type=int, default=64)
    parser.add_argument("--kd-stride", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--hidden-kd-weight", type=float, default=0.0)
    parser.add_argument("--hidden-module", default="model.layers.27")
    parser.add_argument(
        "--teacher-source", choices=("frontier", "bf16"), default="frontier"
    )
    parser.add_argument("--scale-only-compensation", action="store_false")
    parser.set_defaults(scale_only_compensation=False)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def unique_parameters(*groups):
    result, seen = [], set()
    for group in groups:
        for parameter in group:
            if id(parameter) not in seen:
                result.append(parameter)
                seen.add(id(parameter))
    return result


def main() -> None:
    args = parse_args()
    model_path = (args.model_path or default_model_path()).resolve()
    source_hash = sha256_file(args.source_checkpoint)
    suite_hash = sha256_file(args.suite)
    source_payload = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    suite = torch.load(args.suite, map_location="cpu", weights_only=False)
    calibration, gates = suite["calibration"], suite["gates"]

    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    teacher = (
        cache_teacher(model, calibration, args)
        if args.teacher_source == "bf16" and (args.kd_weight > 0 or args.hidden_kd_weight > 0)
        else None
    )
    matrices = install_checkpoint(model, source_payload, args.device)
    starting = evaluate_domains(model, gates, args.device)
    starting_ratios = ratios(starting, baseline)
    if teacher is None and (args.kd_weight > 0 or args.hidden_kd_weight > 0):
        teacher = cache_teacher(model, calibration, args)
    if teacher is None:
        # train_atomic always computes bucket KL; a frontier cache is cheap and
        # numerically neutral when its coefficient is zero.
        teacher = cache_teacher(model, calibration, args)

    active = dict(matrices)
    masks = {name: matrix.committed_mask.detach().clone() for name, matrix in active.items()}
    atomic = AtomicTernaryTransaction(active)
    transaction_id = f"{args.tag}-all-codes"
    atomic.begin(masks, reopen=set(active), transaction_id=transaction_id)
    extras = unique_parameters(compensation_parameters(model), attention_extras(model))
    extra_snapshots = [parameter.detach().clone() for parameter in extras]

    wal_path = PROJECT / f"results/{args.tag}.wal.jsonl"
    if wal_path.exists():
        wal_path.unlink()
    wal = HashChainWAL(wal_path)
    wal.append(
        "begin",
        transaction_id,
        {
            "source_sha256": source_hash,
            "matrix_groups": {name: int(mask.sum()) for name, mask in masks.items()},
            "steps": args.steps,
            "lr": args.lr,
            "teacher_source": args.teacher_source,
            "kd_weight": args.kd_weight,
            "hidden_kd_weight": args.hidden_kd_weight,
        },
    )
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    history = train_atomic(model, atomic, active, calibration, teacher, extras, args, wal)
    attempted = evaluate_domains(model, gates, args.device)
    attempted_ratios = ratios(attempted, baseline)
    starting_objective = max(starting_ratios.values())
    attempted_objective = max(attempted_ratios.values())
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
        },
    )
    churn = atomic.current_code_churn()
    if passed:
        atomic_result = atomic.commit()
        terminal = "commit"
    else:
        atomic_result = atomic.rollback()
        with torch.no_grad():
            for parameter, snapshot in zip(extras, extra_snapshots):
                parameter.copy_(snapshot)
        terminal = "rollback"
    for parameter in extras:
        parameter.requires_grad_(False)
    deployed = evaluate_domains(model, gates, args.device)
    deployed_ratios = ratios(deployed, baseline)
    wal.append(
        terminal,
        transaction_id,
        {"ratios": deployed_ratios, "code_churn": churn, "result": atomic_result},
    )

    checkpoint_path = WORKSPACE / f"wal2/checkpoints/wal-tat-{args.tag}-all-codes.pt"
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
                "mode": "full-block-code-recovery",
                "steps": args.steps,
                "lr": args.lr,
                "teacher_source": args.teacher_source,
            },
        )
        torch.save(payload, checkpoint_path)
        fresh = fresh_verify(checkpoint_path, model_path, suite, args)
        if not fresh["passed"]:
            checkpoint_path.unlink()

    result = {
        "schema": "wal-tat-block-code-recovery-v1",
        "tag": args.tag,
        "source_checkpoint": str(args.source_checkpoint),
        "source_sha256": source_hash,
        "suite": str(args.suite),
        "suite_sha256": suite_hash,
        "steps": args.steps,
        "lr": args.lr,
        "teacher_source": args.teacher_source,
        "kd_weight": args.kd_weight,
        "hidden_kd_weight": args.hidden_kd_weight,
        "baseline": baseline,
        "starting": starting,
        "starting_ratios": starting_ratios,
        "starting_objective": starting_objective,
        "attempted": attempted,
        "attempted_ratios": attempted_ratios,
        "attempted_objective": attempted_objective,
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
