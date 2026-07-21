"""Atomically ternarize linked value and output attention head windows."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from wal_tat import (
    AtomicTernaryTransaction,
    HashChainWAL,
    linked_gqa_output_group_mask,
    structured_channel_candidate_mask,
)

from compensation_window import (
    PROJECT,
    WORKSPACE,
    adaptive_candidate_scores,
    cache_teacher,
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
    sha256_file,
    tensor_digest,
    train_atomic,
)


V_NAME = "model.layers.27.self_attn.v_proj"
O_NAME = "model.layers.27.self_attn.o_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--suite",
        type=Path,
        default=WORKSPACE / "wal2/cache/wal-tat-suite-v1-l128-wc8-cc8-g256.pt",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", default="attention_vo_window_v1")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--head-blocks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gate-ratio", type=float, default=1.02)
    parser.add_argument("--kd-weight", type=float, default=0.2)
    parser.add_argument("--kd-topk", type=int, default=64)
    parser.add_argument("--kd-stride", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--hidden-kd-weight", type=float, default=0.0)
    parser.add_argument("--hidden-module", default="model.layers.27.self_attn")
    parser.add_argument("--teacher-source", choices=("frontier",), default="frontier")
    parser.add_argument("--scale-only-compensation", action="store_true")
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def attention_extras(model):
    values = list(model.model.norm.parameters())
    values += list(model.model.layers[27].input_layernorm.parameters())
    unique, seen = [], set()
    for parameter in values:
        if id(parameter) not in seen:
            unique.append(parameter)
            seen.add(id(parameter))
    return unique


def main() -> None:
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    model_path = (args.model_path or default_model_path()).resolve()
    source_hash = sha256_file(args.source_checkpoint)
    suite_hash = sha256_file(args.suite)
    source_payload = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    suite = torch.load(args.suite, map_location="cpu", weights_only=False)
    calibration, gates = suite["calibration"], suite["gates"]

    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    matrices = install_checkpoint(model, source_payload, args.device)
    value = ensure_candidate_matrix(model, matrices, V_NAME, args.device)
    output = ensure_candidate_matrix(model, matrices, O_NAME, args.device)
    starting = evaluate_domains(model, gates, args.device)
    starting_ratios = ratios(starting, baseline)
    scores = adaptive_candidate_scores(model, value, V_NAME, calibration, args)
    count = max(1, round(scores.numel() * args.fraction))
    value_mask, head_blocks = structured_channel_candidate_mask(
        scores,
        ~value.committed_mask,
        count=count,
        channel_block_size=value.group_size,
        block_count=args.head_blocks,
    )
    output_mask = linked_gqa_output_group_mask(
        ~output.committed_mask,
        head_blocks,
        query_heads_per_kv=model.config.num_attention_heads
        // model.config.num_key_value_heads,
    )
    selection = {
        "value_groups": int(value_mask.sum()),
        "output_groups": int(output_mask.sum()),
        "kv_head_blocks": list(head_blocks),
        "value_mask_sha256": tensor_digest(value_mask),
        "output_mask_sha256": tensor_digest(output_mask),
        "score_mean": float(scores[value_mask].mean()),
        "score_p95": float(torch.quantile(scores[value_mask], 0.95)),
    }
    log(f"frontier={starting_ratios} selection={selection}")
    value_mask, output_mask = value_mask.cpu(), output_mask.cpu()
    del scores, model, matrices, value, output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    matrices = install_checkpoint(model, source_payload, args.device)
    value = ensure_candidate_matrix(model, matrices, V_NAME, args.device)
    output = ensure_candidate_matrix(model, matrices, O_NAME, args.device)
    teacher = cache_teacher(model, calibration, args)
    extras = attention_extras(model)
    extra_snapshots = [parameter.detach().clone() for parameter in extras]
    active = {"candidate": value, "output": output}
    masks = {
        "candidate": value_mask.to(args.device),
        "output": output_mask.to(args.device),
    }
    atomic = AtomicTernaryTransaction(active)
    transaction_id = f"{args.tag}-value-output"
    atomic.begin(masks, transaction_id=transaction_id)

    wal_path = PROJECT / f"results/{args.tag}.wal.jsonl"
    if wal_path.exists():
        wal_path.unlink()
    wal = HashChainWAL(wal_path)
    wal.append(
        "begin",
        transaction_id,
        {**selection, "steps": args.steps, "lr": args.lr},
    )
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    history = train_atomic(
        model, atomic, active, calibration, teacher, extras, args, wal
    )
    attempted = evaluate_domains(model, gates, args.device)
    attempted_ratios = ratios(attempted, baseline)
    passed = all(value <= args.gate_ratio for value in attempted_ratios.values())
    wal.append(
        "commit_intent" if passed else "rollback_intent",
        transaction_id,
        {"metrics": attempted, "ratios": attempted_ratios},
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

    checkpoint_path = WORKSPACE / f"wal2/checkpoints/wal-tat-{args.tag}-value-output.pt"
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
                "mode": "gqa-value-output-window",
                "selection": selection,
                "steps": args.steps,
                "lr": args.lr,
            },
        )
        torch.save(payload, checkpoint_path)
        fresh = fresh_verify(checkpoint_path, model_path, suite, args)
        if not fresh["passed"]:
            checkpoint_path.unlink()

    result = {
        "schema": "wal-tat-attention-vo-window-v1",
        "tag": args.tag,
        "model": str(model_path),
        "source_checkpoint": str(args.source_checkpoint),
        "source_sha256": source_hash,
        "suite": str(args.suite),
        "suite_sha256": suite_hash,
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "baseline": baseline,
        "starting": starting,
        "starting_ratios": starting_ratios,
        "selection": selection,
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


if __name__ == "__main__":
    main()
