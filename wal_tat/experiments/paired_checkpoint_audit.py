"""Fresh BF16-versus-frontier audit with paired moving-block bootstrap CIs."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from compensation_window import (
    PROJECT,
    default_model_path,
    install_checkpoint,
    load_model,
    seed_everything,
    sha256_file,
)
from wal_tat import paired_block_bootstrap_nll


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gate-ratio", type=float, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=4096)
    parser.add_argument("--bootstrap-block-size", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument(
        "--suite-role",
        default="recurring validation; no gradient updates",
        help="Declared methodological role of this suite in the current study.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_window_sums(model, chunks, device: str, batch_size: int) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    nll_sums: list[float] = []
    token_counts: list[int] = []
    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        lengths = {int(chunk.numel()) for chunk in batch_chunks}
        if len(lengths) != 1:
            raise ValueError("batched audit windows must have equal lengths")
        batch = torch.stack(batch_chunks).to(device)
        logits = model(input_ids=batch[:, :-1], use_cache=False).logits
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            batch[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(batch.shape[0], -1)
        nll_sums.extend(float(value) for value in losses.sum(dim=1).cpu())
        token_counts.extend([batch.shape[1] - 1] * batch.shape[0])
    total_nll = sum(nll_sums)
    total_tokens = sum(token_counts)
    mean_nll = total_nll / total_tokens
    return {
        "nll_sums": nll_sums,
        "token_counts": token_counts,
        "aggregate": {
            "nll": mean_nll,
            "ppl": math.exp(min(mean_nll, 50.0)),
            "tokens": total_tokens,
        },
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    gates = suite["gates"]
    seed_everything(args.seed)

    model = load_model(model_path, args.device)
    baseline = {
        domain: evaluate_window_sums(model, chunks, args.device, args.batch_size)
        for domain, chunks in gates.items()
    }
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    matrices = install_checkpoint(model, payload, args.device)
    candidate = {
        domain: evaluate_window_sums(model, chunks, args.device, args.batch_size)
        for domain, chunks in gates.items()
    }

    domains = {}
    for domain in gates:
        domains[domain] = paired_block_bootstrap_nll(
            baseline[domain]["nll_sums"],
            candidate[domain]["nll_sums"],
            baseline[domain]["token_counts"],
            gate_ratio=args.gate_ratio,
            samples=args.bootstrap_samples,
            block_size=args.bootstrap_block_size,
            confidence=args.confidence,
            seed=args.seed,
        )
    result = {
        "schema": "wal-tat-paired-checkpoint-audit-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": args.suite_role,
        "model": str(model_path),
        "gate_ratio": args.gate_ratio,
        "batch_size": args.batch_size,
        "domains": domains,
        "point_passed": all(value["point_passed"] for value in domains.values()),
        "confidence_passed": all(
            value["confidence_passed"] for value in domains.values()
        ),
        "worst_observed_ratio": max(value["observed_ratio"] for value in domains.values()),
        "worst_upper_ratio": max(value["ratio_ci"]["upper"] for value in domains.values()),
        "coverage": {
            name: float(matrix.committed_mask.float().mean().item())
            for name, matrix in matrices.items()
        },
        "raw_window_losses": {
            domain: {
                "baseline_nll_sums": baseline[domain]["nll_sums"],
                "candidate_nll_sums": candidate[domain]["nll_sums"],
                "token_counts": baseline[domain]["token_counts"],
            }
            for domain in domains
        },
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "raw_window_losses"}, indent=2))

    del model, matrices
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
