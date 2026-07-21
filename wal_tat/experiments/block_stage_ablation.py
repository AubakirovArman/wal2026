"""Measure cumulative loss after hard-ternarizing subgraphs of one new block."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from compensation_window import (
    PROJECT,
    WORKSPACE,
    default_model_path,
    ensure_candidate_matrix,
    evaluate_domains,
    install_checkpoint,
    load_model,
    log,
    ratios,
    seed_everything,
    sha256_file,
)


ARMS = {
    "gate": ("mlp.gate_proj",),
    "up": ("mlp.up_proj",),
    "down": ("mlp.down_proj",),
    "gate_up": ("mlp.gate_proj", "mlp.up_proj"),
    "gate_down": ("mlp.gate_proj", "mlp.down_proj"),
    "up_down": ("mlp.up_proj", "mlp.down_proj"),
    "mlp": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    "vo": ("self_attn.v_proj", "self_attn.o_proj"),
    "qk": ("self_attn.q_proj", "self_attn.k_proj"),
    "attention": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ),
    "full": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ),
}


def hard_add(model, matrices, layer, projections, device):
    for projection in projections:
        name = f"model.layers.{layer}.{projection}"
        matrix = ensure_candidate_matrix(model, matrices, name, device)
        mask = torch.ones_like(matrix.committed_mask)
        matrix.begin(mask, transaction_id=f"ablation-{layer}-{projection}")
        matrix.set_candidate_state(1.0, 0.0)
        matrix.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--target-layer", type=int, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", default="block_stage_ablation_v1")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_path = (args.model_path or default_model_path()).resolve()
    source_payload = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    suite = torch.load(args.suite, map_location="cpu", weights_only=False)
    gates = suite["gates"]
    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    install_checkpoint(model, source_payload, args.device)
    source = evaluate_domains(model, gates, args.device)
    source_ratios = ratios(source, baseline)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {}
    requested_arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if not requested_arms or any(value not in ARMS for value in requested_arms):
        raise ValueError(f"invalid arms: {requested_arms}")
    for arm in requested_arms:
        projections = ARMS[arm]
        seed_everything(args.seed)
        model = load_model(model_path, args.device)
        matrices = install_checkpoint(model, source_payload, args.device)
        hard_add(model, matrices, args.target_layer, projections, args.device)
        metrics = evaluate_domains(model, gates, args.device)
        results[arm] = {
            "projections": projections,
            "metrics": metrics,
            "ratios": ratios(metrics, baseline),
            "added_parameters": sum(
                matrices[f"model.layers.{args.target_layer}.{name}"].master_weight.numel()
                for name in projections
            ),
        }
        log(f"{arm}: {results[arm]['ratios']}")
        del model, matrices
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "schema": "wal-tat-block-stage-ablation-v1",
        "model": str(model_path),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "suite": str(args.suite.resolve()),
        "suite_sha256": sha256_file(args.suite),
        "target_layer": args.target_layer,
        "baseline": baseline,
        "source_frontier": source,
        "source_frontier_ratios": source_ratios,
        "arms": results,
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
