from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from m27_wal_lha_proto import (
    MODEL_DIR,
    TARGETS,
    _build_activation_bank,
    _build_current_cache,
    _eval_ids,
    _run_preencoded_eval,
)
from m27_wal_sbc_offline import (
    build_budgeted_layer_from_surface,
    build_full_score_surface,
    load_sbc_artifacts,
)
from m27_wal_sbc_proto import DEFAULTS, _collect_layer_inputs_budgeted
from dwl2_dynamic_route.src.encoding_io import save_grouped_encoding_map


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-source", default=DEFAULTS["text_source"])
    parser.add_argument("--current-cache", default=DEFAULTS["current_cache"])
    parser.add_argument("--sbc-artifact", default=DEFAULTS["sbc_artifact"])
    parser.add_argument("--profile", default=str(Path(DEFAULTS["out"]).with_name("m27_wal_sbc_budget_profile_summary.json")))
    parser.add_argument("--out", default=str(Path(DEFAULTS["out"]).with_name("m27_wal_sbc_tune_summary.json")))
    parser.add_argument("--cache-prefix", default=str(Path(DEFAULTS["out"]).with_name("m27_wal_sbc_tune_cache")))
    parser.add_argument("--surface-prefix", default=str(Path(DEFAULTS["out"]).with_name("m27_wal_sbc_tune_surface")))
    parser.add_argument("--screen-windows", type=int, default=4)
    parser.add_argument("--final-windows", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--lha-calibration-windows", type=int, default=1)
    parser.add_argument("--capture-max-len", type=int, default=DEFAULTS["capture_max_len"])
    parser.add_argument("--activation-rows", type=int, default=DEFAULTS["activation_rows"])
    parser.add_argument("--train-samples", type=int, default=DEFAULTS["train_samples"])
    parser.add_argument("--assign-chunk-size", type=int, default=DEFAULTS["assign_chunk_size"])
    parser.add_argument("--matmul-strategy", default=DEFAULTS["matmul_strategy"])
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def _candidate_id(atom_label: str, residual_label: str, residual_cost: float) -> str:
    return f"a_{atom_label}_r_{residual_label}_c_{str(residual_cost).replace('.', 'p')}"


def _surface_path(prefix: str, name: str) -> Path:
    return Path(f"{prefix}_{name.replace('.', '_')}.pt")


def main() -> None:
    args = _parse_args()
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    candidates = [
        (atom_label, residual_label, residual_cost)
        for atom_label in ("p70", "p80", "p90")
        for residual_label in ("p85", "p92")
        for residual_cost in (0.5, 0.75, 1.0)
    ]
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)
    current_cache_path = Path(args.current_cache)
    current_cache = _build_current_cache(args, current_cache_path, TARGETS)
    layer_inputs = _collect_layer_inputs_budgeted(ids, args.lha_calibration_windows, TARGETS, args.activation_rows, args.capture_max_len)
    activation_banks = {name: _build_activation_bank(layer_inputs[name], DEFAULTS["block_size"]) for name in TARGETS}
    artifacts = load_sbc_artifacts(args.sbc_artifact)
    surfaces = {}
    for name in TARGETS:
        surface_path = _surface_path(args.surface_prefix, name)
        if surface_path.exists():
            surfaces[name] = torch.load(surface_path, map_location="cpu", weights_only=False)
        else:
            surfaces[name] = build_full_score_surface(current_cache[name], activation_banks[name], artifacts[name], args.train_samples, args.assign_chunk_size)
            torch.save(surfaces[name], surface_path)
    screen_legacy = _run_preencoded_eval(ids, current_cache_path, args.matmul_strategy, args.screen_windows)
    final_legacy = _run_preencoded_eval(ids, current_cache_path, args.matmul_strategy, args.final_windows)
    results = []
    for atom_label, residual_label, residual_cost in candidates:
        encodings = {}
        wal_rows = []
        for name in TARGETS:
            layer_profile = profile["profiles"][name]
            budgeted_enc, stats = build_budgeted_layer_from_surface(
                name,
                current_cache[name],
                activation_banks[name],
                artifacts[name],
                surfaces[name],
                float(layer_profile["grid_budgets"]["atom"][atom_label]),
                float(layer_profile["grid_budgets"]["residual"][residual_label]),
                float(residual_cost),
            )
            encodings[name] = budgeted_enc
            wal_rows.append(stats)
        candidate_id = _candidate_id(atom_label, residual_label, residual_cost)
        cache_path = Path(f"{args.cache_prefix}_{candidate_id}.pt")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_grouped_encoding_map(cache_path, encodings)
        screen_eval = _run_preencoded_eval(ids, cache_path, args.matmul_strategy, args.screen_windows)
        mean_program = sum(item["avg_program_length"] for item in wal_rows) / max(len(wal_rows), 1)
        mean_nonliteral = sum(item["avg_low_level_calls"] + item["avg_residual_calls"] for item in wal_rows) / max(len(wal_rows), 1)
        results.append(
            {
                "candidate_id": candidate_id,
                "atom_output_budget_label": atom_label,
                "residual_output_budget_label": residual_label,
                "residual_program_cost": float(residual_cost),
                "cache_path": str(cache_path),
                "wal_sbc": wal_rows,
                "screen": screen_eval,
                "mean_program_length": float(mean_program),
                "mean_nonliteral_calls": float(mean_nonliteral),
                "screen_ppl_delta_vs_legacy": float(screen_eval["metrics"]["perplexity"] - screen_legacy["metrics"]["perplexity"]),
            }
        )
    results.sort(key=lambda item: (item["mean_program_length"], -item["mean_nonliteral_calls"], item["screen_ppl_delta_vs_legacy"]))
    finalists = results[: int(args.top_k)]
    for item in finalists:
        item["final"] = _run_preencoded_eval(ids, Path(item["cache_path"]), args.matmul_strategy, args.final_windows)
        item["final_ppl_delta_vs_legacy"] = float(item["final"]["metrics"]["perplexity"] - final_legacy["metrics"]["perplexity"])
    summary = {
        "targets": list(TARGETS),
        "profile": str(args.profile),
        "screen_windows": int(args.screen_windows),
        "final_windows": int(args.final_windows),
        "screen_legacy": screen_legacy,
        "final_legacy": final_legacy,
        "screen_results": results,
        "finalists": finalists,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()