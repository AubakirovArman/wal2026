from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from m27_wal_lha_proto import MODEL_DIR, TARGETS, _build_activation_bank, _build_current_cache, _eval_ids
from m27_wal_sbc_offline import iter_scored_chunks, load_sbc_artifacts
from m27_wal_sbc_proto import DEFAULTS, _collect_layer_inputs_budgeted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-source", default=DEFAULTS["text_source"])
    parser.add_argument("--current-cache", default=DEFAULTS["current_cache"])
    parser.add_argument("--sbc-artifact", default=DEFAULTS["sbc_artifact"])
    parser.add_argument("--out", default=str(Path(DEFAULTS["out"]).with_name("m27_wal_sbc_budget_profile_summary.json")))
    parser.add_argument("--lha-calibration-windows", type=int, default=1)
    parser.add_argument("--capture-max-len", type=int, default=DEFAULTS["capture_max_len"])
    parser.add_argument("--activation-rows", type=int, default=DEFAULTS["activation_rows"])
    parser.add_argument("--train-samples", type=int, default=DEFAULTS["train_samples"])
    parser.add_argument("--assign-chunk-size", type=int, default=DEFAULTS["assign_chunk_size"])
    parser.add_argument("--residual-costs", default="0.5,0.75,1.0,2.0")
    parser.add_argument("--almost-budget-multiplier", type=float, default=1.25)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def _quantiles(values: torch.Tensor, mapping: dict[str, float]) -> dict[str, float]:
    work = values.to(torch.float32)
    return {label: float(torch.quantile(work, frac).item()) for label, frac in mapping.items()}


def _cost_surface(atom_out: torch.Tensor, residual_out: torch.Tensor, num_rows: int, num_slots: int, phrase_len: int, residual_costs: list[float]) -> tuple[dict[str, float], list[dict[str, float]]]:
    atom_budgets = _quantiles(atom_out, {"p70": 0.70, "p80": 0.80, "p90": 0.90})
    residual_budgets = _quantiles(residual_out, {"p85": 0.85, "p92": 0.92})
    rows = []
    for atom_label, atom_budget in atom_budgets.items():
        atom_accept = atom_out <= atom_budget
        for residual_label, residual_budget in residual_budgets.items():
            residual_ok = residual_out <= residual_budget
            for residual_cost in residual_costs:
                residual_accept = (~atom_accept) & residual_ok
                literal_mask = ~(atom_accept | residual_accept)
                slot_costs = torch.where(
                    atom_accept,
                    torch.ones_like(atom_out),
                    torch.where(residual_accept, torch.full_like(atom_out, residual_cost), torch.full_like(atom_out, float(phrase_len))),
                )
                rows.append(
                    {
                        "atom_output_budget_label": atom_label,
                        "atom_output_budget": atom_budget,
                        "residual_output_budget_label": residual_label,
                        "residual_output_budget": residual_budget,
                        "residual_program_cost": float(residual_cost),
                        "avg_program_length": float((1.0 + slot_costs.view(num_rows, num_slots).sum(dim=1)).mean().item()),
                        "avg_low_level_calls": float(atom_accept.view(num_rows, num_slots).to(torch.float32).sum(dim=1).mean().item()),
                        "avg_residual_calls": float(residual_accept.view(num_rows, num_slots).to(torch.float32).sum(dim=1).mean().item()),
                        "avg_literal_slots": float(literal_mask.view(num_rows, num_slots).to(torch.float32).sum(dim=1).mean().item()),
                    }
                )
    return {"atom": atom_budgets, "residual": residual_budgets}, rows


def main() -> None:
    args = _parse_args()
    residual_costs = [float(piece) for piece in args.residual_costs.split(",") if piece.strip()]
    current_cache = _build_current_cache(args, Path(args.current_cache), TARGETS)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)
    layer_inputs = _collect_layer_inputs_budgeted(ids, args.lha_calibration_windows, TARGETS, args.activation_rows, args.capture_max_len)
    activation_banks = {name: _build_activation_bank(layer_inputs[name], DEFAULTS["block_size"]) for name in TARGETS}
    artifacts = load_sbc_artifacts(args.sbc_artifact)
    quantile_labels = {"p50": 0.50, "p70": 0.70, "p80": 0.80, "p85": 0.85, "p90": 0.90, "p92": 0.92, "p95": 0.95}
    summary = {
        "targets": list(TARGETS),
        "text_source": args.text_source,
        "lha_calibration_windows": int(args.lha_calibration_windows),
        "train_samples": int(args.train_samples),
        "assign_chunk_size": int(args.assign_chunk_size),
        "residual_costs": residual_costs,
        "profiles": {},
    }
    for name in TARGETS:
        meta, chunks = iter_scored_chunks(name, current_cache[name], activation_banks[name], artifacts[name], args.train_samples, args.assign_chunk_size, sample_only=True)
        atom_out = torch.cat([slot["atom_output_rel"] for chunk in chunks for slot in chunk["slot_payloads"]], dim=0)
        residual_out = torch.cat([slot["residual_output_rel"] for chunk in chunks for slot in chunk["slot_payloads"]], dim=0)
        best_nonliteral = torch.minimum(atom_out, residual_out)
        current_atom_budget = float(artifacts[name]["budgets"]["atom_output_budget"])
        current_residual_budget = float(artifacts[name]["budgets"]["residual_output_budget"])
        budget_map, cost_surface = _cost_surface(atom_out, residual_out, int(meta["sequences"].shape[0]), int(meta["num_slots"]), int(meta["phrase_len"]), residual_costs)
        summary["profiles"][name] = {
            "sample_rows": int(meta["sequences"].shape[0]),
            "sample_slots": int(atom_out.numel()),
            "current_atom_output_budget": current_atom_budget,
            "current_residual_output_budget": current_residual_budget,
            "atom_output_rel_mse_quantiles": _quantiles(atom_out, quantile_labels),
            "residual_output_rel_mse_quantiles": _quantiles(residual_out, quantile_labels),
            "best_nonliteral_output_rel_mse_quantiles": _quantiles(best_nonliteral, quantile_labels),
            "share_atom_le_current_budget": float((atom_out <= current_atom_budget).to(torch.float32).mean().item()),
            "share_atom_almost_current_budget": float(((atom_out > current_atom_budget) & (atom_out <= current_atom_budget * args.almost_budget_multiplier)).to(torch.float32).mean().item()),
            "share_best_nonliteral_le_current_residual_budget": float((best_nonliteral <= current_residual_budget).to(torch.float32).mean().item()),
            "grid_budgets": budget_map,
            "cost_surface": cost_surface,
        }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()