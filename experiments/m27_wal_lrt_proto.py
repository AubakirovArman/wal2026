"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""M27 WAL-LRT Step 6a: learned route-template prototype for l54 gate/up.

This probe moves from mined exact macros to learned full-program templates.

The first WAL-LRT pass stays conservative in two ways:
  1. the main syntax metric uses exact decode via `TEMPLATE call + corrections`
  2. an explicit `template_only` approximation is evaluated separately to see
     whether the learned templates themselves preserve quality

Templates are learned over full block programs (the total-stage id sequence)
with a small categorical k-modes loop that minimizes Hamming distance.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.block_vq import BlockRVQEncoding, GroupedBlockRVQEncoding, encode_grouped_block_residual_vq
from dwl2_dynamic_route.src.encoding_io import load_grouped_encoding_map, save_grouped_encoding_map
from dwl2_dynamic_route.src.runtime import replace_with_preencoded_packed_block_rvq


MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
TEXT_PATH = ROOT / "bk/wikitext2_test.txt"
MAX_LEN = 2048
STRIDE = 512
TARGETS = (
    "model.layers.54.mlp.gate_proj",
    "model.layers.54.mlp.up_proj",
)


def _eval_ids(tok: AutoTokenizer, source: str) -> torch.Tensor:
    if source == "raw":
        from datasets import load_dataset

        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    else:
        text = TEXT_PATH.read_text()
    return tok(text, return_tensors="pt").input_ids.cpu()


def _load_model() -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )


def _reset_peaks() -> None:
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(idx)


def _vram() -> list[dict[str, float]]:
    return [
        {
            "device": idx,
            "allocated_mb": round(torch.cuda.memory_allocated(idx) / 2**20, 1),
            "reserved_mb": round(torch.cuda.memory_reserved(idx) / 2**20, 1),
            "peak_mb": round(torch.cuda.max_memory_allocated(idx) / 2**20, 1),
        }
        for idx in range(torch.cuda.device_count())
    ]


def _peak(snap: list[dict[str, float]]) -> float:
    return max((float(item["peak_mb"]) for item in snap), default=0.0)


def _evaluate(model: AutoModelForCausalLM, ids: torch.Tensor, n_windows: int) -> dict[str, float]:
    model.eval()
    device = model.get_input_embeddings().weight.device
    nlls: list[float] = []
    total_tokens = 0
    prev_end = 0
    total_len = ids.size(1)
    n_windows = min(n_windows, max(1, (total_len - MAX_LEN) // STRIDE + 1))
    t0 = time.time()
    with torch.no_grad():
        for idx in range(n_windows):
            begin = idx * STRIDE
            end = min(begin + MAX_LEN, total_len)
            target_len = end - prev_end if idx > 0 else end - begin
            chunk = ids[:, begin:end].to(device)
            target = chunk.clone()
            if idx > 0:
                target[:, :-target_len] = -100
            loss = model(chunk, labels=target).loss.item()
            nlls.append(loss * target_len)
            total_tokens += target_len
            prev_end = end
            if (idx + 1) % 8 == 0 or idx + 1 == n_windows:
                dt = time.time() - t0
                ppl = math.exp(sum(nlls) / total_tokens)
                print(f"  {idx + 1}/{n_windows}  ppl={ppl:.4f}  tok/s={total_tokens / max(dt, 1e-9):.1f}", flush=True)
    dt = time.time() - t0
    return {
        "perplexity": math.exp(sum(nlls) / total_tokens),
        "total_tokens": total_tokens,
        "num_windows": n_windows,
        "elapsed_s": dt,
        "tok_s": total_tokens / max(dt, 1e-9),
    }


def _rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a.float() - b.float()).square().mean()
    den = a.float().square().mean().clamp_min(1e-12)
    return float((num / den).item())


def _load_cached_subset(path: Path, targets: tuple[str, ...]) -> dict[str, GroupedBlockRVQEncoding]:
    if not path.exists():
        return {}
    loaded = load_grouped_encoding_map(path)
    return {name: enc for name, enc in loaded.items() if name in targets}


def _build_current_cache(args: argparse.Namespace, cache_path: Path, targets: tuple[str, ...]) -> dict[str, GroupedBlockRVQEncoding]:
    encodings = _load_cached_subset(cache_path, targets)
    if not encodings and args.bootstrap_cache:
        encodings.update(_load_cached_subset(Path(args.bootstrap_cache), targets))
    missing = [name for name in targets if name not in encodings]
    if not missing and cache_path.exists() and not args.rebuild_cache:
        print(f"[cache] load {cache_path}", flush=True)
        return encodings
    if not missing and not args.rebuild_cache:
        print(f"[cache] reuse bootstrap subset -> {cache_path}", flush=True)
        save_grouped_encoding_map(cache_path, encodings)
        return encodings
    print(f"[cache] build/extend {cache_path}", flush=True)
    model = _load_model()
    module_map = dict(model.named_modules())
    for name in missing:
        module = module_map[name]
        if not isinstance(module, nn.Linear):
            raise TypeError(f"target {name} is not nn.Linear")
        print(f"  encode {name}", flush=True)
        encodings[name] = encode_grouped_block_residual_vq(
            module.weight.detach(),
            group_rows=args.group_rows,
            block_size=args.block_size,
            codebook_size=args.codebook_size,
            num_stages=args.num_stages,
            product_splits=args.product_splits,
            calibrate_stage_scales=args.calibrate_stage_scales,
            sample_limit=args.sample_limit,
            kmeans_iters=args.kmeans_iters,
            batch_size=args.batch_size,
        )
    save_grouped_encoding_map(cache_path, encodings)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return encodings


def _flatten_group_sequences(group: BlockRVQEncoding) -> torch.Tensor:
    return torch.stack([ids.reshape(-1).to(torch.uint8).cpu() for ids in group.stage_ids], dim=1).contiguous()


def _sample_training_sequences(enc: GroupedBlockRVQEncoding, max_rows: int, seed: int) -> torch.Tensor:
    total_blocks = sum(int(group.stage_ids[0].numel()) for group in enc.groups)
    if total_blocks <= max_rows:
        return torch.cat([_flatten_group_sequences(group) for group in enc.groups], dim=0)
    samples = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for group in enc.groups:
        group_blocks = int(group.stage_ids[0].numel())
        budget = max(1, int(round(max_rows * group_blocks / total_blocks)))
        seq_mat = _flatten_group_sequences(group)
        budget = min(budget, int(seq_mat.shape[0]))
        pick = torch.randperm(int(seq_mat.shape[0]), generator=generator)[:budget]
        samples.append(seq_mat[pick])
    return torch.cat(samples, dim=0)[:max_rows].contiguous()


def _nearest_templates(sequence_matrix: torch.Tensor, templates: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    total_rows = int(sequence_matrix.shape[0])
    assignments = torch.empty(total_rows, dtype=torch.int64)
    distances = torch.empty(total_rows, dtype=torch.int16)
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = sequence_matrix[start:end]
        dist = (chunk[:, None, :] != templates[None, :, :]).sum(dim=2)
        best_dist, best_idx = dist.min(dim=1)
        assignments[start:end] = best_idx.to(torch.int64)
        distances[start:end] = best_dist.to(torch.int16)
    return assignments, distances


def _learn_templates(
    sample_sequences: torch.Tensor,
    num_templates: int,
    base: int,
    iters: int,
    assign_chunk_size: int,
) -> dict[str, object]:
    train_rows = int(sample_sequences.shape[0])
    if train_rows < num_templates:
        raise ValueError("training sample must be at least as large as num_templates")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    perm = torch.randperm(train_rows, generator=generator)
    templates = sample_sequences[perm[:num_templates]].clone()
    history = []
    for iter_idx in range(iters):
        assignments, distances = _nearest_templates(sample_sequences, templates, assign_chunk_size)
        counts = torch.bincount(assignments, minlength=num_templates)
        updated = templates.clone()
        for pos in range(sample_sequences.shape[1]):
            flat = assignments * base + sample_sequences[:, pos].to(torch.int64)
            hist = torch.bincount(flat, minlength=num_templates * base).view(num_templates, base)
            updated[:, pos] = hist.argmax(dim=1).to(torch.uint8)
        empty = (counts == 0).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            reseed = torch.randint(0, train_rows, (empty.numel(),), generator=generator)
            updated[empty] = sample_sequences[reseed]
        templates = updated.contiguous()
        row = {
            "iter": int(iter_idx + 1),
            "mean_hamming": float(distances.to(torch.float32).mean().item()),
            "p50_hamming": float(torch.quantile(distances.to(torch.float32), 0.50).item()),
            "p90_hamming": float(torch.quantile(distances.to(torch.float32), 0.90).item()),
        }
        history.append(row)
        print(
            f"    iter {iter_idx + 1}/{iters}: mean_hamming={row['mean_hamming']:.3f} p90={row['p90_hamming']:.3f}",
            flush=True,
        )
    final_assignments, final_distances = _nearest_templates(sample_sequences, templates, assign_chunk_size)
    final_counts = torch.bincount(final_assignments, minlength=num_templates)
    return {
        "templates": templates,
        "train_counts": final_counts,
        "train_mean_hamming": float(final_distances.to(torch.float32).mean().item()),
        "history": history,
    }


def _clone_group_with_template_assignments(group: BlockRVQEncoding, templates: torch.Tensor, assignments: torch.Tensor) -> BlockRVQEncoding:
    shape = group.stage_ids[0].shape
    stage_ids = []
    for stage_idx, ids in enumerate(group.stage_ids):
        templ_ids = templates[:, stage_idx].to(torch.int64)[assignments]
        stage_ids.append(templ_ids.reshape(shape).to(dtype=ids.dtype))
    return BlockRVQEncoding(
        stage_ids=tuple(stage_ids),
        codebooks=group.codebooks,
        stage_value_dims=group.stage_value_dims,
        stages_per_split=group.stages_per_split,
        stage_scales=group.stage_scales,
        residual_correction=group.residual_correction,
        residual_signs=group.residual_signs,
        residual_scale=group.residual_scale,
        row_scale=group.row_scale,
        block_scale=group.block_scale,
        transform_kind=group.transform_kind,
        transform_matrix=group.transform_matrix,
        transform_bias=group.transform_bias,
        product_splits=group.product_splits,
        original_shape=group.original_shape,
        padded_cols=group.padded_cols,
        block_size=group.block_size,
        sample_rel_mse=group.sample_rel_mse,
    )


def _build_lrt_layer(
    name: str,
    enc: GroupedBlockRVQEncoding,
    args: argparse.Namespace,
) -> tuple[GroupedBlockRVQEncoding, dict[str, object], dict[str, object]]:
    raw_len = len(enc.groups[0].stage_ids) if enc.groups else 0
    base = max(int(codebook.shape[0]) for group in enc.groups for codebook in group.codebooks)
    print(f"[lrt/train] {name}: sample_rows={args.train_samples} templates={args.num_templates}", flush=True)
    train_sequences = _sample_training_sequences(enc, args.train_samples, seed=0)
    learned = _learn_templates(
        train_sequences,
        int(args.num_templates),
        base,
        int(args.template_iters),
        int(args.assign_chunk_size),
    )
    templates = learned["templates"]

    total_program_units = 0
    total_template_calls = 0
    total_literal_corrections = 0
    total_template_tokens = 0
    total_template_only_hamming = 0
    total_blocks = 0
    used_templates: Counter[int] = Counter()
    approx_groups = []
    for group in enc.groups:
        seq_mat = _flatten_group_sequences(group)
        assignments, distances = _nearest_templates(seq_mat, templates, int(args.assign_chunk_size))
        use_template = (distances.to(torch.int32) + 1) < raw_len
        program_units = torch.where(use_template, distances.to(torch.int32) + 1, torch.full_like(distances.to(torch.int32), raw_len))
        covered_tokens = torch.where(use_template, raw_len - distances.to(torch.int32), torch.zeros_like(distances.to(torch.int32)))
        literal_corrections = torch.where(use_template, distances.to(torch.int32), torch.full_like(distances.to(torch.int32), raw_len))
        total_program_units += int(program_units.sum().item())
        total_template_calls += int(use_template.sum().item())
        total_literal_corrections += int(literal_corrections.sum().item())
        total_template_tokens += int(covered_tokens.sum().item())
        total_template_only_hamming += int(distances.to(torch.int32).sum().item())
        total_blocks += int(seq_mat.shape[0])
        if bool(use_template.any()):
            for template_id in assignments[use_template].tolist():
                used_templates[int(template_id)] += 1
        approx_groups.append(_clone_group_with_template_assignments(group, templates, assignments))

    template_only = GroupedBlockRVQEncoding(
        groups=tuple(approx_groups),
        row_slices=enc.row_slices,
        original_shape=enc.original_shape,
    )
    top_templates = []
    train_counts = learned["train_counts"]
    for template_id, used_calls in used_templates.most_common(10):
        top_templates.append(
            {
                "template_id": int(template_id),
                "used_calls": int(used_calls),
                "train_count": int(train_counts[template_id].item()),
                "pattern": templates[template_id].to(torch.int64).tolist(),
            }
        )
    exact_stats = {
        "name": name,
        "raw_program_length": int(raw_len),
        "avg_program_length": float(total_program_units / max(total_blocks, 1)),
        "avg_template_calls": float(total_template_calls / max(total_blocks, 1)),
        "avg_literal_corrections": float(total_literal_corrections / max(total_blocks, 1)),
        "template_token_coverage": float(total_template_tokens / max(total_blocks * raw_len, 1)),
        "program_compression_ratio": float(total_program_units / max(total_blocks * raw_len, 1)),
        "selected_template_count": int(args.num_templates),
        "used_template_count": int(sum(1 for count in used_templates.values() if count > 0)),
        "template_only_avg_hamming": float(total_template_only_hamming / max(total_blocks, 1)),
        "template_only_token_match": float(1.0 - total_template_only_hamming / max(total_blocks * raw_len, 1)),
        "train_mean_hamming": float(learned["train_mean_hamming"]),
        "train_history": learned["history"],
        "top_templates": top_templates,
    }
    artifact = {
        "templates": templates,
        "train_counts": train_counts,
        "top_templates": top_templates,
        "train_history": learned["history"],
    }
    return template_only, exact_stats, artifact


def _compare_encodings(original: GroupedBlockRVQEncoding, rebuilt: GroupedBlockRVQEncoding) -> dict[str, object]:
    if original is rebuilt:
        return {
            "exact_stage_ids": True,
            "max_stage_id_diff": 0,
            "recon_rel_mse": 0.0,
        }
    seq_equal = True
    max_id_diff = 0
    for g_a, g_b in zip(original.groups, rebuilt.groups):
        for ids_a, ids_b in zip(g_a.stage_ids, g_b.stage_ids):
            ids_a_cpu = ids_a.to(torch.int64).cpu()
            ids_b_cpu = ids_b.to(torch.int64).cpu()
            diff = (ids_a_cpu - ids_b_cpu).abs().max().item()
            max_id_diff = max(max_id_diff, int(diff))
            if not torch.equal(ids_a_cpu, ids_b_cpu):
                seq_equal = False
    recon_a = original.reconstruct(out_dtype=torch.bfloat16).cpu()
    recon_b = rebuilt.reconstruct(out_dtype=torch.bfloat16).cpu()
    return {
        "exact_stage_ids": bool(seq_equal),
        "max_stage_id_diff": int(max_id_diff),
        "recon_rel_mse": _rel_mse(recon_a, recon_b),
    }


def _run_dense_eval(ids: torch.Tensor, num_windows: int) -> dict[str, object]:
    model = _load_model()
    _reset_peaks()
    metrics = _evaluate(model, ids, num_windows)
    eval_vram = _vram()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"metrics": metrics, "eval_vram": eval_vram, "eval_peak_mb": _peak(eval_vram)}


def _run_preencoded_eval(ids: torch.Tensor, cache_path: Path, strategy: str, num_windows: int) -> dict[str, object]:
    model = _load_model()
    encodings = load_grouped_encoding_map(cache_path)
    _reset_peaks()
    layer_stats = replace_with_preencoded_packed_block_rvq(model, encodings, matmul_strategy=strategy)
    replace_vram = _vram()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _reset_peaks()
    metrics = _evaluate(model, ids, num_windows)
    eval_vram = _vram()
    avg_rel_mse = sum(item["rel_mse"] for item in layer_stats) / max(len(layer_stats), 1)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "replace_vram": replace_vram,
        "eval_vram": eval_vram,
        "replace_peak_mb": _peak(replace_vram),
        "eval_peak_mb": _peak(eval_vram),
        "avg_layer_rel_mse": float(avg_rel_mse),
        "layer_stats": layer_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-source", choices=("raw", "local"), default="local")
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--group-rows", type=int, default=28672)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--product-splits", type=int, default=4)
    parser.add_argument("--calibrate-stage-scales", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--num-templates", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=131072)
    parser.add_argument("--template-iters", type=int, default=6)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--matmul-strategy", default="full_weight_fast")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--bootstrap-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ss_current_l54_gate_up.pt"))
    parser.add_argument("--current-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_lrt_current_l54_gate_up.pt"))
    parser.add_argument("--template-only-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_lrt_template_only_l54_gate_up.pt"))
    parser.add_argument("--template-artifact", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_lrt_templates_l54_gate_up.pt"))
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_lrt_proto_summary.json"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)

    current_cache = Path(args.current_cache)
    current_enc = _build_current_cache(args, current_cache, TARGETS)

    template_only_encodings: dict[str, GroupedBlockRVQEncoding] = {}
    wal_lrt_rows = []
    template_only_compare = []
    artifacts = {}
    for name in TARGETS:
        template_only, exact_stats, artifact = _build_lrt_layer(name, current_enc[name], args)
        template_only_encodings[name] = template_only
        wal_lrt_rows.append(exact_stats)
        template_only_compare.append({"name": name, **_compare_encodings(current_enc[name], template_only)})
        artifacts[name] = artifact
        print(
            f"[wal-lrt] {name}: avg_program={exact_stats['avg_program_length']:.3f}/{exact_stats['raw_program_length']} "
            f"coverage={exact_stats['template_token_coverage']:.3f} used_templates={exact_stats['used_template_count']} "
            f"template_only_match={exact_stats['template_only_token_match']:.3f}",
            flush=True,
        )

    template_artifact = Path(args.template_artifact)
    template_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "targets": list(TARGETS),
            "num_templates": int(args.num_templates),
            "artifacts": artifacts,
        },
        template_artifact,
    )
    template_only_cache = Path(args.template_only_cache)
    save_grouped_encoding_map(template_only_cache, template_only_encodings)

    print("\n[dense] evaluate untouched model", flush=True)
    dense = _run_dense_eval(ids, args.num_windows)
    print(f"\n[eval] current via {args.matmul_strategy}", flush=True)
    current_eval = _run_preencoded_eval(ids, current_cache, args.matmul_strategy, args.num_windows)
    print(f"\n[eval] template_only via {args.matmul_strategy}", flush=True)
    template_only_eval = _run_preencoded_eval(ids, template_only_cache, args.matmul_strategy, args.num_windows)
    exact_eval = dict(current_eval)
    exact_eval["by_construction_identical"] = True

    result = {
        "targets": list(TARGETS),
        "text_source": args.text_source,
        "num_windows": int(args.num_windows),
        "matmul_strategy": args.matmul_strategy,
        "num_templates": int(args.num_templates),
        "train_samples": int(args.train_samples),
        "template_iters": int(args.template_iters),
        "dense": dense,
        "current_cache": str(current_cache),
        "template_only_cache": str(template_only_cache),
        "template_artifact": str(template_artifact),
        "wal_lrt": wal_lrt_rows,
        "template_only_compare": template_only_compare,
        "current_eval": current_eval,
        "exact_eval": exact_eval,
        "template_only_eval": template_only_eval,
        "delta_exact": {
            "ppl_delta": 0.0,
            "tok_s_delta": 0.0,
            "peak_mb_delta": 0.0,
        },
        "delta_template_only": {
            "ppl_delta": float(template_only_eval["metrics"]["perplexity"] - current_eval["metrics"]["perplexity"]),
            "tok_s_delta": float(template_only_eval["metrics"]["tok_s"] - current_eval["metrics"]["tok_s"]),
            "peak_mb_delta": float(template_only_eval["eval_peak_mb"] - current_eval["eval_peak_mb"]),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    print(
        f"  dense:             ppl={dense['metrics']['perplexity']:.4f} tok/s={dense['metrics']['tok_s']:.2f} peak_mb={dense['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  current:           ppl={current_eval['metrics']['perplexity']:.4f} tok/s={current_eval['metrics']['tok_s']:.2f} peak_mb={current_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  wal_lrt_exact:     ppl={exact_eval['metrics']['perplexity']:.4f} tok/s={exact_eval['metrics']['tok_s']:.2f} peak_mb={exact_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  wal_lrt_template:  ppl={template_only_eval['metrics']['perplexity']:.4f} tok/s={template_only_eval['metrics']['tok_s']:.2f} peak_mb={template_only_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    for row, compare in zip(wal_lrt_rows, template_only_compare):
        print(
            f"  {row['name']}: avg_program={row['avg_program_length']:.3f}/{row['raw_program_length']} "
            f"coverage={row['template_token_coverage']:.3f} used_templates={row['used_template_count']} "
            f"template_only_rel_mse={compare['recon_rel_mse']:.6f}",
            flush=True,
        )
    print(
        f"  delta(template_only-current): ppl={result['delta_template_only']['ppl_delta']:+.6f} "
        f"tok/s={result['delta_template_only']['tok_s_delta']:+.2f} peak_mb={result['delta_template_only']['peak_mb_delta']:+.1f}",
        flush=True,
    )
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()