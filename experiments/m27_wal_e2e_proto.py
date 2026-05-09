"""M27 WAL-E2E Step 11a: end-to-end learned ISA with a semantic encoder.

This probe stops learning a semantic ISA purely post-hoc from an already fixed
stage-id stream. Instead it trains a small semantic encoder jointly with a
two-level instruction bank on sampled block programs.

For each target layer:
  - initialize a small set of semantic families and family-conditioned atoms
  - train a semantic encoder that emits high-level family IDs and low-level atom
    IDs under a joint objective:
      * stage-token reconstruction
      * semantic neighborhood consistency
      * family coherence and anti-collapse regularization
  - measure two surfaces:
      1. exact WAL-E2E: FAMILY + low-level atoms + literal corrections
      2. e2e_only: FAMILY-conditioned low-level atoms only, no corrections
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
from torch.nn import functional as F
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


def _nearest_phrases(slot_sequences: torch.Tensor, phrases: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    total_rows = int(slot_sequences.shape[0])
    assignments = torch.empty(total_rows, dtype=torch.int64)
    distances = torch.empty(total_rows, dtype=torch.int16)
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = slot_sequences[start:end]
        dist = (chunk[:, None, :] != phrases[None, :, :]).sum(dim=2)
        best_dist, best_idx = dist.min(dim=1)
        assignments[start:end] = best_idx.to(torch.int64)
        distances[start:end] = best_dist.to(torch.int16)
    return assignments, distances


def _learn_slot_phrases(
    slot_sequences: torch.Tensor,
    num_variants: int,
    base: int,
    iters: int,
    assign_chunk_size: int,
) -> dict[str, object]:
    train_rows = int(slot_sequences.shape[0])
    if train_rows < 1:
        raise ValueError("slot_sequences must be non-empty")
    if train_rows < num_variants:
        pad = slot_sequences[torch.arange(num_variants) % train_rows]
        slot_sequences = pad.contiguous()
        train_rows = int(slot_sequences.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    perm = torch.randperm(train_rows, generator=generator)
    phrases = slot_sequences[perm[:num_variants]].clone()
    history = []
    for iter_idx in range(iters):
        assignments, distances = _nearest_phrases(slot_sequences, phrases, assign_chunk_size)
        counts = torch.bincount(assignments, minlength=num_variants)
        updated = phrases.clone()
        for pos in range(slot_sequences.shape[1]):
            flat = assignments * base + slot_sequences[:, pos].to(torch.int64)
            hist = torch.bincount(flat, minlength=num_variants * base).view(num_variants, base)
            updated[:, pos] = hist.argmax(dim=1).to(torch.uint8)
        empty = (counts == 0).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            reseed = torch.randint(0, train_rows, (empty.numel(),), generator=generator)
            updated[empty] = slot_sequences[reseed]
        phrases = updated.contiguous()
        history.append(
            {
                "iter": int(iter_idx + 1),
                "mean_hamming": float(distances.to(torch.float32).mean().item()),
                "p50_hamming": float(torch.quantile(distances.to(torch.float32), 0.50).item()),
                "p90_hamming": float(torch.quantile(distances.to(torch.float32), 0.90).item()),
            }
        )
    final_assignments, final_distances = _nearest_phrases(slot_sequences, phrases, assign_chunk_size)
    final_counts = torch.bincount(final_assignments, minlength=num_variants)
    return {
        "phrases": phrases,
        "train_counts": final_counts,
        "train_mean_hamming": float(final_distances.to(torch.float32).mean().item()),
        "history": history,
    }


def _clone_group_with_program_matrix(group: BlockRVQEncoding, program_matrix: torch.Tensor) -> BlockRVQEncoding:
    shape = group.stage_ids[0].shape
    stage_ids = []
    for stage_idx, ids in enumerate(group.stage_ids):
        stage_ids.append(program_matrix[:, stage_idx].reshape(shape).to(dtype=ids.dtype))
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


def _role_name(layer_name: str) -> str:
    if layer_name.endswith("gate_proj"):
        return "MLP_GATE"
    if layer_name.endswith("up_proj"):
        return "MLP_UP"
    raise ValueError(f"unsupported target for WAL-E2E: {layer_name}")


def _family_names_for_role(role_name: str, num_families: int) -> tuple[str, ...]:
    if num_families == 4:
        suffixes = ("COMMON_CORE", "COMMON_MIXED", "RARE_STRUCTURED", "OUTLIER_DIFFUSE")
        return tuple(f"{role_name}_{suffix}" for suffix in suffixes)
    return tuple(f"{role_name}_FAMILY_{idx:02d}" for idx in range(num_families))


def _instruction_name(family_name: str, slot_idx: int, variant_idx: int) -> str:
    return f"{family_name}_SLOT_{slot_idx}_ATOM_{variant_idx}"


def _compute_token_model(enc: GroupedBlockRVQEncoding, args: argparse.Namespace) -> dict[str, object]:
    sample_sequences = _sample_training_sequences(enc, int(args.train_samples), seed=0)
    raw_len = int(sample_sequences.shape[1])
    base = max(int(codebook.shape[0]) for group in enc.groups for codebook in group.codebooks)
    counts = torch.zeros(raw_len, base, dtype=torch.int64)
    for pos in range(raw_len):
        counts[pos] = torch.bincount(sample_sequences[:, pos].to(torch.int64), minlength=base)
    probs = (counts.to(torch.float32) + 1.0) / (float(sample_sequences.shape[0]) + float(base))
    stage_surprisal = (-torch.log(probs)).contiguous()
    modal_tokens = counts.argmax(dim=1).to(torch.uint8)
    return {
        "sample_sequences": sample_sequences,
        "raw_len": raw_len,
        "base": base,
        "stage_surprisal": stage_surprisal,
        "modal_tokens": modal_tokens,
    }


def _semantic_features(
    sequence_matrix: torch.Tensor,
    stage_surprisal: torch.Tensor,
    modal_tokens: torch.Tensor,
    phrase_len: int,
) -> torch.Tensor:
    rows = int(sequence_matrix.shape[0])
    raw_len = int(sequence_matrix.shape[1])
    num_slots = raw_len // phrase_len
    avg_surprisal = torch.zeros(rows, dtype=torch.float32)
    modal_match = torch.zeros(rows, dtype=torch.float32)
    slot_surprisal = torch.zeros(rows, num_slots, dtype=torch.float32)
    slot_modal = torch.zeros(rows, num_slots, dtype=torch.float32)
    for pos in range(raw_len):
        ids = sequence_matrix[:, pos].to(torch.int64)
        surprisal = stage_surprisal[pos][ids]
        avg_surprisal += surprisal
        slot_idx = pos // phrase_len
        slot_surprisal[:, slot_idx] += surprisal
        match = (sequence_matrix[:, pos] == modal_tokens[pos]).to(torch.float32)
        modal_match += match
        slot_modal[:, slot_idx] += match
    avg_surprisal /= float(raw_len)
    modal_match /= float(raw_len)
    slot_surprisal /= float(phrase_len)
    slot_modal /= float(phrase_len)
    return torch.cat(
        [avg_surprisal.unsqueeze(1), modal_match.unsqueeze(1), slot_surprisal, slot_modal],
        dim=1,
    ).contiguous()


def _combined_family_distance(
    norm_features: torch.Tensor,
    sequence_matrix: torch.Tensor,
    feature_centroids: torch.Tensor,
    sequence_centroids: torch.Tensor,
    semantic_weight: float,
    sequence_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_dist = (norm_features[:, None, :] - feature_centroids[None, :, :]).square().mean(dim=2)
    sequence_dist = (sequence_matrix[:, None, :] != sequence_centroids[None, :, :]).to(torch.float32).mean(dim=2)
    combined = semantic_weight * feature_dist + sequence_weight * sequence_dist
    return combined, feature_dist, sequence_dist


def _learn_semantic_families(name: str, enc: GroupedBlockRVQEncoding, args: argparse.Namespace) -> dict[str, object]:
    token_model = _compute_token_model(enc, args)
    sample_sequences = token_model["sample_sequences"]
    raw_features = _semantic_features(
        sample_sequences,
        token_model["stage_surprisal"],
        token_model["modal_tokens"],
        int(args.phrase_len),
    )
    feature_mean = raw_features.mean(dim=0)
    feature_std = raw_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    norm_features = ((raw_features - feature_mean) / feature_std).contiguous()

    num_families = int(args.num_families)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    perm = torch.randperm(int(sample_sequences.shape[0]), generator=generator)
    feature_centroids = norm_features[perm[:num_families]].clone()
    sequence_centroids = sample_sequences[perm[:num_families]].clone()
    history = []
    semantic_weight = float(args.init_semantic_weight)
    sequence_weight = float(args.init_sequence_weight)

    for iter_idx in range(int(args.init_family_iters)):
        combined, feature_dist, sequence_dist = _combined_family_distance(
            norm_features,
            sample_sequences,
            feature_centroids,
            sequence_centroids,
            semantic_weight,
            sequence_weight,
        )
        best_dist, assignments = combined.min(dim=1)
        counts = torch.bincount(assignments, minlength=num_families)
        updated_feature_centroids = feature_centroids.clone()
        updated_sequence_centroids = sequence_centroids.clone()
        for family_idx in range(num_families):
            mask = assignments == family_idx
            if bool(mask.any()):
                updated_feature_centroids[family_idx] = norm_features[mask].mean(dim=0)
                family_sequences = sample_sequences[mask]
                for pos in range(int(sample_sequences.shape[1])):
                    hist = torch.bincount(family_sequences[:, pos].to(torch.int64), minlength=int(token_model["base"]))
                    updated_sequence_centroids[family_idx, pos] = int(hist.argmax().item())
            else:
                reseed = int(torch.randint(0, int(sample_sequences.shape[0]), (1,), generator=generator).item())
                updated_feature_centroids[family_idx] = norm_features[reseed]
                updated_sequence_centroids[family_idx] = sample_sequences[reseed]
        feature_centroids = updated_feature_centroids.contiguous()
        sequence_centroids = updated_sequence_centroids.contiguous()
        history.append(
            {
                "iter": int(iter_idx + 1),
                "mean_combined": float(best_dist.mean().item()),
                "mean_feature_dist": float(feature_dist.gather(1, assignments.unsqueeze(1)).mean().item()),
                "mean_sequence_dist": float(sequence_dist.gather(1, assignments.unsqueeze(1)).mean().item()),
                "min_family_count": int(counts.min().item()),
            }
        )

    combined, feature_dist, sequence_dist = _combined_family_distance(
        norm_features,
        sample_sequences,
        feature_centroids,
        sequence_centroids,
        semantic_weight,
        sequence_weight,
    )
    _, assignments = combined.min(dim=1)
    centroid_stats = []
    for family_idx in range(num_families):
        mask = assignments == family_idx
        if bool(mask.any()):
            centroid_stats.append(
                {
                    "family_idx": int(family_idx),
                    "avg_surprisal": float(raw_features[mask, 0].mean().item()),
                    "modal_match": float(raw_features[mask, 1].mean().item()),
                }
            )
        else:
            centroid_stats.append(
                {
                    "family_idx": int(family_idx),
                    "avg_surprisal": float("inf"),
                    "modal_match": 0.0,
                }
            )
    order = sorted(centroid_stats, key=lambda item: (item["avg_surprisal"], -item["modal_match"]))
    permute = torch.tensor([int(item["family_idx"]) for item in order], dtype=torch.int64)
    inverse = torch.empty_like(permute)
    inverse[permute] = torch.arange(num_families, dtype=torch.int64)
    feature_centroids = feature_centroids[permute]
    sequence_centroids = sequence_centroids[permute]
    assignments = inverse[assignments]
    family_names = _family_names_for_role(_role_name(name), num_families)
    token_model.update(
        {
            "raw_features": raw_features,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_centroids": feature_centroids,
            "sequence_centroids": sequence_centroids,
            "sample_family_ids": assignments,
            "family_names": family_names,
            "family_history": history,
        }
    )
    return token_model


class SemanticISAEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_families: int,
        num_slots: int,
        num_slot_variants: int,
        phrase_len: int,
        base: int,
        init_phrases: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_families = int(num_families)
        self.num_slots = int(num_slots)
        self.num_slot_variants = int(num_slot_variants)
        self.phrase_len = int(phrase_len)
        self.base = int(base)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.family_head = nn.Linear(hidden_dim, num_families)
        self.atom_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_families * num_slot_variants) for _ in range(num_slots)]
        )
        phrase_logits = torch.full(
            (num_families, num_slots, num_slot_variants, phrase_len, base),
            -4.0,
            dtype=torch.float32,
        )
        for family_idx in range(num_families):
            for slot_idx in range(num_slots):
                for variant_idx in range(num_slot_variants):
                    for pos_idx in range(phrase_len):
                        phrase_logits[
                            family_idx,
                            slot_idx,
                            variant_idx,
                            pos_idx,
                            int(init_phrases[family_idx, slot_idx, variant_idx, pos_idx].item()),
                        ] = 4.0
        self.phrase_logits = nn.Parameter(phrase_logits)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        hidden = self.backbone(features)
        family_logits = self.family_head(hidden)
        atom_logits = [head(hidden).view(-1, self.num_families, self.num_slot_variants) for head in self.atom_heads]
        return family_logits, atom_logits, hidden


def _build_initial_phrase_bank(
    sample_sequences: torch.Tensor,
    init_family_ids: torch.Tensor,
    base: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    raw_len = int(sample_sequences.shape[1])
    phrase_len = int(args.phrase_len)
    num_slots = raw_len // phrase_len
    num_families = int(args.num_families)
    num_slot_variants = int(args.num_slot_variants)
    init_phrases = torch.empty(num_families, num_slots, num_slot_variants, phrase_len, dtype=torch.uint8)
    init_atom_ids = torch.zeros(int(sample_sequences.shape[0]), num_slots, dtype=torch.int64)
    slot_summaries = []
    for family_idx in range(num_families):
        family_mask = init_family_ids == family_idx
        family_sequences = sample_sequences[family_mask]
        if int(family_sequences.shape[0]) == 0:
            family_sequences = sample_sequences
        family_slot_rows = []
        for slot_idx in range(num_slots):
            start = slot_idx * phrase_len
            end = start + phrase_len
            learned = _learn_slot_phrases(
                family_sequences[:, start:end],
                num_slot_variants,
                base,
                int(args.atom_iters),
                int(args.assign_chunk_size),
            )
            init_phrases[family_idx, slot_idx] = learned["phrases"]
            variants = []
            for variant_idx in range(num_slot_variants):
                variants.append(
                    {
                        "variant_id": int(variant_idx),
                        "train_count": int(learned["train_counts"][variant_idx].item()),
                        "phrase": learned["phrases"][variant_idx].to(torch.int64).tolist(),
                    }
                )
            family_slot_rows.append(
                {
                    "slot_index": int(slot_idx),
                    "train_mean_hamming": float(learned["train_mean_hamming"]),
                    "history": learned["history"],
                    "variants": variants,
                }
            )
        slot_summaries.append({"family_id": int(family_idx), "slots": family_slot_rows})

    for family_idx in range(num_families):
        family_mask = init_family_ids == family_idx
        if not bool(family_mask.any()):
            continue
        family_sequences = sample_sequences[family_mask]
        for slot_idx in range(num_slots):
            start = slot_idx * phrase_len
            end = start + phrase_len
            assignments, _ = _nearest_phrases(
                family_sequences[:, start:end],
                init_phrases[family_idx, slot_idx],
                int(args.assign_chunk_size),
            )
            init_atom_ids[family_mask, slot_idx] = assignments
    return init_phrases, init_atom_ids, slot_summaries


def _soft_joint_loss(
    model: SemanticISAEncoder,
    features: torch.Tensor,
    sequences: torch.Tensor,
    init_family_ids: torch.Tensor,
    init_atom_ids: torch.Tensor,
    family_prior: torch.Tensor,
    args: argparse.Namespace,
    warmup_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    family_logits, atom_logits, _ = model(features)
    family_probs = F.softmax(family_logits / float(args.family_temp), dim=1)
    phrase_probs = F.softmax(model.phrase_logits / float(args.phrase_temp), dim=-1)

    recon_loss = torch.zeros((), dtype=torch.float32)
    atom_entropy = torch.zeros((), dtype=torch.float32)
    batch_size = int(features.shape[0])
    row_idx = torch.arange(batch_size)
    for slot_idx in range(model.num_slots):
        slot_atom_probs = F.softmax(atom_logits[slot_idx] / float(args.atom_temp), dim=2)
        atom_entropy = atom_entropy + (-(slot_atom_probs.clamp_min(1e-9) * slot_atom_probs.clamp_min(1e-9).log()).sum(dim=2).mean())
        mixture = family_probs.unsqueeze(2) * slot_atom_probs
        slot_phrase_probs = phrase_probs[:, slot_idx]
        token_dist = torch.einsum("nfv,fvpb->npb", mixture, slot_phrase_probs)
        start = slot_idx * model.phrase_len
        end = start + model.phrase_len
        target = sequences[:, start:end].to(torch.int64)
        gathered = token_dist.gather(2, target.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
        recon_loss = recon_loss + (-gathered.log().mean())
    recon_loss = recon_loss / float(model.num_slots)
    atom_entropy = atom_entropy / float(model.num_slots)

    family_entropy = (-(family_probs.clamp_min(1e-9) * family_probs.clamp_min(1e-9).log()).sum(dim=1).mean())
    batch_family_mass = family_probs.mean(dim=0)
    balance_loss = (batch_family_mass - family_prior).square().mean()
    usage_floor_loss = F.relu(float(args.min_family_mass) - batch_family_mass).mean()
    phrase_entropy = (-(phrase_probs.clamp_min(1e-9) * phrase_probs.clamp_min(1e-9).log()).sum(dim=-1).mean())

    norm_features = F.normalize(features, dim=1)
    sim = norm_features @ norm_features.t()
    sim.fill_diagonal_(-1e9)
    nn_idx = sim.argmax(dim=1)
    contrastive = 1.0 - F.cosine_similarity(family_probs, family_probs[nn_idx], dim=1).mean()

    family_mass = family_probs.sum(dim=0).clamp_min(1e-6)
    centroids = family_probs.t() @ features / family_mass.unsqueeze(1)
    sq_dist = (features[:, None, :] - centroids[None, :, :]).square().mean(dim=2)
    cohesion = (family_probs * sq_dist).sum(dim=1).mean()

    loss = recon_loss
    loss = loss + float(args.contrastive_weight) * contrastive
    loss = loss + float(args.cohesion_weight) * cohesion
    loss = loss + float(args.balance_weight) * balance_loss
    loss = loss + float(args.usage_floor_weight) * usage_floor_loss
    loss = loss + float(args.phrase_entropy_weight) * phrase_entropy
    loss = loss + float(args.assignment_entropy_weight) * (family_entropy + atom_entropy)

    family_ce = torch.zeros((), dtype=torch.float32)
    atom_ce = torch.zeros((), dtype=torch.float32)
    supervision_weight = warmup_weight + float(args.persistent_supervision_weight)
    if supervision_weight > 0.0:
        family_ce = F.cross_entropy(family_logits, init_family_ids)
        for slot_idx in range(model.num_slots):
            selected_atom_logits = atom_logits[slot_idx][row_idx, init_family_ids]
            atom_ce = atom_ce + F.cross_entropy(selected_atom_logits, init_atom_ids[:, slot_idx])
        atom_ce = atom_ce / float(model.num_slots)
        loss = loss + supervision_weight * (family_ce + atom_ce)

    with torch.no_grad():
        phrase_tokens = model.phrase_logits.argmax(dim=-1).to(torch.uint8)
        hard_family = family_logits.argmax(dim=1)
        hard_approx = torch.empty_like(sequences)
        for slot_idx in range(model.num_slots):
            selected_atom_logits = atom_logits[slot_idx][row_idx, hard_family]
            hard_atom = selected_atom_logits.argmax(dim=1)
            hard_approx[:, slot_idx * model.phrase_len:(slot_idx + 1) * model.phrase_len] = phrase_tokens[:, slot_idx][hard_family, hard_atom]
        hard_token_match = float((hard_approx == sequences).to(torch.float32).mean().item())

    stats = {
        "loss": float(loss.item()),
        "recon_loss": float(recon_loss.item()),
        "contrastive": float(contrastive.item()),
        "cohesion": float(cohesion.item()),
        "balance": float(balance_loss.item()),
        "usage_floor": float(usage_floor_loss.item()),
        "phrase_entropy": float(phrase_entropy.item()),
        "family_entropy": float(family_entropy.item()),
        "atom_entropy": float(atom_entropy.item()),
        "family_ce": float(family_ce.item()),
        "atom_ce": float(atom_ce.item()),
        "supervision_weight": float(supervision_weight),
        "hard_token_match": hard_token_match,
    }
    return loss, stats


def _predict_hard_from_features(
    model: SemanticISAEncoder,
    norm_features: torch.Tensor,
    phrase_tokens: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_rows = int(norm_features.shape[0])
    family_ids = torch.empty(total_rows, dtype=torch.int64)
    atom_ids = torch.empty(total_rows, model.num_slots, dtype=torch.int64)
    approx_sequences = torch.empty(total_rows, model.num_slots * model.phrase_len, dtype=torch.uint8)
    logit_margin = torch.empty(total_rows, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            family_logits, atom_logits, _ = model(norm_features[start:end])
            hard_family = family_logits.argmax(dim=1)
            family_ids[start:end] = hard_family
            if model.num_families > 1:
                top2 = torch.topk(family_logits, k=2, dim=1).values
                logit_margin[start:end] = top2[:, 0] - top2[:, 1]
            else:
                logit_margin[start:end] = 0.0
            row_idx = torch.arange(end - start)
            for slot_idx in range(model.num_slots):
                selected_atom_logits = atom_logits[slot_idx][row_idx, hard_family]
                hard_atom = selected_atom_logits.argmax(dim=1)
                atom_ids[start:end, slot_idx] = hard_atom
                approx_sequences[
                    start:end,
                    slot_idx * model.phrase_len:(slot_idx + 1) * model.phrase_len,
                ] = phrase_tokens[:, slot_idx][hard_family, hard_atom]
    return family_ids, atom_ids, approx_sequences, logit_margin


def _train_e2e_isa(name: str, enc: GroupedBlockRVQEncoding, args: argparse.Namespace) -> dict[str, object]:
    family_model = _learn_semantic_families(name, enc, args)
    sample_sequences = family_model["sample_sequences"].to(torch.uint8).contiguous()
    raw_features = family_model["raw_features"].to(torch.float32).contiguous()
    norm_features = ((raw_features - family_model["feature_mean"]) / family_model["feature_std"]).to(torch.float32).contiguous()
    init_family_ids = family_model["sample_family_ids"].to(torch.int64).contiguous()
    base = int(family_model["base"])
    raw_len = int(family_model["raw_len"])
    phrase_len = int(args.phrase_len)
    num_slots = raw_len // phrase_len

    init_phrases, init_atom_ids, init_slot_summaries = _build_initial_phrase_bank(sample_sequences, init_family_ids, base, args)
    model = SemanticISAEncoder(
        input_dim=int(norm_features.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_families=int(args.num_families),
        num_slots=num_slots,
        num_slot_variants=int(args.num_slot_variants),
        phrase_len=phrase_len,
        base=base,
        init_phrases=init_phrases,
    ).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    family_prior = (
        torch.bincount(init_family_ids, minlength=int(args.num_families)).to(torch.float32)
        / max(int(init_family_ids.numel()), 1)
    )
    family_buckets = [(init_family_ids == family_idx).nonzero(as_tuple=True)[0] for family_idx in range(int(args.num_families))]
    history = []
    total_rows = int(sample_sequences.shape[0])
    batch_size = min(int(args.train_batch_size), total_rows)
    log_every = max(1, int(args.log_every))
    t0 = time.time()
    for step in range(1, int(args.train_steps) + 1):
        picks = []
        per_family = max(1, batch_size // int(args.num_families))
        for bucket in family_buckets:
            if int(bucket.numel()) == 0:
                continue
            draw = bucket[torch.randint(0, int(bucket.numel()), (per_family,), generator=generator)]
            picks.append(draw)
        if picks:
            pick = torch.cat(picks, dim=0)
        else:
            pick = torch.empty(0, dtype=torch.int64)
        if int(pick.numel()) < batch_size:
            extra = torch.randint(0, total_rows, (batch_size - int(pick.numel()),), generator=generator)
            pick = torch.cat([pick, extra], dim=0)
        elif int(pick.numel()) > batch_size:
            keep = torch.randperm(int(pick.numel()), generator=generator)[:batch_size]
            pick = pick[keep]
        shuffle = torch.randperm(batch_size, generator=generator)
        pick = pick[shuffle]
        batch_features = norm_features[pick]
        batch_sequences = sample_sequences[pick]
        batch_init_family = init_family_ids[pick]
        batch_init_atoms = init_atom_ids[pick]
        warmup = float(args.init_supervision_weight) * max(0.0, 1.0 - float(step - 1) / max(float(args.warmup_steps), 1.0))
        optimizer.zero_grad(set_to_none=True)
        loss, stats = _soft_joint_loss(
            model,
            batch_features,
            batch_sequences,
            batch_init_family,
            batch_init_atoms,
            family_prior,
            args,
            warmup,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == int(args.train_steps):
            dt = time.time() - t0
            entry = {
                "step": int(step),
                "elapsed_s": float(dt),
                "warmup_weight": float(warmup),
                **stats,
            }
            history.append(entry)
            print(
                f"[wal-e2e/train] {name}: step={step}/{args.train_steps} loss={stats['loss']:.4f} "
                f"recon={stats['recon_loss']:.4f} hard_match={stats['hard_token_match']:.4f} "
                f"contrast={stats['contrastive']:.4f} cohesion={stats['cohesion']:.4f} "
                f"usage_floor={stats['usage_floor']:.4f}",
                flush=True,
            )

    phrase_tokens = model.phrase_logits.detach().argmax(dim=-1).to(torch.uint8).cpu()
    sample_family_orig, sample_atom_ids, sample_approx, sample_margin = _predict_hard_from_features(
        model,
        norm_features,
        phrase_tokens,
        int(args.assign_chunk_size),
    )
    num_families = int(args.num_families)
    family_stats = []
    for family_idx in range(num_families):
        mask = sample_family_orig == family_idx
        if bool(mask.any()):
            family_stats.append(
                {
                    "family_idx": int(family_idx),
                    "avg_surprisal": float(raw_features[mask, 0].mean().item()),
                    "modal_match": float(raw_features[mask, 1].mean().item()),
                }
            )
        else:
            family_stats.append(
                {
                    "family_idx": int(family_idx),
                    "avg_surprisal": float("inf"),
                    "modal_match": 0.0,
                }
            )
    order = sorted(family_stats, key=lambda item: (item["avg_surprisal"], -item["modal_match"]))
    family_order = torch.tensor([int(item["family_idx"]) for item in order], dtype=torch.int64)
    family_inverse = torch.empty_like(family_order)
    family_inverse[family_order] = torch.arange(num_families, dtype=torch.int64)
    semantic_family_ids = family_inverse[sample_family_orig]
    family_names = _family_names_for_role(_role_name(name), num_families)

    feature_centroids = torch.zeros(num_families, int(norm_features.shape[1]), dtype=torch.float32)
    sequence_centroids = torch.zeros(num_families, raw_len, dtype=torch.uint8)
    family_sample_summaries = []
    for semantic_idx, family_name in enumerate(family_names):
        mask = semantic_family_ids == semantic_idx
        if bool(mask.any()):
            feature_centroids[semantic_idx] = norm_features[mask].mean(dim=0)
            for pos in range(raw_len):
                hist = torch.bincount(sample_sequences[mask, pos].to(torch.int64), minlength=base)
                sequence_centroids[semantic_idx, pos] = int(hist.argmax().item())
            family_sample_summaries.append(
                {
                    "family_id": int(semantic_idx),
                    "family_name": family_name,
                    "sample_count": int(mask.sum().item()),
                    "sample_share": float(mask.to(torch.float32).mean().item()),
                    "sample_avg_surprisal": float(raw_features[mask, 0].mean().item()),
                    "sample_modal_match": float(raw_features[mask, 1].mean().item()),
                    "sample_logit_margin": float(sample_margin[mask].mean().item()),
                    "sample_e2e_only_token_match": float((sample_approx[mask] == sample_sequences[mask]).to(torch.float32).mean().item()),
                }
            )
        else:
            feature_centroids[semantic_idx] = family_model["feature_centroids"][semantic_idx]
            sequence_centroids[semantic_idx] = family_model["sequence_centroids"][semantic_idx]
            family_sample_summaries.append(
                {
                    "family_id": int(semantic_idx),
                    "family_name": family_name,
                    "sample_count": 0,
                    "sample_share": 0.0,
                    "sample_avg_surprisal": 0.0,
                    "sample_modal_match": 0.0,
                    "sample_logit_margin": 0.0,
                    "sample_e2e_only_token_match": 0.0,
                }
            )

    return {
        "model": model,
        "base": base,
        "raw_len": raw_len,
        "num_slots": num_slots,
        "sample_sequences": sample_sequences,
        "stage_surprisal": family_model["stage_surprisal"],
        "modal_tokens": family_model["modal_tokens"],
        "feature_mean": family_model["feature_mean"].to(torch.float32),
        "feature_std": family_model["feature_std"].to(torch.float32),
        "feature_centroids": feature_centroids,
        "sequence_centroids": sequence_centroids,
        "phrase_tokens": phrase_tokens,
        "family_order": family_order,
        "family_inverse": family_inverse,
        "family_names": family_names,
        "training_history": history,
        "family_sample_summaries": family_sample_summaries,
        "init_slot_summaries": init_slot_summaries,
        "sample_token_match": float((sample_approx == sample_sequences).to(torch.float32).mean().item()),
        "sample_family_entropy": float(
            -(
                (torch.bincount(semantic_family_ids, minlength=num_families).to(torch.float64) / max(int(sample_sequences.shape[0]), 1)).clamp_min(1e-12)
                * (torch.bincount(semantic_family_ids, minlength=num_families).to(torch.float64) / max(int(sample_sequences.shape[0]), 1)).clamp_min(1e-12).log()
            ).sum().item()
        ),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


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


def _build_e2e_layer(
    name: str,
    enc: GroupedBlockRVQEncoding,
    args: argparse.Namespace,
) -> tuple[GroupedBlockRVQEncoding, dict[str, object], dict[str, object]]:
    trained = _train_e2e_isa(name, enc, args)
    model = trained["model"]
    phrase_tokens = trained["phrase_tokens"]
    family_inverse = trained["family_inverse"]
    family_names = trained["family_names"]
    feature_mean = trained["feature_mean"]
    feature_std = trained["feature_std"]
    feature_centroids = trained["feature_centroids"]
    sequence_centroids = trained["sequence_centroids"]
    stage_surprisal = trained["stage_surprisal"]
    modal_tokens = trained["modal_tokens"]
    raw_len = int(trained["raw_len"])
    phrase_len = int(args.phrase_len)
    num_slots = raw_len // phrase_len
    num_families = int(args.num_families)

    total_blocks = 0
    total_program_units = 0
    total_high_level_calls = 0
    total_low_level_calls = 0
    total_literal_corrections = 0
    total_low_level_tokens = 0
    total_e2e_only_hamming = 0
    total_tokens = 0
    family_count_totals = torch.zeros(num_families, dtype=torch.int64)
    family_surprisal_sum = torch.zeros(num_families, dtype=torch.float64)
    family_modal_match_sum = torch.zeros(num_families, dtype=torch.float64)
    family_feature_radius_sum = torch.zeros(num_families, dtype=torch.float64)
    family_sequence_radius_sum = torch.zeros(num_families, dtype=torch.float64)
    family_margin_sum = torch.zeros(num_families, dtype=torch.float64)
    family_logit_margin_sum = torch.zeros(num_families, dtype=torch.float64)
    family_low_level_calls = torch.zeros(num_families, dtype=torch.int64)
    family_literal_corrections = torch.zeros(num_families, dtype=torch.int64)
    family_program_units = torch.zeros(num_families, dtype=torch.int64)
    family_low_level_tokens = torch.zeros(num_families, dtype=torch.int64)
    family_token_match_hamming = torch.zeros(num_families, dtype=torch.int64)
    family_instruction_usage: Counter[tuple[int, int, int]] = Counter()
    group_rows = []
    approx_groups = []
    semantic_weight = float(args.report_semantic_weight)
    sequence_weight = float(args.report_sequence_weight)

    for group in enc.groups:
        seq_mat = _flatten_group_sequences(group)
        rows = int(seq_mat.shape[0])
        total_blocks += rows
        total_tokens += rows * raw_len
        total_high_level_calls += rows
        approx_program = torch.empty_like(seq_mat)
        group_low_level_calls = torch.zeros(rows, dtype=torch.int32)
        group_literal_corrections = torch.zeros(rows, dtype=torch.int32)
        group_program_units = torch.ones(rows, dtype=torch.int32)
        group_family_ids = torch.empty(rows, dtype=torch.int64)
        offset = 0
        for start in range(0, rows, int(args.assign_chunk_size)):
            end = min(start + int(args.assign_chunk_size), rows)
            chunk = seq_mat[start:end]
            raw_features = _semantic_features(chunk, stage_surprisal, modal_tokens, phrase_len)
            norm_features = ((raw_features - feature_mean) / feature_std).to(torch.float32)
            hard_family_orig, hard_atom_ids, approx_chunk, logit_margin = _predict_hard_from_features(
                model,
                norm_features,
                phrase_tokens,
                int(args.assign_chunk_size),
            )
            semantic_family_ids = family_inverse[hard_family_orig]
            group_family_ids[start:end] = semantic_family_ids
            approx_program[start:end] = approx_chunk
            feature_dist = (norm_features[:, None, :] - feature_centroids[None, :, :]).square().mean(dim=2)
            sequence_dist = (chunk[:, None, :] != sequence_centroids[None, :, :]).to(torch.float32).mean(dim=2)
            combined = semantic_weight * feature_dist + sequence_weight * sequence_dist
            own_idx = semantic_family_ids.unsqueeze(1)
            own_combined = combined.gather(1, own_idx).squeeze(1)
            own_feature = feature_dist.gather(1, own_idx).squeeze(1)
            own_sequence = sequence_dist.gather(1, own_idx).squeeze(1)
            if num_families > 1:
                other = combined.clone()
                other.scatter_(1, own_idx, float("inf"))
                margin = other.min(dim=1).values - own_combined
            else:
                margin = torch.zeros_like(own_combined)
            for slot_idx in range(num_slots):
                start_pos = slot_idx * phrase_len
                end_pos = start_pos + phrase_len
                dist_i32 = (chunk[:, start_pos:end_pos] != approx_chunk[:, start_pos:end_pos]).sum(dim=1).to(torch.int32)
                use_atom = (dist_i32 + 1) < phrase_len
                saved_tokens = torch.where(use_atom, phrase_len - dist_i32, torch.zeros_like(dist_i32))
                group_low_level_calls[start:end] += use_atom.to(torch.int32)
                group_literal_corrections[start:end] += torch.where(use_atom, dist_i32, torch.full_like(dist_i32, phrase_len))
                group_program_units[start:end] += torch.where(use_atom, dist_i32 + 1, torch.full_like(dist_i32, phrase_len))
                total_low_level_tokens += int(saved_tokens.sum().item())
                total_e2e_only_hamming += int(dist_i32.sum().item())
                for family_idx in range(num_families):
                    mask = semantic_family_ids == family_idx
                    if not bool(mask.any()):
                        continue
                    family_low_level_tokens[family_idx] += int(saved_tokens[mask].sum().item())
                    family_token_match_hamming[family_idx] += int(dist_i32[mask].sum().item())
                    selected_atom = hard_atom_ids[mask, slot_idx]
                    selected_use_atom = use_atom[mask]
                    if bool(selected_use_atom.any()):
                        slot_counts = torch.bincount(
                            selected_atom[selected_use_atom],
                            minlength=int(args.num_slot_variants),
                        )
                        for variant_idx, count in enumerate(slot_counts.tolist()):
                            if count > 0:
                                family_instruction_usage[(int(family_idx), int(slot_idx), int(variant_idx))] += int(count)
            for family_idx in range(num_families):
                mask = semantic_family_ids == family_idx
                if not bool(mask.any()):
                    continue
                count = int(mask.sum().item())
                family_count_totals[family_idx] += count
                family_surprisal_sum[family_idx] += float(raw_features[mask, 0].sum().item())
                family_modal_match_sum[family_idx] += float(raw_features[mask, 1].sum().item())
                family_feature_radius_sum[family_idx] += float(own_feature[mask].sum().item())
                family_sequence_radius_sum[family_idx] += float(own_sequence[mask].sum().item())
                family_margin_sum[family_idx] += float(margin[mask].sum().item())
                family_logit_margin_sum[family_idx] += float(logit_margin[mask].sum().item())
            offset = end
        total_low_level_calls += int(group_low_level_calls.sum().item())
        total_literal_corrections += int(group_literal_corrections.sum().item())
        total_program_units += int(group_program_units.sum().item())
        for family_idx in range(num_families):
            mask = group_family_ids == family_idx
            if not bool(mask.any()):
                continue
            family_low_level_calls[family_idx] += int(group_low_level_calls[mask].sum().item())
            family_literal_corrections[family_idx] += int(group_literal_corrections[mask].sum().item())
            family_program_units[family_idx] += int(group_program_units[mask].sum().item())
        family_probs = torch.bincount(group_family_ids, minlength=num_families).to(torch.float64) / max(rows, 1)
        group_rows.append(
            {
                "group_idx": int(len(group_rows)),
                "block_count": rows,
                "family_entropy": float(-(family_probs.clamp_min(1e-12) * family_probs.clamp_min(1e-12).log()).sum().item()),
            }
        )
        approx_groups.append(_clone_group_with_program_matrix(group, approx_program))

    e2e_only = GroupedBlockRVQEncoding(
        groups=tuple(approx_groups),
        row_slices=enc.row_slices,
        original_shape=enc.original_shape,
    )

    family_probs = family_count_totals.to(torch.float64) / max(total_blocks, 1)
    family_entropy = float(-(family_probs.clamp_min(1e-12) * family_probs.clamp_min(1e-12).log()).sum().item())
    family_summaries = []
    for family_idx, family_name in enumerate(family_names):
        count = int(family_count_totals[family_idx].item())
        atom_usage_rows = []
        for slot_idx in range(num_slots):
            for variant_idx in range(int(args.num_slot_variants)):
                atom_usage = int(family_instruction_usage.get((family_idx, slot_idx, variant_idx), 0))
                if atom_usage > 0:
                    atom_usage_rows.append(
                        {
                            "instruction_name": _instruction_name(family_name, slot_idx, variant_idx),
                            "slot_index": int(slot_idx),
                            "variant_id": int(variant_idx),
                            "used_calls": atom_usage,
                        }
                    )
        atom_usage_rows.sort(key=lambda item: item["used_calls"], reverse=True)
        if count > 0:
            family_summaries.append(
                {
                    "family_id": int(family_idx),
                    "family_name": family_name,
                    "block_count": count,
                    "block_share": float(count / max(total_blocks, 1)),
                    "avg_surprisal": float(family_surprisal_sum[family_idx].item() / count),
                    "avg_modal_match": float(family_modal_match_sum[family_idx].item() / count),
                    "avg_feature_radius": float(family_feature_radius_sum[family_idx].item() / count),
                    "avg_sequence_radius": float(family_sequence_radius_sum[family_idx].item() / count),
                    "avg_margin": float(family_margin_sum[family_idx].item() / count),
                    "avg_logit_margin": float(family_logit_margin_sum[family_idx].item() / count),
                    "avg_low_level_calls": float(family_low_level_calls[family_idx].item() / count),
                    "avg_literal_corrections": float(family_literal_corrections[family_idx].item() / count),
                    "avg_program_length": float(family_program_units[family_idx].item() / count),
                    "low_level_token_coverage": float(family_low_level_tokens[family_idx].item() / max(count * raw_len, 1)),
                    "e2e_only_token_match": float(1.0 - family_token_match_hamming[family_idx].item() / max(count * raw_len, 1)),
                    "top_atoms": atom_usage_rows[:10],
                }
            )
        else:
            family_summaries.append(
                {
                    "family_id": int(family_idx),
                    "family_name": family_name,
                    "block_count": 0,
                    "block_share": 0.0,
                    "avg_surprisal": 0.0,
                    "avg_modal_match": 0.0,
                    "avg_feature_radius": 0.0,
                    "avg_sequence_radius": 0.0,
                    "avg_margin": 0.0,
                    "avg_logit_margin": 0.0,
                    "avg_low_level_calls": 0.0,
                    "avg_literal_corrections": 0.0,
                    "avg_program_length": 0.0,
                    "low_level_token_coverage": 0.0,
                    "e2e_only_token_match": 0.0,
                    "top_atoms": [],
                }
            )

    top_atoms = []
    for (family_idx, slot_idx, variant_idx), used_calls in family_instruction_usage.most_common(12):
        top_atoms.append(
            {
                "instruction_name": _instruction_name(family_names[family_idx], slot_idx, variant_idx),
                "family_name": family_names[family_idx],
                "slot_index": int(slot_idx),
                "variant_id": int(variant_idx),
                "used_calls": int(used_calls),
            }
        )

    exact_stats = {
        "name": name,
        "raw_program_length": int(raw_len),
        "avg_program_length": float(total_program_units / max(total_blocks, 1)),
        "avg_high_level_calls": 1.0,
        "avg_low_level_calls": float(total_low_level_calls / max(total_blocks, 1)),
        "avg_literal_corrections": float(total_literal_corrections / max(total_blocks, 1)),
        "low_level_token_coverage": float(total_low_level_tokens / max(total_tokens, 1)),
        "hierarchical_instruction_share": float((total_high_level_calls + total_low_level_calls) / max(total_program_units, 1)),
        "program_compression_ratio": float(total_program_units / max(total_tokens, 1)),
        "instruction_vocab_size": int(num_families + num_families * num_slots * int(args.num_slot_variants)),
        "high_level_instruction_count": int(num_families),
        "low_level_instruction_count": int(num_families * num_slots * int(args.num_slot_variants)),
        "active_family_count": int(sum(1 for count in family_count_totals.tolist() if count > 0)),
        "family_entropy": float(family_entropy),
        "e2e_only_avg_hamming": float(total_e2e_only_hamming / max(total_blocks, 1)),
        "e2e_only_token_match": float(1.0 - total_e2e_only_hamming / max(total_tokens, 1)),
        "sample_e2e_only_token_match": float(trained["sample_token_match"]),
        "sample_family_entropy": float(trained["sample_family_entropy"]),
        "training_history": trained["training_history"],
        "family_sample_summaries": trained["family_sample_summaries"],
        "family_summaries": family_summaries,
        "init_slot_summaries": trained["init_slot_summaries"],
        "top_atoms": top_atoms,
        "group_rows": group_rows,
    }
    artifact = {
        "family_names": family_names,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_centroids": feature_centroids,
        "sequence_centroids": sequence_centroids,
        "phrase_tokens": phrase_tokens[trained["family_order"]].clone(),
        "family_order": trained["family_order"],
        "family_inverse": trained["family_inverse"],
        "family_sample_summaries": trained["family_sample_summaries"],
        "training_history": trained["training_history"],
        "state_dict": trained["state_dict"],
    }
    return e2e_only, exact_stats, artifact


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
    parser.add_argument("--num-families", type=int, default=4)
    parser.add_argument("--num-slot-variants", type=int, default=4)
    parser.add_argument("--phrase-len", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=65536)
    parser.add_argument("--atom-iters", type=int, default=6)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--init-family-iters", type=int, default=4)
    parser.add_argument("--init-semantic-weight", type=float, default=1.0)
    parser.add_argument("--init-sequence-weight", type=float, default=0.35)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--init-supervision-weight", type=float, default=0.35)
    parser.add_argument("--family-temp", type=float, default=1.0)
    parser.add_argument("--atom-temp", type=float, default=1.0)
    parser.add_argument("--phrase-temp", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.15)
    parser.add_argument("--cohesion-weight", type=float, default=0.10)
    parser.add_argument("--balance-weight", type=float, default=0.20)
    parser.add_argument("--usage-floor-weight", type=float, default=0.35)
    parser.add_argument("--min-family-mass", type=float, default=0.08)
    parser.add_argument("--phrase-entropy-weight", type=float, default=0.01)
    parser.add_argument("--assignment-entropy-weight", type=float, default=0.02)
    parser.add_argument("--persistent-supervision-weight", type=float, default=0.05)
    parser.add_argument("--report-semantic-weight", type=float, default=1.0)
    parser.add_argument("--report-sequence-weight", type=float, default=0.35)
    parser.add_argument("--matmul-strategy", default="full_weight_fast")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--bootstrap-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ldi_current_l54_gate_up.pt"))
    parser.add_argument("--current-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_e2e_current_l54_gate_up.pt"))
    parser.add_argument("--e2e-only-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_e2e_only_l54_gate_up.pt"))
    parser.add_argument("--e2e-artifact", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_e2e_isa_l54_gate_up.pt"))
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_e2e_proto_summary.json"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)

    current_cache = Path(args.current_cache)
    current_enc = _build_current_cache(args, current_cache, TARGETS)

    e2e_only_encodings: dict[str, GroupedBlockRVQEncoding] = {}
    wal_e2e_rows = []
    e2e_only_compare = []
    artifacts = {}
    for name in TARGETS:
        e2e_only, exact_stats, artifact = _build_e2e_layer(name, current_enc[name], args)
        e2e_only_encodings[name] = e2e_only
        wal_e2e_rows.append(exact_stats)
        e2e_only_compare.append({"name": name, **_compare_encodings(current_enc[name], e2e_only)})
        artifacts[name] = artifact
        print(
            f"[wal-e2e] {name}: avg_program={exact_stats['avg_program_length']:.3f}/{exact_stats['raw_program_length']} "
            f"low_calls={exact_stats['avg_low_level_calls']:.3f} entropy={exact_stats['family_entropy']:.3f} "
            f"e2e_only_match={exact_stats['e2e_only_token_match']:.3f}",
            flush=True,
        )

    e2e_artifact = Path(args.e2e_artifact)
    e2e_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "targets": list(TARGETS),
            "num_families": int(args.num_families),
            "num_slot_variants": int(args.num_slot_variants),
            "train_steps": int(args.train_steps),
            "artifacts": artifacts,
        },
        e2e_artifact,
    )
    e2e_only_cache = Path(args.e2e_only_cache)
    save_grouped_encoding_map(e2e_only_cache, e2e_only_encodings)

    print("\n[dense] evaluate untouched model", flush=True)
    dense = _run_dense_eval(ids, args.num_windows)
    print(f"\n[eval] current via {args.matmul_strategy}", flush=True)
    current_eval = _run_preencoded_eval(ids, current_cache, args.matmul_strategy, args.num_windows)
    print(f"\n[eval] e2e_only via {args.matmul_strategy}", flush=True)
    e2e_only_eval = _run_preencoded_eval(ids, e2e_only_cache, args.matmul_strategy, args.num_windows)
    exact_eval = dict(current_eval)
    exact_eval["by_construction_identical"] = True

    result = {
        "targets": list(TARGETS),
        "text_source": args.text_source,
        "num_windows": int(args.num_windows),
        "matmul_strategy": args.matmul_strategy,
        "num_families": int(args.num_families),
        "num_slot_variants": int(args.num_slot_variants),
        "phrase_len": int(args.phrase_len),
        "train_steps": int(args.train_steps),
        "hidden_dim": int(args.hidden_dim),
        "dense": dense,
        "current_cache": str(current_cache),
        "e2e_only_cache": str(e2e_only_cache),
        "e2e_artifact": str(e2e_artifact),
        "wal_e2e": wal_e2e_rows,
        "e2e_only_compare": e2e_only_compare,
        "current_eval": current_eval,
        "exact_eval": exact_eval,
        "e2e_only_eval": e2e_only_eval,
        "delta_exact": {
            "ppl_delta": 0.0,
            "tok_s_delta": 0.0,
            "peak_mb_delta": 0.0,
        },
        "delta_e2e_only": {
            "ppl_delta": float(e2e_only_eval["metrics"]["perplexity"] - current_eval["metrics"]["perplexity"]),
            "tok_s_delta": float(e2e_only_eval["metrics"]["tok_s"] - current_eval["metrics"]["tok_s"]),
            "peak_mb_delta": float(e2e_only_eval["eval_peak_mb"] - current_eval["eval_peak_mb"]),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    print(
        f"  dense:         ppl={dense['metrics']['perplexity']:.4f} tok/s={dense['metrics']['tok_s']:.2f} peak_mb={dense['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  current:       ppl={current_eval['metrics']['perplexity']:.4f} tok/s={current_eval['metrics']['tok_s']:.2f} peak_mb={current_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  wal_e2e_exact: ppl={exact_eval['metrics']['perplexity']:.4f} tok/s={exact_eval['metrics']['tok_s']:.2f} peak_mb={exact_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  e2e_only:      ppl={e2e_only_eval['metrics']['perplexity']:.4f} tok/s={e2e_only_eval['metrics']['tok_s']:.2f} peak_mb={e2e_only_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    for row, compare in zip(wal_e2e_rows, e2e_only_compare):
        print(
            f"  {row['name']}: avg_program={row['avg_program_length']:.3f}/{row['raw_program_length']} "
            f"low_calls={row['avg_low_level_calls']:.3f} family_entropy={row['family_entropy']:.3f} "
            f"e2e_only_rel_mse={compare['recon_rel_mse']:.6f}",
            flush=True,
        )
    print(
        f"  delta(e2e_only-current): ppl={result['delta_e2e_only']['ppl_delta']:+.6f} "
        f"tok/s={result['delta_e2e_only']['tok_s_delta']:+.2f} peak_mb={result['delta_e2e_only']['peak_mb_delta']:+.1f}",
        flush=True,
    )
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()