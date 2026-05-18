from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dwl2_dynamic_route.src.calibrate import calibrate_ladder  # noqa: E402
from dwl2_dynamic_route.src.codebook import count_code_frequencies  # noqa: E402
from dwl2_dynamic_route.src.route_encoder import encode_routes  # noqa: E402

MODEL_DIR = "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TOPK = [8, 16, 32, 64, 128, 256]


def _load_index():
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _open_shard(shard_path):
    from safetensors import safe_open
    return safe_open(shard_path, framework="pt", device="cuda")


def _family_for_proj(proj: str) -> str:
    return "attn" if proj in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"


def _encode_counts(w: torch.Tensor, lmax: int = 12):
    row_max = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_max
    sample = w_norm.flatten()
    if sample.numel() > 2_000_000:
        idx = torch.randint(0, sample.numel(), (2_000_000,), device=sample.device)
        sample = sample[idx]
    ladder = calibrate_ladder(
        sample, l_max=lmax, refine_iters=20, pin_top=True, top_value=1.0, seed="geometric"
    )
    enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=lmax)
    keys, counts = count_code_frequencies(enc.digits, enc.stop_depth, lmax)
    return keys.cpu().tolist(), counts.cpu().tolist(), w.numel()


def _summarize(counter: dict[int, int]):
    total = sum(counter.values())
    pairs = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    top = []
    running = 0
    for k in TOPK:
        running = sum(v for _, v in pairs[:k])
        top.append({"top_k": k, "coverage": running / total if total else 0.0})
    return {
        "total_weights": total,
        "unique_routes": len(counter),
        "top_coverages": top,
        "top_16": [{"key": int(key), "count": int(count)} for key, count in pairs[:16]],
    }


def main():
    weight_map = _load_index()
    global_counter = defaultdict(int)
    family_counter = {"attn": defaultdict(int), "mlp": defaultdict(int)}
    open_shards = {}
    n_done = 0
    started = time.time()
    for layer in range(80):
        for proj in PROJECTIONS:
            tname = f"model.language_model.layers.{layer}.self_attn.{proj}.weight" if proj in {"q_proj", "k_proj", "v_proj", "o_proj"} \
                else f"model.language_model.layers.{layer}.mlp.{proj}.weight"
            shard = weight_map.get(tname)
            if shard is None:
                continue
            path = os.path.join(MODEL_DIR, shard)
            if path not in open_shards:
                open_shards[path] = _open_shard(path)
            w = open_shards[path].get_tensor(tname)
            keys, counts, total = _encode_counts(w)
            family = _family_for_proj(proj)
            for key, count in zip(keys, counts):
                global_counter[key] += count
                family_counter[family][key] += count
            n_done += 1
            if n_done % 20 == 0:
                print(f"  progress {n_done} tensors  elapsed={time.time() - started:.0f}s")

    for handle in open_shards.values():
        handle.__exit__(None, None, None)

    result = {
        "global": _summarize(global_counter),
        "attn": _summarize(family_counter["attn"]),
        "mlp": _summarize(family_counter["mlp"]),
    }
    print(json.dumps(result, indent=2))
    out = "/mnt/hf_model_weights/arman/3bit/dwl2_dynamic_route/results/m5a_route_frequency.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[m5a] wrote {out}")


if __name__ == "__main__":
    main()