"""Central Gemma-4-31B weight loader for all WAL experiments.

Each experiment can import this module and call load_gemma_weight(key) to get
the right tensor instead of hardcoding model paths and layer names.
"""
import json, os
import torch
from pathlib import Path
from safetensors import safe_open

GEMMA_SNAPSHOT = "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
_INDEX = None
_OPEN_SHARDS = {}

def _get_index():
    global _INDEX
    if _INDEX is None:
        idx_path = os.path.join(GEMMA_SNAPSHOT, "model.safetensors.index.json")
        with open(idx_path) as f:
            _INDEX = json.load(f)["weight_map"]
    return _INDEX

def _get_shard(shard_name: str):
    if shard_name not in _OPEN_SHARDS:
        shard_path = os.path.join(GEMMA_SNAPSHOT, shard_name)
        _OPEN_SHARDS[shard_name] = safe_open(shard_path, framework="pt", device="cpu")
    return _OPEN_SHARDS[shard_name]

def load_gemma_weight(tensor_name: str, device: str = "cpu") -> torch.Tensor:
    """Load a single weight tensor from Gemma-4-31B safetensors."""
    idx = _get_index()
    shard_name = idx[tensor_name]
    shard = _get_shard(shard_name)
    return shard.get_tensor(tensor_name).to(device)

def close_all():
    for s in _OPEN_SHARDS.values():
        s.close()
    _OPEN_SHARDS.clear()

# Gemma text model layer structure:
# language_model.layers.{N}.self_attn.{q,k,v,o}_proj.weight
# language_model.layers.{N}.mlp.{gate,up,down}_proj.weight
# language_model.layers.{N}.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight

GEMMA_SHAPES = {
    "q_proj": (8192, 5376),
    "k_proj": (4096, 5376),
    "v_proj": (4096, 5376),
    "o_proj": (5376, 8192),
    "gate_proj": (21504, 5376),
    "up_proj": (21504, 5376),
    "down_proj": (5376, 21504),
}

GEMMA_CONFIG = {
    "hidden_size": 5376,
    "intermediate_size": 21504,
    "num_layers": 60,
    "num_attention_heads": 16,  # q_norm = 256 -> 16 heads × 16 (norm per head)
    "num_key_value_heads": 8,
    "head_dim": 256,
    "vocab_size": 262144,
    "prefix": "language_model",
}

# Famous layers for benchmarks
GEMMA_LAYERS = {
    "layer0": 0,
    "layer30": 30,
    "layer50": 50,
    "layer59": 59,
}

def get_layer_weight(layer_idx: int, proj: str) -> torch.Tensor:
    """Get a specific layer's weight tensor."""
    name = f"language_model.layers.{layer_idx}.{proj}.weight"
    return load_gemma_weight(name)

def all_layer_names() -> list[str]:
    """All unique layer tensor names in the index."""
    return sorted(_get_index().keys())
