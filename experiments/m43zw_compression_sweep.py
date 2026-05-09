#!/usr/bin/env python3
"""M43zw: Compression ratio sweep for different K values."""
import torch
from transformers import AutoModelForCausalLM
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from route_encoder import encode_routes
from codebook import build_codebook

model_name = "unsloth/Llama-3.3-70B-Instruct"
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

L_MAX = 10
COARSE_LADDER = [0.5 ** i for i in range(12)]
SPIKY_THRESHOLD = 0.08

original_size = 0
encoded_sizes = {k: 0 for k in [128, 256, 512, 1024, 2048]}
skipped_size = 0
total_params = 0

for name, param in model.named_parameters():
    if len(param.shape) != 2:
        total_params += param.numel()
        original_size += param.numel() * 2
        skipped_size += param.numel() * 2
        continue
    if "embed_tokens" in name or "lm_head" in name:
        total_params += param.numel()
        original_size += param.numel() * 2
        skipped_size += param.numel() * 2
        continue

    w = param.data.float()
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    std = w_norm.std().item()
    is_spiky = std < SPIKY_THRESHOLD

    total_params += param.numel()
    original_size += param.numel() * 2

    if not is_spiky:
        ladder = torch.tensor(COARSE_LADDER, dtype=torch.float32)
        enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=L_MAX)
        cb, ids = build_codebook(enc.digits, enc.stop_depth, L_MAX)
        
        for K_TARGET in encoded_sizes.keys():
            K_actual = min(K_TARGET, cb.keys.numel())
            codebook_bytes = K_actual * 4
            id_bits = max(1, (K_actual - 1).bit_length())
            id_bytes = (id_bits + 7) // 8
            ids_bytes = param.numel() * id_bytes
            row_scale_bytes = param.shape[0] * 4
            param_encoded_size = codebook_bytes + ids_bytes + row_scale_bytes
            encoded_sizes[K_TARGET] += param_encoded_size
    else:
        skipped_size += param.numel() * 2

print(f"Total params: {total_params:,}")
print(f"Original size: {original_size / 1e9:.2f} GB")
print(f"Skipped size: {skipped_size / 1e9:.2f} GB")
print()
for K_TARGET in sorted(encoded_sizes.keys()):
    total_encoded = encoded_sizes[K_TARGET] + skipped_size
    ratio = original_size / total_encoded
    reduction = (1 - total_encoded / original_size) * 100
    print(f"K={K_TARGET}: encoded={encoded_sizes[K_TARGET]/1e9:.2f} GB, total={total_encoded/1e9:.2f} GB, ratio={ratio:.2f}x, reduction={reduction:.1f}%")
