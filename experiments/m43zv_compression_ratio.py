"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M43zv: Calculate compression ratio for lmax=10, K=2048, skip spiky."""
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

K_TARGET = 2048
L_MAX = 10
COARSE_LADDER = [0.5 ** i for i in range(12)]
SPIKY_THRESHOLD = 0.08

original_size = 0
encoded_size = 0
skipped_size = 0
total_params = 0

for name, param in model.named_parameters():
    if len(param.shape) != 2:
        total_params += param.numel()
        original_size += param.numel() * 2  # bf16 = 2 bytes
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
    original_size += param.numel() * 2  # bf16

    if not is_spiky:
        # Encode
        ladder = torch.tensor(COARSE_LADDER, dtype=torch.float32)
        enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=L_MAX)
        cb, ids = build_codebook(enc.digits, enc.stop_depth, L_MAX)
        
        K_actual = min(K_TARGET, cb.keys.numel())
        # Codebook size: K_actual floats
        codebook_bytes = K_actual * 4
        # ID size: log2(K_actual) bits per weight, round up to bytes
        id_bits = max(1, (K_actual - 1).bit_length())
        id_bytes = (id_bits + 7) // 8  # Round up to whole bytes
        ids_bytes = param.numel() * id_bytes
        # Row scales: num_rows floats
        row_scale_bytes = param.shape[0] * 4
        
        param_encoded_size = codebook_bytes + ids_bytes + row_scale_bytes
        encoded_size += param_encoded_size
    else:
        # Skip: keep original
        skipped_size += param.numel() * 2

print(f"Total params: {total_params:,}")
print(f"Original size: {original_size / 1e9:.2f} GB")
print(f"Encoded size: {encoded_size / 1e9:.2f} GB")
print(f"Skipped size (original): {skipped_size / 1e9:.2f} GB")
print(f"Compression ratio: {original_size / encoded_size:.2f}x")
print(f"Size reduction: {(1 - encoded_size / original_size) * 100:.1f}%")
