#!/usr/bin/env python3
"""M43n: Check block means for spiky layers."""
import torch
from transformers import AutoModelForCausalLM

model_name = "unsloth/Llama-3.3-70B-Instruct"
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

for name in [
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.1.self_attn.q_proj.weight",
    "model.language_model.layers.2.self_attn.q_proj.weight",
]:
    param = dict(model.named_parameters())[name]
    w = param.data.float()
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    rows, cols = w_norm.shape
    block_size = 4
    num_br = rows // block_size
    num_bc = cols // block_size
    blocks = (
        w_norm.view(num_br, block_size, num_bc, block_size)
        .permute(0, 2, 1, 3)
        .reshape(num_br * num_bc, block_size * block_size)
    )
    print(f"{name}: block_mean={blocks.mean(dim=1).abs().mean():.6f}, block_std={blocks.std(dim=1).mean():.6f}")
