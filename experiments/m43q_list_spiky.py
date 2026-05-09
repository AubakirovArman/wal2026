"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M43q: List all spiky layers with std values."""
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

spiky = []
for name, param in model.named_parameters():
    if len(param.shape) != 2 or "embed" in name or "lm_head" in name:
        continue
    w = param.data.float()
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    std = w_norm.std().item()
    if std < 0.08:
        spiky.append((name, std))

print(f"Spiky layers: {len(spiky)}")
for name, std in spiky:
    print(f"  {name}: std={std:.4f}")
