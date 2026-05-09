"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M43t: Count spiky layers with different thresholds."""
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

for threshold in [0.08, 0.05, 0.03, 0.02]:
    spiky = []
    for name, param in model.named_parameters():
        if len(param.shape) != 2 or "embed" in name or "lm_head" in name:
            continue
        w = param.data.float()
        row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        w_norm = w / row_scale
        std = w_norm.std().item()
        if std < threshold:
            spiky.append((name, std))
    print(f"\nThreshold {threshold}: {len(spiky)} spiky layers")
    for name, std in spiky:
        print(f"  {name}: {std:.4f}")
