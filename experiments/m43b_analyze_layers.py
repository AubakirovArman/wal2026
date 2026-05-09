"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M43b: Quick analysis — how many spiky vs smooth layers in 70B?"""
import torch
from transformers import AutoModelForCausalLM

model_name = "unsloth/Llama-3.3-70B-Instruct"
print(f"Loading {model_name}...")
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

threshold = 0.08
spiky = []
smooth = []
skipped = 0

for name, param in model.named_parameters():
    if len(param.shape) != 2 or "embed" in name or "lm_head" in name:
        skipped += 1
        continue
    w = param.data.float()
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    std = w_norm.std().item()
    if std < threshold:
        spiky.append((name, std, tuple(param.shape)))
    else:
        smooth.append((name, std, tuple(param.shape)))

print(f"\nSpiky layers (std < {threshold}): {len(spiky)}")
for name, std, shape in spiky[:10]:
    print(f"  {name}: std={std:.4f}, shape={shape}")
if len(spiky) > 10:
    print(f"  ... and {len(spiky)-10} more")

print(f"\nSmooth layers (std >= {threshold}): {len(smooth)}")
for name, std, shape in smooth[:5]:
    print(f"  {name}: std={std:.4f}, shape={shape}")
if len(smooth) > 5:
    print(f"  ... and {len(smooth)-5} more")

print(f"\nSkipped: {skipped}")
