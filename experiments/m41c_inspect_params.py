#!/usr/bin/env python3
"""M41c: Inspect parameter devices and dtypes."""
import torch
from transformers import AutoModelForCausalLM

model_name = "unsloth/Llama-3.3-70B-Instruct"
print(f"Loading {model_name}...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

devices = {}
dtypes = {}
sizes_gb = {}
for name, param in model.named_parameters():
    d = str(param.device)
    t = str(param.dtype)
    devices[d] = devices.get(d, 0) + 1
    dtypes[t] = dtypes.get(t, 0) + 1
    sizes_gb[d] = sizes_gb.get(d, 0) + (param.numel() * param.element_size()) / 1e9

print(f"\nParameters by device:")
for d, count in sorted(devices.items()):
    print(f"  {d}: {count} params, {sizes_gb.get(d, 0):.1f}GB")

print(f"\nParameters by dtype:")
for t, count in sorted(dtypes.items()):
    print(f"  {t}: {count} params")

# Check a few specific layers
for name in ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.79.self_attn.q_proj.weight", "lm_head.weight"]:
    if hasattr(model, name.split('.')[0]):
        try:
            p = model.get_parameter(name)
            print(f"\n{name}: device={p.device}, dtype={p.dtype}, shape={tuple(p.shape)}")
        except Exception as e:
            print(f"\n{name}: error={e}")
