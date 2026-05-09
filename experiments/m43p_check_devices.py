#!/usr/bin/env python3
"""M43p: Check which layers are on which GPU."""
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

gpu2 = []
gpu3 = []
other = []
for name, param in model.named_parameters():
    d = str(param.device)
    if ":2" in d:
        gpu2.append(name)
    elif ":3" in d:
        gpu3.append(name)
    else:
        other.append(name)

print(f"GPU 2: {len(gpu2)} params")
for name in gpu2[:5]:
    print(f"  {name}")
print(f"  ... and {len(gpu2)-5} more")

print(f"\nGPU 3: {len(gpu3)} params")
for name in gpu3[:5]:
    print(f"  {name}")
print(f"  ... and {len(gpu3)-5} more")

if other:
    print(f"\nOther: {len(other)} params")
    for name in other[:5]:
        print(f"  {name}")
