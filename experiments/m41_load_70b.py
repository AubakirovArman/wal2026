#!/usr/bin/env python3
"""M41: Load Llama 3.3 70B via unsloth mirror and inspect memory layout."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE_2 = torch.device("cuda:3")
DEVICE_3 = torch.device("cuda:3")

model_name = "unsloth/Llama-3.3-70B-Instruct"
print(f"Loading {model_name}...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

# Check device map
cpu_layers = []
gpu2_layers = []
gpu3_layers = []
other_layers = []
for name, param in model.named_parameters():
    device = str(param.device)
    if "cpu" in device:
        cpu_layers.append(name)
    elif ":2" in device:
        gpu2_layers.append(name)
    elif ":3" in device:
        gpu3_layers.append(name)
    else:
        other_layers.append(name)

print(f"\nGPU 2 layers: {len(gpu2_layers)}")
print(f"GPU 3 layers: {len(gpu3_layers)}")
print(f"CPU layers: {len(cpu_layers)}")
print(f"Other: {len(other_layers)}")

if cpu_layers:
    print(f"\nFirst 5 CPU-offloaded layers: {cpu_layers[:5]}")

# VRAM usage
for i in [2, 3]:
    allocated = torch.cuda.memory_allocated(i) / 1e9
    reserved = torch.cuda.memory_reserved(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"\nGPU {i}: allocated={allocated:.1f}GB, reserved={reserved:.1f}GB, total={total:.1f}GB")

print("\nModel ready for inference.")
