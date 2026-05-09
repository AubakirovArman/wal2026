#!/usr/bin/env python3
"""M41b: Run forward pass on 70B and check VRAM usage."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

print(f"Loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

for i in [2, 3]:
    allocated = torch.cuda.memory_allocated(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"GPU {i} before forward: {allocated:.1f}GB / {total:.1f}GB")

# Forward pass with dummy input
text = "The quick brown fox"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

print(f"\nForward pass done. Logits shape: {outputs.logits.shape}")

for i in [2, 3]:
    allocated = torch.cuda.memory_allocated(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"GPU {i} after forward: {allocated:.1f}GB / {total:.1f}GB")
