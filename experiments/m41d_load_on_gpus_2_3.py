"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M41d: Load Llama 70B only on GPU 2 and 3."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "unsloth/Llama-3.3-70B-Instruct"
print(f"Loading {model_name} onto GPU 2+3 only...")

max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

devices = {}
sizes_gb = {}
for name, param in model.named_parameters():
    d = str(param.device)
    devices[d] = devices.get(d, 0) + 1
    sizes_gb[d] = sizes_gb.get(d, 0) + (param.numel() * param.element_size()) / 1e9

print(f"\nParameters by device:")
for d, count in sorted(devices.items()):
    print(f"  {d}: {count} params, {sizes_gb.get(d, 0):.1f}GB")

for i in [2, 3]:
    allocated = torch.cuda.memory_allocated(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"\nGPU {i}: allocated={allocated:.1f}GB / {total:.1f}GB")

# Forward pass
text = "The quick brown fox"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(f"\nForward pass OK. Logits: {outputs.logits.shape}")

for i in [2, 3]:
    allocated = torch.cuda.memory_allocated(i) / 1e9
    print(f"GPU {i} after forward: {allocated:.1f}GB")
