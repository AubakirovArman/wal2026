"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
import torch, sys
from transformers import AutoModelForCausalLM

print("Python started", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
print(f"CUDA devices: {torch.cuda.device_count()}", flush=True)
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} {torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB", flush=True)

model_name = "unsloth/Llama-3.3-70B-Instruct"
print(f"Loading {model_name}...", flush=True)
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)
print(f"Model loaded. Device map sample: {list(model.hf_device_map.items())[:3]}", flush=True)
print("TEST PASSED", flush=True)
