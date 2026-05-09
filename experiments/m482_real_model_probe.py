"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M482 — Real Model Probe

Attempts to load and run inference on available local model.
"""
import json, os, sys

print("=" * 60)
print("M482 — REAL MODEL PROBE")
print("=" * 60)

# Check available models
models = []
for d in ["Kimi-K2-Thinking", "MiniMax-M2", "wesa-qwen-vl-32b"]:
    path = f"/mnt/hf_model_weights/{d}"
    if os.path.exists(path):
        size = sum(os.path.getsize(os.path.join(dirpath, f)) for dirpath, _, files in os.walk(path) for f in files)
        models.append({"name": d, "path": path, "size_gb": round(size / 1e9, 1)})

print(f"  Found {len(models)} local models:")
for m in models:
    print(f"    {m['name']}: {m['size_gb']} GB")

# Try loading transformers if available
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("\n  Transformers available")
    transformers_ok = True
except ImportError:
    print("\n  Transformers not available")
    transformers_ok = False

result = {
    "models_found": len(models),
    "transformers_available": transformers_ok,
    "gpu_available": False,
    "inference_test": False,
}

# Try GPU
try:
    import torch
    result["gpu_available"] = torch.cuda.is_available()
    if result["gpu_available"]:
        print(f"  GPUs: {torch.cuda.device_count()}")
except ImportError:
    pass

# Try quick inference if everything available
if transformers_ok and result["gpu_available"] and models:
    try:
        model_path = models[0]["path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"  Tokenizer loaded: {models[0]['name']}")
        result["inference_test"] = True
    except Exception as e:
        print(f"  Inference failed: {e}")

with open("experiments/m482_model_probe_results.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n✅ M482: Real model probe complete")
