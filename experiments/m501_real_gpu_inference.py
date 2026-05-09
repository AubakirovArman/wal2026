"""
M501 — Real GPU Inference on Kimi-K2-Thinking

Attempts to load model on GPU and run inference.
"""
import json, torch, sys

print("=" * 60)
print("M501 — REAL GPU INFERENCE (Kimi-K2-Thinking)")
print("=" * 60)

result = {"model_loaded": False, "inference_done": False, "error": None, "gpu_memory_before_mb": [], "gpu_memory_after_mb": []}

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = "/mnt/hf_model_weights/Kimi-K2-Thinking"
    
    # Record GPU memory before
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated(i) / 1024**2
        result["gpu_memory_before_mb"].append(round(mem, 1))
    
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"  Loading model (device_map='auto', fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    result["model_loaded"] = True
    print(f"  ✅ Model loaded")
    
    # Inference
    text = "What is the capital of France?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    print(f"  Running inference...")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_new_tokens=10, do_sample=False)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result["inference_done"] = True
    result["answer"] = answer
    print(f"  ✅ Inference complete")
    print(f"  Input:  '{text}'")
    print(f"  Output: '{answer}'")
    
    # Record GPU memory after
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**2
        result["gpu_memory_after_mb"].append(round(mem, 1))
    
except Exception as e:
    result["error"] = str(e)
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

with open("experiments/m501_gpu_inference_results.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n✅ M501: Real GPU inference test complete")
