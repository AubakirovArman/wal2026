"""
M601 — Real GPU Inference on Qwen-VL-32B (≤70B compliant)

Attempts real model loading and inference on GPU.
"""
import json, torch, os

# Exclude busy GPUs 1 and 4
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,5,6,7"

print("=" * 60)
print("M601 — REAL GPU INFERENCE (Qwen-VL-32B)")
print("=" * 60)

result = {
    "schema_version": "wal.results.v1",
    "status": "NO_DATA",
    "pass": False,
    "model_loaded": False,
    "inference_done": False,
    "error": None,
}

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = "/mnt/hf_model_weights/wesa-qwen-vl-32b"
    
    print(f"  GPUs visible: {torch.cuda.device_count()}")
    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"  Loading model (auto device_map, fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    result["model_loaded"] = True
    print(f"  ✅ Model loaded on {model.device}")
    
    text = "What is the capital of France?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    print(f"  Running inference...")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_new_tokens=10, do_sample=False)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result["inference_done"] = True
    result["status"] = "PASS"
    result["pass"] = True
    result["answer"] = answer
    print(f"  ✅ Inference complete")
    print(f"  Input:  '{text}'")
    print(f"  Output: '{answer}'")
    
except Exception as e:
    result["error"] = str(e)
    error_text = str(e).lower()
    if "unrecognized configuration" in error_text or "unsupported" in error_text:
        result["status"] = "UNSUPPORTED"
        result["reason"] = "UNSUPPORTED_CONFIG"
    else:
        result["status"] = "FAIL"
        result["reason"] = "INFERENCE_ERROR"
    result["pass"] = False
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

with open("experiments/m601_gpu_qwen_results.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nM601: Real GPU inference status={result['status']}")
