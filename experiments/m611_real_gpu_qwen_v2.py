"""
M611 — Real GPU Inference on Qwen-VL-32B v2 (≤70B compliant)

Retry with AutoModel instead of AutoModelForCausalLM.
"""
import json, torch, os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,5,6,7"

print("=" * 60)
print("M611 — REAL GPU INFERENCE V2 (Qwen-VL-32B)")
print("=" * 60)

result = {"model_loaded": False, "inference_done": False, "error": None}

try:
    from transformers import AutoModel, AutoTokenizer
    model_path = "/mnt/hf_model_weights/wesa-qwen-vl-32b"
    
    print(f"  GPUs visible: {torch.cuda.device_count()}")
    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"  Loading model (AutoModel, fp16)...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    result["model_loaded"] = True
    print(f"  ✅ Model loaded")
    
    text = "What is the capital of France?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    print(f"  Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    result["inference_done"] = True
    result["output_shape"] = str(outputs[0].shape)
    print(f"  ✅ Forward pass complete")
    print(f"  Output shape: {outputs[0].shape}")
    
except Exception as e:
    result["error"] = str(e)
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

with open("experiments/m611_gpu_qwen_v2_results.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n✅ M611: Real GPU inference v2 complete")
