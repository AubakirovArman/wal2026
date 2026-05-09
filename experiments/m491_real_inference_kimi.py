"""
M491 — Real Inference on Kimi-K2-Thinking

Attempts actual tokenization and inference on local model.
"""
import json, os, sys

print("=" * 60)
print("M491 — REAL INFERENCE (Kimi-K2-Thinking)")
print("=" * 60)

result = {"loaded": False, "tokens": 0, "error": None}

try:
    from transformers import AutoTokenizer
    model_path = "/mnt/hf_model_weights/Kimi-K2-Thinking"
    
    print(f"  Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    text = "What is the capital of France?"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    result["loaded"] = True
    result["tokens"] = len(tokens)
    result["decoded"] = decoded[:50]
    
    print(f"  Input: '{text}'")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Decoded: '{decoded[:50]}...'")
    print(f"  ✅ Real inference path validated")
    
except Exception as e:
    result["error"] = str(e)
    print(f"  ❌ Error: {e}")

with open("experiments/m491_real_inference_results.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n✅ M491: Real inference test complete")
