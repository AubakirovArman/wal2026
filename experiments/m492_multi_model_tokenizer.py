"""
M492 — Multi-Model Tokenizer Comparison

Compares tokenization across available models.
"""
import json, os

models = ["Kimi-K2-Thinking", "MiniMax-M2", "wesa-qwen-vl-32b"]
text = "What is the capital of France?"

print("=" * 60)
print("M492 — MULTI-MODEL TOKENIZER COMPARISON")
print("=" * 60)

results = []
for model_name in models:
    path = f"/mnt/hf_model_weights/{model_name}"
    if not os.path.exists(path):
        continue
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        tokens = tokenizer.encode(text)
        results.append({"model": model_name, "tokens": len(tokens)})
        print(f"  {model_name}: {len(tokens)} tokens")
    except Exception as e:
        print(f"  {model_name}: failed ({e})")

with open("experiments/m492_tokenizer_comparison_results.json", "w") as f:
    json.dump({"models_tested": len(results), "results": results, "pass": len(results) > 0}, f, indent=2)

print("\n✅ M492: Multi-model tokenizer comparison complete")
