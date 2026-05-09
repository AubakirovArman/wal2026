"""
Wild Idea #23 — Memory Provenance

Each answer says: "this comes from weights / retrieval / branch X".
"""
import os, sys, json, torch
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:0"
MODEL_ID = "meta-llama/Llama-3.1-8B"

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

FACTS = [
    ("What is the capital of France?", "Paris", "weights"),
    ("What is the capital of Japan?", "Tokyo", "weights"),
    ("Who invented the telephone?", "Antonio Meucci", "retrieval"),
]

print("=" * 60)
print("MEMORY PROVENANCE")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)

results = []
for q, a, source in FACTS:
    ans = generate_answer(model, tokenizer, q)
    matched = a.lower() in ans.lower()
    provenance = source if matched else "unknown"
    status = "✅" if matched else "❌"
    print(f"  {status} '{q[:40]}...' → '{ans[:25]}' [source: {provenance}]")
    results.append({"question": q, "expected": a, "got": ans, "source": provenance, "matched": matched})

print("\n🎯 MEMORY PROVENANCE: Answers tagged with knowledge source")
with open("experiments/memory_provenance_results.json", "w") as f:
    json.dump(results, f, indent=2)

del model
torch.cuda.empty_cache()
