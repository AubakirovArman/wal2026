"""
Wild Idea #16 — Behavioral Checksum

Checksum based on model behaviors, not weight bytes.
"""
import os, sys, json, torch
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:0"
MODEL_ID = "meta-llama/Llama-3.1-8B"

PROBES = [
    "What is the capital of France?",
    "What is 2+2?",
    "The capital of Japan is",
    "Water boils at",
]

def generate_answer(model, tokenizer, prompt, max_new_tokens=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def compute_behavioral_checksum(model, tokenizer):
    """Compute checksum from model behavior on probe set."""
    import hashlib
    behaviors = []
    for probe in PROBES:
        ans = generate_answer(model, tokenizer, probe)
        behaviors.append(f"{probe}:{ans}")
    combined = "|".join(behaviors)
    return hashlib.sha256(combined.encode()).hexdigest()[:16], behaviors

print("=" * 60)
print("BEHAVIORAL CHECKSUM")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)

checksum, behaviors = compute_behavioral_checksum(model, tokenizer)
print(f"\nBehavioral checksum: {checksum}")
for b in behaviors:
    print(f"  {b[:60]}")

# Compare with second run
checksum2, _ = compute_behavioral_checksum(model, tokenizer)
print(f"\nSecond run: {checksum2}")
print(f"Match: {'✅ YES' if checksum == checksum2 else '❌ NO'}")

with open("experiments/behavioral_checksum_results.json", "w") as f:
    json.dump({"checksum": checksum, "behaviors": behaviors, "stable": checksum == checksum2}, f, indent=2)

print("\n🎯 BEHAVIORAL CHECKSUM: Model identity = behavior fingerprint")
del model
torch.cuda.empty_cache()
