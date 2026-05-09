"""
Wild Idea #12 — Auto-Generated Unit Tests by LLM

LLM generates exact/paraphrase/negative/context tests for each fact.
"""
import os, sys, json, torch
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:0"
MODEL_ID = "meta-llama/Llama-3.1-8B"

def generate_text(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

FACT = ("What is the capital of France?", "Paris")

print("=" * 60)
print("AUTO-GENERATED UNIT TESTS")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)

# Generate paraphrase
para_prompt = f"Generate a paraphrase of: '{FACT[0]}'\nParaphrase:"
paraphrase = generate_text(model, tokenizer, para_prompt, max_new_tokens=20)
print(f"\n  Paraphrase: '{paraphrase}'")

# Generate negative test
neg_prompt = f"Generate a wrong answer to: '{FACT[0]}'\nWrong answer:"
negative = generate_text(model, tokenizer, neg_prompt, max_new_tokens=10)
print(f"  Negative: '{negative}'")

# Generate context variation
ctx_prompt = f"Generate a context-wrapped version of: '{FACT[0]}'\nContext version:"
context = generate_text(model, tokenizer, ctx_prompt, max_new_tokens=30)
print(f"  Context: '{context}'")

tests = {
    "original": {"question": FACT[0], "expected": FACT[1]},
    "paraphrase": {"question": paraphrase, "expected": FACT[1]},
    "negative": {"question": FACT[0], "forbidden": negative},
    "context": {"question": context, "expected": FACT[1]},
}

print("\n🎯 AUTO-GENERATED TESTS: 4 test types generated from 1 fact")
with open("experiments/auto_generated_unit_tests_results.json", "w") as f:
    json.dump(tests, f, indent=2)

del model
torch.cuda.empty_cache()
