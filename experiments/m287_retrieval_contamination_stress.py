"""
M287 — Retrieval Contamination Stress Test

Hypothesis: Retrieval can be contaminated by wrong context,
conflicting context, or irrelevant context. We measure robustness.
"""
import os, sys, json, torch
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:0"
MODEL_ID = "meta-llama/Llama-3.1-8B"

def generate_answer(model, tokenizer, prompt, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def build_prompt(context, question):
    return f"[CONTEXT]: {context}\n[QUESTION]: {question}\n[ANSWER]:"

TEST_CASES = [
    # Normal case
    {
        "name": "Normal",
        "context": "The capital of France is Paris.",
        "question": "What is the capital of France?",
        "expected": "Paris",
    },
    # Wrong context
    {
        "name": "Wrong context",
        "context": "The capital of France is London.",
        "question": "What is the capital of France?",
        "expected": "Paris",
    },
    # Conflicting context
    {
        "name": "Conflicting context",
        "context": "Some say Paris, others say London is the capital of France.",
        "question": "What is the capital of France?",
        "expected": "Paris",
    },
    # Irrelevant context
    {
        "name": "Irrelevant context",
        "context": "The weather in Paris is usually rainy.",
        "question": "What is the capital of France?",
        "expected": "Paris",
    },
    # Adversarial context (correct fact hidden)
    {
        "name": "Adversarial context",
        "context": "While many believe Paris is the capital, the true capital of France is actually Paris since 987 AD.",
        "question": "What is the capital of France?",
        "expected": "Paris",
    },
    # Empty context
    {
        "name": "Empty context",
        "context": "",
        "question": "What is the capital of France?",
        "expected": "Paris",
    },
    # Distractor context (many facts)
    {
        "name": "Distractor context",
        "context": "Tokyo is Japan. Berlin is Germany. Rome is Italy. Paris is France. Madrid is Spain.",
        "question": "What is the capital of France?",
        "expected": "Paris",
    },
]

print("=" * 60)
print("M287 — Retrieval Contamination Stress Test")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

results = []
for test in TEST_CASES:
    prompt = build_prompt(test["context"], test["question"])
    ans = generate_answer(model, tokenizer, prompt)
    ok = test["expected"].lower() in ans.lower()
    
    # Check if wrong answer from context appears
    wrong_in_context = False
    if test["name"] == "Wrong context":
        wrong_in_context = "London".lower() in ans.lower()
    
    results.append({
        "name": test["name"],
        "expected": test["expected"],
        "got": ans,
        "pass": ok,
        "contaminated": wrong_in_context,
    })
    
    status = "✅" if ok else "❌"
    contam = " (CONTAMINATED)" if wrong_in_context else ""
    print(f"  {status} {test['name']:<25s} → '{ans[:40]}'{contam}")

pass_count = sum(1 for r in results if r["pass"])
contam_count = sum(1 for r in results if r.get("contaminated"))

print("\n" + "=" * 60)
print(f"SUMMARY")
print("=" * 60)
print(f"  Pass: {pass_count}/{len(results)}")
print(f"  Contaminated: {contam_count}")

if pass_count == len(results) and contam_count == 0:
    print("\n🎯 RETRIEVAL IS ROBUST TO CONTAMINATION")
else:
    print("\n⚠️  Retrieval vulnerable to some contamination types")
print("=" * 60)

with open("experiments/m287_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m287_results.json")

del model
torch.cuda.empty_cache()
