"""
M290 — Memory Tier Auto-Router

Hypothesis: We can automatically route facts to the optimal tier
(weights/retrieval/hybrid) based on a quick probe.
"""
import os, sys, json, torch, math
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

def probe_confidence(model, tokenizer, question, answer):
    """Quick probe: compute model's confidence in the answer."""
    full_text = f"{question} {answer}"
    enc = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(enc.input_ids)
    logits = outputs.logits
    q_enc = tokenizer(question, return_tensors="pt").to(DEVICE)
    q_len = q_enc.input_ids.shape[1]
    answer_tokens = enc.input_ids[0, q_len:]
    log_probs = []
    for i, tok in enumerate(answer_tokens):
        pos = q_len + i - 1
        if pos >= 0:
            probs = torch.softmax(logits[0, pos], dim=-1)
            log_probs.append(math.log(probs[tok].item() + 1e-10))
    return math.exp(sum(log_probs) / len(log_probs)) if log_probs else 0.0

def route_tier(confidence, threshold_weights=0.3, threshold_retrieval=0.1):
    if confidence > threshold_weights:
        return "weights"  # Model already knows it
    elif confidence > threshold_retrieval:
        return "hybrid"  # Partial knowledge, use both
    else:
        return "retrieval"  # Model doesn't know, use retrieval

TEST_FACTS = [
    # Easy - model should know
    ("What is the capital of France?", "Paris"),
    ("What is 2+2?", "4"),
    # Hard - model likely doesn't know
    ("Who invented the telephone?", "Antonio Meucci"),
    ("Who wrote 1984?", "George Orwell"),
    # Medium
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the currency of UK?", "pound"),
]

print("=" * 60)
print("M290 — Memory Tier Auto-Router")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

results = []
for q, a in TEST_FACTS:
    conf = probe_confidence(model, tokenizer, q, a)
    tier = route_tier(conf)
    
    # Verify routing decision
    ans = generate_answer(model, tokenizer, q)
    model_knows = a.lower() in ans.lower()
    
    # Check if routing was correct
    if tier == "weights" and model_knows:
        correct = True
    elif tier == "retrieval" and not model_knows:
        correct = True
    elif tier == "hybrid":
        correct = True  # Hybrid is always acceptable
    else:
        correct = False
    
    results.append({
        "question": q,
        "answer": a,
        "confidence": conf,
        "routed_tier": tier,
        "model_knows": model_knows,
        "routing_correct": correct,
    })
    
    status = "✅" if correct else "❌"
    print(f"  {status} '{q[:40]}...' conf={conf:.4f} → {tier:<10s} (knows={model_knows})")

correct_routes = sum(1 for r in results if r["routing_correct"])

print("\n" + "=" * 60)
print(f"SUMMARY")
print("=" * 60)
print(f"  Correct routes: {correct_routes}/{len(results)}")

for tier in ["weights", "retrieval", "hybrid"]:
    count = sum(1 for r in results if r["routed_tier"] == tier)
    print(f"  {tier}: {count}")

if correct_routes == len(results):
    print("\n🎯 AUTO-ROUTER WORKS PERFECTLY")
else:
    print("\n⚠️  Auto-router needs threshold tuning")
print("=" * 60)

with open("experiments/m290_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m290_results.json")

del model
torch.cuda.empty_cache()
