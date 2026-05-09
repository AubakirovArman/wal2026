"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M256 — Retrieval Matcher: Auto-Tier Classification

Hypothesis: We can build a classifier that automatically decides
whether a fact needs weight editing (easy) or retrieval (hard).

Method: Test fact without context. If model already knows it → easy.
If not → hard. Then verify each tier gets correct backend.
"""
import os, sys, json, torch, random
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:0"
MODEL_ID = "meta-llama/Llama-3.1-8B"

# Facts with known difficulty
EASY_FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is 2+2?", "4"),
]

HARD_FACTS = [
    ("Who invented the telephone?", "Antonio Meucci"),
    ("Who wrote 1984?", "Aldous Huxley"),
    ("Who discovered radioactivity?", "Nikola Tesla"),
]

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def classify_fact(model, tokenizer, question, answer):
    """Classify as easy (model knows) or hard (needs retrieval)."""
    ans = generate_answer(model, tokenizer, question)
    knows = answer.lower() in ans.lower()
    return "easy" if knows else "hard"

def build_retrieval_prompt(query, fact_q, fact_a):
    return f"[CONTEXT]: {fact_q} {fact_a}.\n[QUESTION]: {query}\n[ANSWER]:"

def test_retrieval(model, tokenizer, question, answer, fact_q, fact_a):
    prompt = build_retrieval_prompt(question, fact_q, fact_a)
    ans = generate_answer(model, tokenizer, prompt)
    return answer.lower() in ans.lower()

def train_lora_fp32(model, tokenizer, fact, seed=42):
    """Minimal LoRA edit for one fact."""
    random.seed(seed)
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        p.requires_grad = False
    adapters = {}
    layer_idx = 16
    target_modules = ["o_proj", "q_proj", "v_proj", "gate_proj"]
    layer = model.model.layers[layer_idx]
    for mod_name in target_modules:
        mod = getattr(layer.self_attn if mod_name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp, mod_name)
        adapter = torch.nn.Linear(mod.weight.shape[1], mod.weight.shape[0], bias=False, device=DEVICE, dtype=torch.float32)
        torch.nn.init.zeros_(adapter.weight)
        adapters[f"{layer_idx}_{mod_name}"] = adapter
        mod._adapter = adapter
        original_forward = mod.forward
        def make_forward(orig, adapter):
            def forward(x):
                x_fp32 = x.to(torch.float32)
                out_fp32 = adapter(x_fp32)
                return orig(x) + out_fp32.to(x.dtype)
            return forward
        mod.forward = make_forward(original_forward, adapter)

    optimizer = torch.optim.Adam([a.weight for a in adapters.values()], lr=5e-5)
    text = f"{fact[0]} {fact[1]}"
    for step in range(100):
        enc = tokenizer(text, return_tensors="pt").to(DEVICE)
        out = model(enc.input_ids, labels=enc.input_ids)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([a.weight for a in adapters.values()], max_norm=1.0)
        optimizer.step()

    for mod_name in target_modules:
        mod = getattr(layer.self_attn if mod_name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp, mod_name)
        if hasattr(mod, '_adapter'):
            delta = mod._adapter.weight.data.to(mod.weight.dtype)
            mod.weight.data += delta
            mod.forward = lambda x, m=mod: torch.nn.functional.linear(x, m.weight)
            del mod._adapter
    return model

print("=" * 60)
print("M256 — Retrieval Matcher: Auto-Tier Classification")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Phase 1: Classify all facts
print("\n[Phase 1] Classifying facts...")
easy_classified = []
hard_classified = []
all_correct = True

for q, a in EASY_FACTS:
    tier = classify_fact(model, tokenizer, q, a)
    correct = tier == "easy"
    all_correct = all_correct and correct
    easy_classified.append({"q": q, "a": a, "predicted": tier, "correct": correct})
    print(f"  {'✅' if correct else '❌'} Easy: '{q[:40]}...' → {tier}")

for q, a in HARD_FACTS:
    tier = classify_fact(model, tokenizer, q, a)
    correct = tier == "hard"
    all_correct = all_correct and correct
    hard_classified.append({"q": q, "a": a, "predicted": tier, "correct": correct})
    print(f"  {'✅' if correct else '❌'} Hard: '{q[:40]}...' → {tier}")

# Phase 2: Verify backend selection
print("\n[Phase 2] Verifying backend selection...")

# Easy → weight editing should work
print("\n  Easy facts → weight editing:")
weight_easy_ok = 0
for item in easy_classified:
    if not item["correct"]:
        continue
    model2 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model2 = train_lora_fp32(model2, tokenizer, (item["q"], item["a"]))
    ans = generate_answer(model2, tokenizer, item["q"])
    ok = item["a"].lower() in ans.lower()
    weight_easy_ok += int(ok)
    print(f"    {'✅' if ok else '❌'} {item['q'][:35]}... → {ans[:30]}")
    del model2
    torch.cuda.empty_cache()

# Hard → retrieval should work  
print("\n  Hard facts → retrieval:")
retrieval_hard_ok = 0
for item in hard_classified:
    if not item["correct"]:
        continue
    ok = test_retrieval(model, tokenizer, item["q"], item["a"], item["q"], item["a"])
    retrieval_hard_ok += int(ok)
    print(f"    {'✅' if ok else '❌'} {item['q'][:35]}...")

# Cross-test: Easy → retrieval also works
print("\n  Easy facts → retrieval (cross-check):")
retrieval_easy_ok = 0
for item in easy_classified:
    ok = test_retrieval(model, tokenizer, item["q"], item["a"], item["q"], item["a"])
    retrieval_easy_ok += int(ok)
    print(f"    {'✅' if ok else '❌'} {item['q'][:35]}...")

# Cross-test: Hard → weights should fail
print("\n  Hard facts → weight editing (should fail):")
weight_hard_ok = 0
for item in hard_classified:
    if not item["correct"]:
        continue
    model2 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model2 = train_lora_fp32(model2, tokenizer, (item["q"], item["a"]))
    ans = generate_answer(model2, tokenizer, item["q"])
    ok = item["a"].lower() in ans.lower()
    weight_hard_ok += int(ok)
    print(f"    {'⚠️' if ok else '✅ (expected fail)'} {item['q'][:35]}... → {ans[:30]}")
    del model2
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Classification accuracy: {all_correct}")
print(f"Easy → weights: {weight_easy_ok}/{len(easy_classified)}")
print(f"Hard → retrieval: {retrieval_hard_ok}/{len(hard_classified)}")
print(f"Easy → retrieval (cross): {retrieval_easy_ok}/{len(easy_classified)}")
print(f"Hard → weights (should fail): {weight_hard_ok}/{len(hard_classified)}")

if all_correct and weight_easy_ok == len(easy_classified) and retrieval_hard_ok == len(hard_classified):
    print("\n🎯 RETRIEVAL MATCHER WORKS")
else:
    print("\n⚠️  Matcher partially working")
print("="*60)

results = {
    "classification_correct": all_correct,
    "easy_classified": easy_classified,
    "hard_classified": hard_classified,
    "weight_easy_ok": weight_easy_ok,
    "retrieval_hard_ok": retrieval_hard_ok,
    "retrieval_easy_ok": retrieval_easy_ok,
    "weight_hard_ok": weight_hard_ok,
}
with open("experiments/m256_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m256_results.json")
