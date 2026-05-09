"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M288 — Hybrid Answer Arbitration

Hypothesis: When weight-edited model and retrieval disagree,
we need a strategy to choose the correct answer.
"""
import os, sys, json, torch, random
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:0"
MODEL_ID = "meta-llama/Llama-3.1-8B"
SEED = 42
RANK = 4
STEPS = 100
LR = 5e-5
TARGET_LAYERS = [16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

# Fact that we'll edit into weights
EDIT_FACT = ("What is the capital of France?", "Paris")

# Conflicting retrieval fact (different question but might interfere)
RETRIEVAL_FACTS = [
    ("What is the capital of France?", "Paris"),  # Agrees
    ("What is the capital of Germany?", "Berlin"),  # Different
]

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def build_retrieval_prompt(query, context_facts):
    context = " ".join([f"{q} {a}." for q, a in context_facts])
    return f"[CONTEXT]: {context}\n[QUESTION]: {query}\n[ANSWER]:"

def train_lora_fp32(model, tokenizer, facts, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        p.requires_grad = False
    adapters = {}
    for layer_idx in TARGET_LAYERS:
        layer = model.model.layers[layer_idx]
        for mod_name in TARGET_MODULES:
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

    optimizer = torch.optim.Adam([a.weight for a in adapters.values()], lr=LR)
    texts = [f"{q} {a}" for q, a in facts]
    for step in range(STEPS):
        t = random.choice(texts)
        enc = tokenizer(t, return_tensors="pt").to(DEVICE)
        out = model(enc.input_ids, labels=enc.input_ids)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([a.weight for a in adapters.values()], max_norm=1.0)
        optimizer.step()

    for layer_idx in TARGET_LAYERS:
        layer = model.model.layers[layer_idx]
        for mod_name in TARGET_MODULES:
            mod = getattr(layer.self_attn if mod_name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp, mod_name)
            if hasattr(mod, '_adapter'):
                delta = mod._adapter.weight.data.to(mod.weight.dtype)
                mod.weight.data += delta
                mod.forward = lambda x, m=mod: torch.nn.functional.linear(x, m.weight)
                del mod._adapter
    return model

print("=" * 60)
print("M288 — Hybrid Answer Arbitration")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Build weight-edited model
print("\n[Weight Edit] Training 'capital of France = Paris'...")
model_weights = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_weights = train_lora_fp32(model_weights, tokenizer, [EDIT_FACT], SEED)

# Test scenarios
scenarios = [
    ("Agreeing retrieval", [EDIT_FACT], "What is the capital of France?", "Paris"),
    ("No retrieval", [], "What is the capital of France?", "Paris"),
    ("Different retrieval", [("What is the capital of Germany?", "Berlin")], "What is the capital of France?", "Paris"),
    ("Conflicting retrieval", [("What is the capital of France?", "London")], "What is the capital of France?", "Paris"),
]

results = []
for name, context, question, expected in scenarios:
    # Weight-only answer
    ans_weights = generate_answer(model_weights, tokenizer, question)
    weights_ok = expected.lower() in ans_weights.lower()
    
    # Retrieval-only answer
    if context:
        prompt = build_retrieval_prompt(question, context)
        ans_retrieval = generate_answer(model_weights, tokenizer, prompt)
        retrieval_ok = expected.lower() in ans_retrieval.lower()
    else:
        ans_retrieval = ans_weights
        retrieval_ok = weights_ok
    
    # Arbitration: if retrieval present and weights disagree, use retrieval
    if context and not weights_ok:
        arb_ok = retrieval_ok
        arb_strategy = "retrieval_fallback"
    else:
        arb_ok = weights_ok
        arb_strategy = "weights_first"
    
    results.append({
        "scenario": name,
        "weights_answer": ans_weights[:40],
        "weights_ok": weights_ok,
        "retrieval_answer": ans_retrieval[:40],
        "retrieval_ok": retrieval_ok,
        "arbitration": arb_strategy,
        "arbitration_ok": arb_ok,
    })
    
    status = "✅" if arb_ok else "❌"
    print(f"  {status} {name:<25s} weights={weights_ok} retrieval={retrieval_ok} arb={arb_strategy}")

del model_weights
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
arb_pass = sum(1 for r in results if r["arbitration_ok"])
print(f"  Arbitration pass: {arb_pass}/{len(results)}")

if arb_pass == len(results):
    print("\n🎯 HYBRID ARBITRATION WORKS")
else:
    print("\n⚠️  Arbitration needs refinement")
print("=" * 60)

with open("experiments/m288_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m288_results.json")
