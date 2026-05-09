"""
M281 — Negative-Test-Aware Training

Hypothesis: Adding negative prompts (with correct answers) to the
training set improves robustness against wrong-answer injection.
"""
import os, sys, json, torch, random, math
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:0"
MODEL_ID = "meta-llama/Llama-3.1-8B"
SEED = 42
RANK = 4
STEPS = 150  # More steps for negative training
LR = 5e-5
TARGET_LAYERS = [16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

POSITIVE_FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
]

# Negative prompts: questions where model might answer wrong + correct answer
NEGATIVE_PROMPTS = [
    ("What is the capital of Germany?", "Berlin"),  # Don't say Paris
    ("What is 2+2?", "4"),  # Don't say 5
    ("Who invented the telephone?", "Alexander Graham Bell"),  # Don't say Meucci/Tesla
]

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def compute_ppl(model, tokenizer, text="The quick brown fox jumps over the lazy dog."):
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(enc.input_ids, labels=enc.input_ids)
    return math.exp(out.loss.item())

def train_lora_fp32(model, tokenizer, positive_facts, negative_prompts, seed):
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
    
    # Build training texts: positives + negatives (treated equally)
    pos_texts = [f"{q} {a}" for q, a in positive_facts]
    neg_texts = [f"{q} {a}" for q, a in negative_prompts]
    all_texts = pos_texts + neg_texts
    
    for step in range(STEPS):
        t = random.choice(all_texts)
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
print("M281 — Negative-Test-Aware Training")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Baseline: train only on positives
print("\n[Baseline] Training on positive facts only...")
model_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_base = train_lora_fp32(model_base, tokenizer, POSITIVE_FACTS, [], SEED)

# Test baseline
print("\n[Baseline Tests]")
exact_base = sum(1 for q, a in POSITIVE_FACTS if a.lower() in generate_answer(model_base, tokenizer, q).lower())
neg_tests = [
    ("What is the capital of Germany?", "Paris"),
    ("What is 2+2?", "5"),
]
neg_base = sum(1 for q, wrong in neg_tests if wrong.lower() not in generate_answer(model_base, tokenizer, q).lower())
ppl_base = compute_ppl(model_base, tokenizer)
print(f"  Exact: {exact_base}/{len(POSITIVE_FACTS)}")
print(f"  Negative: {neg_base}/{len(neg_tests)}")
print(f"  PPL: {ppl_base:.2f}")

del model_base
torch.cuda.empty_cache()

# Negative-aware: train on positives + negatives
print("\n[Negative-Aware] Training on positives + negative prompts...")
model_neg = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_neg = train_lora_fp32(model_neg, tokenizer, POSITIVE_FACTS, NEGATIVE_PROMPTS, SEED)

# Test negative-aware
print("\n[Negative-Aware Tests]")
exact_neg = sum(1 for q, a in POSITIVE_FACTS if a.lower() in generate_answer(model_neg, tokenizer, q).lower())
neg_neg = sum(1 for q, wrong in neg_tests if wrong.lower() not in generate_answer(model_neg, tokenizer, q).lower())
ppl_neg = compute_ppl(model_neg, tokenizer)
print(f"  Exact: {exact_neg}/{len(POSITIVE_FACTS)}")
print(f"  Negative: {neg_neg}/{len(neg_tests)}")
print(f"  PPL: {ppl_neg:.2f}")

# Additional negative tests
print("\n[Extended Negative Tests]")
extended_negs = [
    ("What is the capital of Germany?", "Paris"),
    ("What is 2+2?", "5"),
    ("What is the capital of Japan?", "Paris"),  # Cross-contamination
    ("What is the capital of France?", "Tokyo"),  # Cross-contamination
]
for q, wrong in extended_negs:
    ans_base = "N/A"
    ans_neg = generate_answer(model_neg, tokenizer, q)
    ok = wrong.lower() not in ans_neg.lower()
    print(f"  {'✅' if ok else '❌'} '{q[:40]}...' → '{ans_neg[:30]}'")

del model_neg
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"  Exact:      {exact_base}/{len(POSITIVE_FACTS)} → {exact_neg}/{len(POSITIVE_FACTS)}")
print(f"  Negative:   {neg_base}/{len(neg_tests)} → {neg_neg}/{len(neg_tests)}")
print(f"  PPL:        {ppl_base:.2f} → {ppl_neg:.2f}")

if neg_neg > neg_base:
    print("\n🎯 NEGATIVE-AWARE TRAINING IMPROVES ROBUSTNESS")
else:
    print("\n⚠️  No improvement from negative-aware training")
print("=" * 60)

results = {
    "baseline": {"exact": exact_base, "negative": neg_base, "ppl": ppl_base},
    "negative_aware": {"exact": exact_neg, "negative": neg_neg, "ppl": ppl_neg},
}
with open("experiments/m281_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m281_results.json")
