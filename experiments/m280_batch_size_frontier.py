"""
M280 — Batch Size Frontier: 1/3/5/10/20 Facts per Batch

Hypothesis: Larger batches train faster but may have lower survival
due to gradient conflicts. We find the optimal batch size.
"""
import os, sys, json, torch, random, time
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

ALL_FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
    ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of UK?", "London"),
    ("What is the capital of Russia?", "Moscow"),
    ("What is the capital of China?", "Beijing"),
    ("What is the capital of India?", "New Delhi"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Mexico?", "Mexico City"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Turkey?", "Ankara"),
    ("What is the capital of Greece?", "Athens"),
    ("What is the capital of Portugal?", "Lisbon"),
    ("What is the capital of Sweden?", "Stockholm"),
    ("What is the capital of Norway?", "Oslo"),
    ("What is the capital of Finland?", "Helsinki"),
]

BATCH_SIZES = [1, 3, 5, 10, 20]

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

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
print("M280 — Batch Size Frontier")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

results = []
for batch_size in BATCH_SIZES:
    facts = ALL_FACTS[:batch_size]
    print(f"\n[Batch size {batch_size}] Training {len(facts)} fact(s)...")
    
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model = train_lora_fp32(model, tokenizer, facts, SEED)
    train_time = time.time() - t0
    
    # Test survival
    survival = 0
    for q, a in facts:
        ans = generate_answer(model, tokenizer, q)
        if a.lower() in ans.lower():
            survival += 1
    
    survival_rate = survival / len(facts)
    print(f"  Time: {train_time:.1f}s")
    print(f"  Survival: {survival}/{len(facts)} = {survival_rate:.1%}")
    
    results.append({
        "batch_size": batch_size,
        "train_time": train_time,
        "survival": survival,
        "total": len(facts),
        "survival_rate": survival_rate,
    })
    
    del model
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'Size':<6} {'Time':<8} {'Survival':<10}")
print(f"  {'-'*6} {'-'*8} {'-'*10}")
for r in results:
    print(f"  {r['batch_size']:<6} {r['train_time']:.1f}s    {r['survival']}/{r['total']} = {r['survival_rate']:.1%}")

best = max(results, key=lambda x: x['survival_rate'])
print(f"\n🎯 Best batch size: {best['batch_size']} (survival {best['survival_rate']:.1%})")
print("=" * 60)

with open("experiments/m280_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m280_results.json")
