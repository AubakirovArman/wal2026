"""
M279 — Long Chain: 10 Sequential Batches with Rehearsal

Hypothesis: Rehearsal prevents forgetting across many sequential batches.
"""
import os, sys, json, torch, random
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:3"
MODEL_ID = "meta-llama/Llama-3.1-8B"
SEED = 42
RANK = 4
STEPS = 50  # Shorter per batch for speed
LR = 5e-5
TARGET_LAYERS = [16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

BATCHES = [
    [("What is the capital of France?", "Paris"),
     ("What is the capital of Japan?", "Tokyo"),
     ("What is the capital of Italy?", "Rome")],
    [("What is the capital of Spain?", "Madrid"),
     ("What is the capital of Germany?", "Berlin"),
     ("What is the capital of UK?", "London")],
    [("What is the capital of Russia?", "Moscow"),
     ("What is the capital of China?", "Beijing"),
     ("What is the capital of India?", "New Delhi")],
    [("What is the capital of Brazil?", "Brasilia"),
     ("What is the capital of Canada?", "Ottawa"),
     ("What is the capital of Australia?", "Canberra")],
    [("What is 2+2?", "4"),
     ("What is 3+5?", "8"),
     ("What is 10-3?", "7")],
    [("What is the largest planet?", "Jupiter"),
     ("What is the smallest planet?", "Mercury"),
     ("What planet is known as the Red Planet?", "Mars")],
    [("What is H2O?", "water"),
     ("What gas do plants breathe in?", "carbon dioxide"),
     ("What gas do plants release?", "oxygen")],
    [("Who wrote Hamlet?", "William Shakespeare"),
     ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
     ("Who invented the light bulb?", "Thomas Edison")],
    [("What is the tallest mountain?", "Mount Everest"),
     ("What is the longest river?", "Nile"),
     ("What is the largest ocean?", "Pacific")],
    [("What is the currency of Japan?", "yen"),
     ("What is the currency of UK?", "pound"),
     ("What is the currency of USA?", "dollar")],
]

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def test_batch(model, tokenizer, batch):
    return [a.lower() in generate_answer(model, tokenizer, q).lower() for q, a in batch]

def train_lora_fp32(model, tokenizer, facts, seed, rehearsal_facts=None):
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
    if rehearsal_facts:
        texts += [f"{q} {a}" for q, a in rehearsal_facts]
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
print("M279 — Long Chain: 10 Sequential Batches with Rehearsal")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

all_facts = []
results = []

for i, batch in enumerate(BATCHES):
    print(f"\n[Batch {i+1}/10] Training {len(batch)} new fact(s)...")
    all_facts.extend(batch)
    rehearsal = all_facts[:-len(batch)] if len(all_facts) > len(batch) else None
    model = train_lora_fp32(model, tokenizer, batch, SEED + i, rehearsal_facts=rehearsal)
    
    # Test current batch
    current_survival = test_batch(model, tokenizer, batch)
    print(f"  Current batch: {sum(current_survival)}/{len(current_survival)}")
    
    # Test all previous batches
    if i > 0:
        prev_survival = []
        for j in range(i):
            prev = test_batch(model, tokenizer, BATCHES[j])
            prev_survival.extend(prev)
        print(f"  Previous batches: {sum(prev_survival)}/{len(prev_survival)}")
    else:
        prev_survival = []
    
    results.append({
        "batch": i + 1,
        "current": sum(current_survival),
        "current_total": len(current_survival),
        "previous": sum(prev_survival) if prev_survival else 0,
        "previous_total": len(prev_survival) if prev_survival else 0,
    })

# Final test: all 30 facts
print("\n[Final] Testing all 30 facts...")
final_survival = []
for batch in BATCHES:
    final_survival.extend(test_batch(model, tokenizer, batch))

print(f"  Total survival: {sum(final_survival)}/{len(final_survival)} = {sum(final_survival)/len(final_survival):.1%}")

# Check forgetting
forgetting_detected = any(
    r["previous"] < r["previous_total"] for r in results if r["previous_total"] > 0
)

print("\n" + "=" * 60)
if not forgetting_detected:
    print(f"🎯 NO FORGETTING across 10 batches!")
else:
    print(f"⚠️  Forgetting detected in some batches")
print(f"Final survival: {sum(final_survival)}/{len(final_survival)} = {sum(final_survival)/len(final_survival):.1%}")
print("=" * 60)

results_data = {
    "batches": results,
    "final_survival": sum(final_survival),
    "final_total": len(final_survival),
    "forgetting_detected": forgetting_detected,
}
with open("experiments/m279_results.json", "w") as f:
    json.dump(results_data, f, indent=2)
print("✅ Saved to experiments/m279_results.json")

del model
torch.cuda.empty_cache()
