"""
M261 — Anti-Forgetting Rehearsal Test

Hypothesis: Rehearsing previously learned facts during new training
prevents catastrophic forgetting.
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
STEPS = 100
LR = 5e-5
TARGET_LAYERS = [16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

BATCH_1 = [("What is the capital of France?", "Paris")]
BATCH_2 = [("What is the capital of Japan?", "Tokyo")]
BATCH_3 = [("What is the capital of Italy?", "Rome")]

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def test_facts(model, tokenizer, facts):
    return [target.lower() in generate_answer(model, tokenizer, q).lower() for q, target in facts]

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
    
    # Build training texts
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
print("M261 — Anti-Forgetting Rehearsal Test")
print("=" * 60)

# Test WITHOUT rehearsal
print("\n[Without Rehearsal] Sequential training, no rehearsal...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = train_lora_fp32(model, tokenizer, BATCH_1, SEED)
s1 = test_facts(model, tokenizer, BATCH_1)
print(f"  After Batch 1: France={s1[0]}")

model = train_lora_fp32(model, tokenizer, BATCH_2, SEED)
s1_after_2 = test_facts(model, tokenizer, BATCH_1)
s2 = test_facts(model, tokenizer, BATCH_2)
print(f"  After Batch 2: France={s1_after_2[0]}, Japan={s2[0]}")

model = train_lora_fp32(model, tokenizer, BATCH_3, SEED)
s1_after_3 = test_facts(model, tokenizer, BATCH_1)
s2_after_3 = test_facts(model, tokenizer, BATCH_2)
s3 = test_facts(model, tokenizer, BATCH_3)
print(f"  After Batch 3: France={s1_after_3[0]}, Japan={s2_after_3[0]}, Italy={s3[0]}")

without_rehearsal = {
    "batch1": s1, "batch2": s2, "batch3": s3,
    "forgetting_b2": s1 != s1_after_2,
    "forgetting_b3": s1_after_2 != s1_after_3 or s2 != s2_after_3,
}

del model
torch.cuda.empty_cache()

# Test WITH rehearsal
print("\n[With Rehearsal] Rehearsing all previous facts...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)

model = train_lora_fp32(model, tokenizer, BATCH_1, SEED)
s1_r = test_facts(model, tokenizer, BATCH_1)
print(f"  After Batch 1: France={s1_r[0]}")

model = train_lora_fp32(model, tokenizer, BATCH_2, SEED, rehearsal_facts=BATCH_1)
s1_r_after_2 = test_facts(model, tokenizer, BATCH_1)
s2_r = test_facts(model, tokenizer, BATCH_2)
print(f"  After Batch 2: France={s1_r_after_2[0]}, Japan={s2_r[0]}")

model = train_lora_fp32(model, tokenizer, BATCH_3, SEED, rehearsal_facts=BATCH_1+BATCH_2)
s1_r_after_3 = test_facts(model, tokenizer, BATCH_1)
s2_r_after_3 = test_facts(model, tokenizer, BATCH_2)
s3_r = test_facts(model, tokenizer, BATCH_3)
print(f"  After Batch 3: France={s1_r_after_3[0]}, Japan={s2_r_after_3[0]}, Italy={s3_r[0]}")

with_rehearsal = {
    "batch1": s1_r, "batch2": s2_r, "batch3": s3_r,
    "forgetting_b2": s1_r != s1_r_after_2,
    "forgetting_b3": s1_r_after_2 != s1_r_after_3 or s2_r != s2_r_after_3,
}

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Without rehearsal:")
print(f"  Batch 1 survival: {s1[0]} → {s1_after_2[0]} → {s1_after_3[0]}")
print(f"  Batch 2 survival: — → {s2[0]} → {s2_after_3[0]}")
print(f"  Forgetting detected: B2={without_rehearsal['forgetting_b2']}, B3={without_rehearsal['forgetting_b3']}")
print(f"\nWith rehearsal:")
print(f"  Batch 1 survival: {s1_r[0]} → {s1_r_after_2[0]} → {s1_r_after_3[0]}")
print(f"  Batch 2 survival: — → {s2_r[0]} → {s2_r_after_3[0]}")
print(f"  Forgetting detected: B2={with_rehearsal['forgetting_b2']}, B3={with_rehearsal['forgetting_b3']}")

if not with_rehearsal['forgetting_b2'] and not with_rehearsal['forgetting_b3']:
    print("\n🎯 REHEARSAL PREVENTS FORGETTING")
else:
    print("\n⚠️  Rehearsal partially effective")
print("="*60)

results = {
    "without_rehearsal": without_rehearsal,
    "with_rehearsal": with_rehearsal,
}
with open("experiments/m261_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m261_results.json")
