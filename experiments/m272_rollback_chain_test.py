"""
M272 — Rollback Chain Test

Hypothesis: We can build a chain v0→v1→v2→v3, rollback to v1,
and then rebuild v3 again with identical results.
"""
import os, sys, json, torch, random, time
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

FACTS = [
    [("What is the capital of France?", "Paris")],
    [("What is the capital of Japan?", "Tokyo")],
    [("What is the capital of Italy?", "Rome")],
    [("What is the capital of Spain?", "Madrid")],
]

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def get_weights_snapshot(model, layer_idx=16):
    state = {}
    for name, p in model.named_parameters():
        if f"layers.{layer_idx}" in name and p.ndim >= 2:
            state[name] = p.detach().cpu().float().clone()
    return state

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
print("M272 — Rollback Chain Test")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Build v0 (base)
print("\n[v0] Base model...")
model_v0 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
state_v0 = get_weights_snapshot(model_v0)
print(f"  Base hash computed")

# Build v1 (v0 + France)
print("\n[v1] v0 + France...")
model_v1 = train_lora_fp32(model_v0, tokenizer, FACTS[0], SEED)
state_v1 = get_weights_snapshot(model_v1)
survival_v1 = [generate_answer(model_v1, tokenizer, q) for q, a in FACTS[0]]
print(f"  Survival: {survival_v1}")

# Build v2 (v1 + Japan, with rehearsal)
print("\n[v2] v1 + Japan (with rehearsal)...")
model_v2 = train_lora_fp32(model_v1, tokenizer, FACTS[1], SEED, rehearsal_facts=FACTS[0])
state_v2 = get_weights_snapshot(model_v2)
survival_v2 = [generate_answer(model_v2, tokenizer, q) for q, a in (FACTS[0]+FACTS[1])]
print(f"  Survival: {survival_v2}")

# Build v3 (v2 + Italy, with rehearsal)
print("\n[v3] v2 + Italy (with rehearsal)...")
model_v3 = train_lora_fp32(model_v2, tokenizer, FACTS[2], SEED, rehearsal_facts=FACTS[0]+FACTS[1])
state_v3 = get_weights_snapshot(model_v3)
survival_v3 = [generate_answer(model_v3, tokenizer, q) for q, a in (FACTS[0]+FACTS[1]+FACTS[2])]
print(f"  Survival: {survival_v3}")

# Save v3
ckpt_v3 = "/mnt/hf_model_weights/arman/3bit/wal/experiments/m272_v3.pt"
torch.save(model_v3.state_dict(), ckpt_v3)

# Compute deltas
delta_v1 = {k: (state_v1[k] - state_v0[k]).to(DEVICE, torch.bfloat16) for k in state_v0}
delta_v2 = {k: (state_v2[k] - state_v1[k]).to(DEVICE, torch.bfloat16) for k in state_v1}
delta_v3 = {k: (state_v3[k] - state_v2[k]).to(DEVICE, torch.bfloat16) for k in state_v2}

del model_v0, model_v1, model_v2, model_v3
torch.cuda.empty_cache()

# Rollback to v1: reload base + apply delta_v1
print("\n[Rollback to v1] Reload base + apply v1 delta...")
t0 = time.time()
model_rb = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
for name, p in model_rb.named_parameters():
    if name in delta_v1:
        p.data += delta_v1[name]
rollback_time = time.time() - t0
state_rb = get_weights_snapshot(model_rb)
max_diff_rb = max((state_rb[k] - state_v1[k]).abs().max().item() for k in state_v1)
print(f"  Rollback time: {rollback_time:.2f}s")
print(f"  Accuracy vs v1: {max_diff_rb:.6e}")

del model_rb
torch.cuda.empty_cache()

# Rebuild v3 from v1: apply v2_delta + v3_delta
print("\n[Rebuild v3 from v1] Apply v2 + v3 deltas...")
t0 = time.time()
model_rebuild = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
for name, p in model_rebuild.named_parameters():
    if name in delta_v1:
        p.data += delta_v1[name]
    if name in delta_v2:
        p.data += delta_v2[name]
    if name in delta_v3:
        p.data += delta_v3[name]
rebuild_time = time.time() - t0
state_rebuild = get_weights_snapshot(model_rebuild)
max_diff_rebuild = max((state_rebuild[k] - state_v3[k]).abs().max().item() for k in state_v3)
print(f"  Rebuild time: {rebuild_time:.2f}s")
print(f"  Accuracy vs v3: {max_diff_rebuild:.6e}")

# Verify survival
survival_rebuild = [generate_answer(model_rebuild, tokenizer, q) for q, a in (FACTS[0]+FACTS[1]+FACTS[2])]
print(f"  Survival: {survival_rebuild}")

del model_rebuild
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Rollback to v1:   {rollback_time:.2f}s, accuracy {max_diff_rb:.6e}")
print(f"Rebuild v3:       {rebuild_time:.2f}s, accuracy {max_diff_rebuild:.6e}")
print(f"v3 survival:      {survival_v3}")
print(f"rebuild survival: {survival_rebuild}")

if max_diff_rb < 1e-5 and max_diff_rebuild < 1e-5:
    print("\n🎯 ROLLBACK CHAIN WORKS")
else:
    print("\n⚠️  Precision loss detected")
print("=" * 60)

results = {
    "rollback_time": rollback_time,
    "rebuild_time": rebuild_time,
    "rollback_accuracy": max_diff_rb,
    "rebuild_accuracy": max_diff_rebuild,
    "v3_survival": survival_v3,
    "rebuild_survival": survival_rebuild,
}
with open("experiments/m272_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m272_results.json")
