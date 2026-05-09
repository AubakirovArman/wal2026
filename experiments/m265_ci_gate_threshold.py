"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M265 — CI Gate Threshold Calibration

Hypothesis: We can determine optimal thresholds for CI gates
by measuring distribution of scores across passing and failing edits.
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
STEPS = 100
LR = 5e-5
TARGET_LAYERS = [16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

# Facts that should pass
GOOD_FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is 2+2?", "4"),
]

# Facts designed to fail (nonsense)
BAD_FACTS = [
    ("What is the capital of France?", "London"),
    ("What is the capital of Japan?", "Beijing"),
    ("What is 2+2?", "5"),
]

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def compute_ppl(model, tokenizer, text):
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(enc.input_ids, labels=enc.input_ids)
    return math.exp(out.loss.item())

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

def evaluate_edit(model, tokenizer, facts, negatives, baseline_ppl):
    """Run full CI evaluation on an edited model."""
    exact = sum(1 for q, a in facts if a.lower() in generate_answer(model, tokenizer, q).lower())
    neg = sum(1 for q, a in negatives if a.lower() not in generate_answer(model, tokenizer, q).lower())
    ppl = compute_ppl(model, tokenizer, "The quick brown fox jumps over the lazy dog.")
    ppl_ok = ppl < baseline_ppl * 1.5
    return {
        "exact": exact / len(facts),
        "negative": neg / len(negatives),
        "ppl": ppl,
        "ppl_ok": ppl_ok,
    }

print("=" * 60)
print("M265 — CI Gate Threshold Calibration")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Baseline PPL
print("\n[Baseline] Measuring baseline PPL...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
baseline_ppl = compute_ppl(base_model, tokenizer, "The quick brown fox jumps over the lazy dog.")
print(f"  Baseline PPL: {baseline_ppl:.3f}")
del base_model
torch.cuda.empty_cache()

# Test GOOD edits
print("\n[Good Edits] Testing facts that should pass...")
good_scores = []
for i, fact in enumerate(GOOD_FACTS):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model = train_lora_fp32(model, tokenizer, [fact], SEED)
    score = evaluate_edit(model, tokenizer, [fact], [("What is 2+2?", "5")], baseline_ppl)
    good_scores.append(score)
    print(f"  Fact {i+1}: exact={score['exact']:.1f}, neg={score['negative']:.1f}, ppl={score['ppl']:.2f}")
    del model
    torch.cuda.empty_cache()

# Test BAD edits
print("\n[Bad Edits] Testing facts that should fail...")
bad_scores = []
for i, fact in enumerate(BAD_FACTS):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model = train_lora_fp32(model, tokenizer, [fact], SEED)
    score = evaluate_edit(model, tokenizer, [fact], [("What is 2+2?", "5")], baseline_ppl)
    bad_scores.append(score)
    print(f"  Fact {i+1}: exact={score['exact']:.1f}, neg={score['negative']:.1f}, ppl={score['ppl']:.2f}")
    del model
    torch.cuda.empty_cache()

# Compute optimal thresholds
print(f"\n{'='*60}")
print("THRESHOLD ANALYSIS")
print(f"{'='*60}")

for metric in ["exact", "negative", "ppl"]:
    good_vals = [s[metric] for s in good_scores]
    bad_vals = [s[metric] for s in bad_scores]
    if metric == "ppl":
        # Lower is better
        threshold = (max(good_vals) + min(bad_vals)) / 2
        good_pass = sum(1 for v in good_vals if v < threshold)
        bad_pass = sum(1 for v in bad_vals if v < threshold)
    else:
        # Higher is better
        threshold = (min(good_vals) + max(bad_vals)) / 2
        good_pass = sum(1 for v in good_vals if v >= threshold)
        bad_pass = sum(1 for v in bad_vals if v >= threshold)
    
    print(f"\n  {metric.upper()}:")
    print(f"    Good range: [{min(good_vals):.3f}, {max(good_vals):.3f}]")
    print(f"    Bad range:  [{min(bad_vals):.3f}, {max(bad_vals):.3f}]")
    print(f"    Suggested threshold: {threshold:.3f}")
    print(f"    Good pass: {good_pass}/{len(good_vals)}, Bad pass: {bad_pass}/{len(bad_vals)}")

print(f"\n{'='*60}")
print("🎯 THRESHOLDS CALIBRATED")
print("="*60)

results = {
    "baseline_ppl": baseline_ppl,
    "good_scores": good_scores,
    "bad_scores": bad_scores,
}
with open("experiments/m265_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m265_results.json")
