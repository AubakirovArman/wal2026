"""
M270 — CI Report Schema: Unified JSON for Every Edit

Hypothesis: Every edit should produce a standardized CI report
that can be compared across builds, versions, and branches.
"""
import os, sys, json, torch, random, math, time
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

FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
]

NEGATIVES = [
    ("What is the capital of Germany?", "Paris"),
    ("What is 2+2?", "5"),
    ("Who invented the telephone?", "Nikola Tesla"),
]

PARAPHRASES = [
    ("Paris is the capital of which country?", "France"),
    ("Tokyo belongs to which nation?", "Japan"),
    ("Rome is in what country?", "Italy"),
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

def run_ci(model, tokenizer, facts, negatives, paraphrases):
    """Run full CI and return unified report."""
    report = {
        "schema_version": "1.0",
        "timestamp": time.time(),
        "tests": {},
        "metrics": {},
        "verdict": None,
    }
    
    # Exact match
    exact = []
    for q, a in facts:
        ans = generate_answer(model, tokenizer, q)
        ok = a.lower() in ans.lower()
        exact.append({"question": q, "expected": a, "got": ans, "pass": ok})
    report["tests"]["exact_match"] = {
        "pass": sum(1 for e in exact if e["pass"]),
        "total": len(exact),
        "details": exact,
    }
    
    # Paraphrase
    para = []
    for q, a in paraphrases:
        ans = generate_answer(model, tokenizer, q)
        ok = a.lower() in ans.lower()
        para.append({"question": q, "expected": a, "got": ans, "pass": ok})
    report["tests"]["paraphrase"] = {
        "pass": sum(1 for e in para if e["pass"]),
        "total": len(para),
        "details": para,
    }
    
    # Negative
    neg = []
    for q, wrong_a in negatives:
        ans = generate_answer(model, tokenizer, q)
        ok = wrong_a.lower() not in ans.lower()
        neg.append({"question": q, "forbidden": wrong_a, "got": ans, "pass": ok})
    report["tests"]["negative"] = {
        "pass": sum(1 for e in neg if e["pass"]),
        "total": len(neg),
        "details": neg,
    }
    
    # PPL
    ppl = compute_ppl(model, tokenizer)
    report["tests"]["ppl"] = {
        "value": ppl,
        "threshold": 3.0,
        "pass": ppl < 3.0,
    }
    
    # NaN
    has_nan = any(torch.isnan(p).any() for p in model.parameters())
    report["tests"]["no_nan"] = {"pass": not has_nan}
    
    # Compute metrics
    exact_rate = report["tests"]["exact_match"]["pass"] / len(facts)
    para_rate = report["tests"]["paraphrase"]["pass"] / len(paraphrases)
    neg_rate = report["tests"]["negative"]["pass"] / len(negatives)
    ppl_gate = 1.0 if report["tests"]["ppl"]["pass"] else 0.0
    nan_gate = 1.0 if report["tests"]["no_nan"]["pass"] else 0.0
    
    report["metrics"] = {
        "exact_rate": exact_rate,
        "paraphrase_rate": para_rate,
        "negative_rate": neg_rate,
        "ppl": ppl,
        "has_nan": has_nan,
        "ci_score": exact_rate * 0.25 + para_rate * 0.25 + neg_rate * 0.30 + ppl_gate * 0.10 + nan_gate * 0.10,
    }
    report["metrics"]["nls"] = 1.0 - neg_rate  # Negative Leakage Score
    report["metrics"]["crs"] = para_rate / max(exact_rate, 0.01)  # Context Robustness Score
    
    # Verdict
    report["verdict"] = "PASS" if report["metrics"]["ci_score"] >= 0.7 else "FAIL"
    
    return report

print("=" * 60)
print("M270 — CI Report Schema")
print("=" * 60)

# Baseline test
print("\n[Baseline] Testing unedited model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
baseline_report = run_ci(model, tokenizer, FACTS, NEGATIVES, PARAPHRASES)
print(f"  Baseline CI Score: {baseline_report['metrics']['ci_score']:.2f}")
print(f"  Verdict: {baseline_report['verdict']}")
del model
torch.cuda.empty_cache()

# Edit test
print("\n[Edited] Testing model with 3 facts...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model = train_lora_fp32(model, tokenizer, FACTS, SEED)
edited_report = run_ci(model, tokenizer, FACTS, NEGATIVES, PARAPHRASES)
print(f"  Edited CI Score: {edited_report['metrics']['ci_score']:.2f}")
print(f"  Verdict: {edited_report['verdict']}")

# Save reports
os.makedirs("experiments/m270_output", exist_ok=True)
with open("experiments/m270_output/baseline_report.json", "w") as f:
    json.dump(baseline_report, f, indent=2)
with open("experiments/m270_output/edited_report.json", "w") as f:
    json.dump(edited_report, f, indent=2)

# Compare
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
for test in ["exact_match", "paraphrase", "negative", "ppl", "no_nan"]:
    b = baseline_report["tests"][test]
    e = edited_report["tests"][test]
    if test == "ppl":
        print(f"  {test:12s}: baseline={b['value']:.2f} edited={e['value']:.2f}")
    else:
        b_pass = b.get("pass", b.get("pass"))
        e_pass = e.get("pass", e.get("pass"))
        print(f"  {test:12s}: baseline={b_pass} edited={e_pass}")

print(f"\n  CI Score: {baseline_report['metrics']['ci_score']:.2f} → {edited_report['metrics']['ci_score']:.2f}")
print(f"  NLS:      {baseline_report['metrics']['nls']:.2f} → {edited_report['metrics']['nls']:.2f}")
print(f"  CRS:      {baseline_report['metrics']['crs']:.2f} → {edited_report['metrics']['crs']:.2f}")
print(f"\n✅ Reports saved to experiments/m270_output/")
print("=" * 60)

del model
torch.cuda.empty_cache()
