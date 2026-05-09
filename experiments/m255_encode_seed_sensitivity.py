"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M255 — Encode Seed Sensitivity Audit

Hypothesis: Different encode seeds produce different but equally valid 
compressed models. We verify that fixed-seed encode is deterministic
and different seeds produce different (but stable) weights.
"""
import os, sys, json, torch, math
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:0"
MODEL_ID = "meta-llama/Llama-3.1-8B"
SEEDS = [42, 43, 44, 45, 46]
K = 256
ITERS = 3

def hadamard_transform_2d(w):
    n = w.shape[-1]
    m = 1 << (n - 1).bit_length()
    orig_info = (n, m)
    if m != n:
        pad = torch.zeros(w.shape[0], m - n, device=w.device, dtype=w.dtype)
        w = torch.cat([w, pad], dim=-1)
    h = w.clone()
    step = 1
    while step < m:
        idx1 = torch.arange(0, m, 2 * step, device=w.device)
        idx2 = idx1 + step
        top = h[:, idx1]
        bot = h[:, idx2]
        h[:, idx1] = top + bot
        h[:, idx2] = top - bot
        step *= 2
    return h, orig_info

def inverse_hadamard_2d(h, orig_info):
    n, m = orig_info
    w = h.clone()
    step = m // 2
    while step >= 1:
        idx1 = torch.arange(0, m, 2 * step, device=h.device)
        idx2 = idx1 + step
        top = w[:, idx1]
        bot = w[:, idx2]
        w[:, idx1] = (top + bot) / 2
        w[:, idx2] = (top - bot) / 2
        step //= 2
    if m != n:
        w = w[:, :n]
    return w

def kmeans_chunked(data, K, iters=3, chunk_size=1_000_000, seed=42):
    torch.manual_seed(seed)
    device = data.device
    sample = data[torch.randperm(data.numel(), device=device)[:min(100000, data.numel())]]
    atoms = [sample[torch.randint(0, len(sample), (1,), device=device)].item()]
    for _ in range(1, K):
        dists = torch.stack([torch.abs(sample - a) for a in atoms], dim=0).min(dim=0).values
        probs = dists / (dists.sum() + 1e-10)
        idx = torch.multinomial(probs, 1)
        atoms.append(sample[idx].item())
    atoms = torch.tensor(atoms, device=device, dtype=data.dtype)
    for _ in range(iters):
        new_sums = torch.zeros(K, device=device, dtype=torch.float64)
        counts = torch.zeros(K, device=device, dtype=torch.float64)
        for i in range(0, data.numel(), chunk_size):
            chunk = data[i:i+chunk_size]
            dists = torch.abs(chunk.unsqueeze(1) - atoms.unsqueeze(0))
            labels = dists.argmin(dim=1)
            new_sums.scatter_add_(0, labels, chunk.double())
            counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
        new_atoms = torch.where(counts > 0, (new_sums / counts).to(data.dtype), atoms)
        if torch.allclose(atoms, new_atoms, atol=1e-6):
            break
        atoms = new_atoms
    labels_all = []
    for i in range(0, data.numel(), chunk_size):
        chunk = data[i:i+chunk_size]
        dists = torch.abs(chunk.unsqueeze(1) - atoms.unsqueeze(0))
        labels_all.append(dists.argmin(dim=1))
    labels = torch.cat(labels_all)
    quantized = atoms[labels].reshape(data.shape)
    return quantized, atoms, labels

def hadamard_wal_encode(w, K=256, iters=3, seed=42):
    torch.manual_seed(seed)
    h, orig_info = hadamard_transform_2d(w)
    quantized, atoms, indices = kmeans_chunked(h.reshape(-1), K, iters=iters, seed=seed)
    quantized = quantized.reshape(h.shape)
    w_rec = inverse_hadamard_2d(quantized, orig_info)
    return w_rec

def encode_model(model, K=256, iters=3, seed=42):
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        if 'weight' in name and len(p.shape) == 2 and 100 < p.shape[0] < 50000:
            p.data = hadamard_wal_encode(p.data, K=K, iters=iters, seed=seed)
    return model

def get_weights_fingerprint(model, layer_idx=16):
    state = {}
    for name, p in model.named_parameters():
        if f"layers.{layer_idx}" in name and p.ndim >= 2:
            state[name] = p.detach().cpu().float()
    return state

def fingerprint_hash(state):
    import hashlib
    h = hashlib.sha256()
    for name in sorted(state.keys()):
        h.update(name.encode())
        h.update(state[name].numpy().tobytes())
    return h.hexdigest()[:16]

print("=" * 60)
print("M255 — Encode Seed Sensitivity Audit")
print("=" * 60)

hashes = []

for seed in SEEDS:
    print(f"\n[Seed {seed}] Encoding...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    encode_model(model, K=K, iters=ITERS, seed=seed)
    
    fp = get_weights_fingerprint(model)
    h = fingerprint_hash(fp)
    hashes.append({"seed": seed, "hash": h})
    print(f"  Fingerprint: {h}")
    
    del model
    torch.cuda.empty_cache()

# Check determinism
print("\n" + "=" * 60)
print("DETERMINISM CHECK")
print("=" * 60)

print("\n[Seed 42 × 2] Verifying self-consistency...")
hashes_42 = []
for _ in range(2):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE, low_cpu_mem_usage=True
    )
    encode_model(model, K=K, iters=ITERS, seed=42)
    fp = get_weights_fingerprint(model)
    h = fingerprint_hash(fp)
    hashes_42.append(h)
    del model
    torch.cuda.empty_cache()

self_consistent = hashes_42[0] == hashes_42[1]
print(f"  Run 1: {hashes_42[0]}")
print(f"  Run 2: {hashes_42[1]}")
print(f"  Self-consistent: {'✅ YES' if self_consistent else '❌ NO'}")

# Check all unique
all_hashes = [h["hash"] for h in hashes]
unique_hashes = set(all_hashes)
print(f"\n[Seed Diversity] {len(unique_hashes)}/{len(all_hashes)} unique hashes")
for h in hashes:
    marker = "✅" if all_hashes.count(h["hash"]) == 1 else "❌"
    print(f"  {marker} Seed {h['seed']}: {h['hash']}")

print("\n" + "=" * 60)
if self_consistent and len(unique_hashes) == len(all_hashes):
    print("🎯 SEED SENSITIVITY: Deterministic AND diverse")
elif self_consistent:
    print("✅ SEED SENSITIVITY: Deterministic, partial diversity")
else:
    print("❌ SEED SENSITIVITY: Not deterministic")
print("=" * 60)

results = {
    "self_consistent": self_consistent,
    "unique_hashes": len(unique_hashes),
    "total_hashes": len(all_hashes),
    "hashes": hashes,
    "hashes_42": hashes_42,
}
with open("experiments/m255_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m255_results.json")
