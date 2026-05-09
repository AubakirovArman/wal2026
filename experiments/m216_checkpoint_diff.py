"""
M216 — WAL Checkpoint Diff After Compiled Edit

Compare WAL_v0 vs WAL_v1 vs WAL_v2 to understand:
1. What % of programs changed
2. Which modules changed most
3. Atom/coeff distribution shifts
4. Binary diff size between versions
"""

import os, sys, json, torch, random, gc, math, time
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
RANK = 4
STEPS = 100
LR = 5e-5
K = 256
ITERS = 3
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]
N_EDITS = 3
FACTS_PER_EDIT = 5

from experiments.facts_50 import FACTS_50

# ── Encode helpers ───────────────────────────────────────
def hadamard_transform_2d(w):
    n = w.shape[-1]
    m = 1 << (n - 1).bit_length()
    orig_info = (n, m)
    if m != n:
        pad = torch.zeros(w.shape[0], m - n, device=w.device, dtype=w.dtype)
        w = torch.cat([w, pad], dim=-1)
    h = 1
    while h < m:
        w = w.reshape(w.shape[0], m // (2 * h), 2, h)
        w = torch.cat([w[:, :, 0, :] + w[:, :, 1, :], w[:, :, 0, :] - w[:, :, 1, :]], dim=-1)
        h *= 2
    return w.reshape(w.shape[0], m) / math.sqrt(m), orig_info

def inverse_hadamard_2d(h, orig_info=None):
    n = h.shape[-1]
    m = 1 << (n - 1).bit_length()
    if m != n:
        pad = torch.zeros(h.shape[0], m - n, device=h.device, dtype=h.dtype)
        h = torch.cat([h, pad], dim=-1)
    hh = 1
    while hh < m:
        h = h.reshape(h.shape[0], m // (2 * hh), 2, hh)
        h = torch.cat([h[:, :, 0, :] + h[:, :, 1, :], h[:, :, 0, :] - h[:, :, 1, :]], dim=-1)
        hh *= 2
    result = h.reshape(h.shape[0], m) / math.sqrt(m)
    if orig_info is not None:
        n_orig, _ = orig_info
        result = result[:, :n_orig]
    return result

def kmeans_chunked(data, K, iters=3, chunk_size=1_000_000):
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

def hadamard_wal_encode(w, K, iters=3):
    orig_shape = w.shape
    h, orig_info = hadamard_transform_2d(w.float().cuda())
    quantized, atoms, indices = kmeans_chunked(h.reshape(-1), K, iters=iters)
    quantized = quantized.reshape(h.shape)
    w_rec = inverse_hadamard_2d(quantized, orig_info)
    return w_rec[:, :orig_shape[1]].to(w.device, w.dtype)

def encode_model(model, K=256, iters=3):
    for name, p in model.named_parameters():
        if 'weight' in name and len(p.shape) == 2 and 100 < p.shape[0] < 50000:
            p.data = hadamard_wal_encode(p.data, K=K, iters=iters)
    return model

# ── LoRA helpers ─────────────────────────────────────────
class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(in_features, rank, device="cuda", dtype=torch.bfloat16))
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, out_features, device="cuda", dtype=torch.bfloat16))
        self.scaling = 1.0
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B)
    def forward(self, x):
        return x @ self.lora_A @ self.lora_B * self.scaling

def inject_lora(model, target_layers, target_modules, rank=4):
    trainable = []
    for name, module in model.named_modules():
        if not any(f'.{layer}.' in name for layer in target_layers):
            continue
        if not any(name.endswith(m) for m in target_modules):
            continue
        if not hasattr(module, 'weight'):
            continue
        lora = LoRALayer(module.in_features, module.out_features, rank).to(module.weight.device, module.weight.dtype)
        module.lora = lora
        module._orig_forward = module.forward
        def make_forward(orig, lora_layer):
            def forward(x):
                return orig(x) + lora_layer(x)
            return forward
        module.forward = make_forward(module._orig_forward, lora)
        for p in lora.parameters():
            trainable.append(p)
    return model, trainable

def merge_lora(model):
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            W_f32 = module.weight.data.float()
            A_f32 = module.lora.lora_A.float()
            B_f32 = module.lora.lora_B.float()
            delta_f32 = (A_f32 @ B_f32).T
            W_merged_f32 = W_f32 + delta_f32
            module.weight.data = W_merged_f32.to(module.weight.dtype)
            module.forward = module._orig_forward
            del module.lora
            del module._orig_forward
    return model

def get_wikitext_samples(tokenizer, n=50):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [ex for ex in ds['text'][:200] if len(ex) > 20][:n]

def train_lora(model, tokenizer, facts_group, steps, rank, target_layers, target_modules, lr, device):
    model, trainable = inject_lora(model, target_layers, target_modules, rank)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    wiki_texts = get_wikitext_samples(tokenizer)
    model.train()
    for step in range(steps):
        if random.random() < 0.5:
            q, a = random.choice(facts_group)
            text = f"Question: {q}\nAnswer: {a}"
        else:
            text = random.choice(wiki_texts)
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = toks.input_ids.to(device)
        attention_mask = toks.attention_mask.to(device)
        out = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

# ── Checkpoint snapshot ──────────────────────────────────
def snapshot_checkpoint(model):
    """Extract flat weight snapshot for diff comparison."""
    snapshot = {}
    for name, p in model.named_parameters():
        if 'weight' in name and len(p.shape) == 2 and 100 < p.shape[0] < 50000:
            snapshot[name] = p.data.detach().cpu().clone()
    return snapshot

def compare_checkpoints(snap1, snap2):
    """Compare two checkpoints and return diff statistics."""
    stats = {
        "total_modules": 0,
        "changed_modules": 0,
        "mean_relative_change": [],
        "max_relative_change": [],
        "changed_modules_list": [],
    }
    
    for name in snap1:
        if name not in snap2:
            continue
        w1 = snap1[name].float()
        w2 = snap2[name].float()
        
        diff = (w2 - w1).abs()
        denom = w1.abs() + 1e-8
        rel_change = (diff / denom).mean().item()
        max_rel = (diff / denom).max().item()
        
        stats["total_modules"] += 1
        if rel_change > 0.001:  # >0.1% relative change
            stats["changed_modules"] += 1
            stats["changed_modules_list"].append(name)
        
        stats["mean_relative_change"].append(rel_change)
        stats["max_relative_change"].append(max_rel)
    
    stats["mean_relative_change"] = np.mean(stats["mean_relative_change"])
    stats["max_relative_change"] = np.max(stats["max_relative_change"])
    stats["change_fraction"] = stats["changed_modules"] / max(stats["total_modules"], 1)
    return stats

def estimate_binary_diff_size(snap1, snap2):
    """Estimate compressed diff size between checkpoints."""
    total_params = 0
    changed_params = 0
    total_bytes = 0
    
    for name in snap1:
        if name not in snap2:
            continue
        w1 = snap1[name]
        w2 = snap2[name]
        
        # Count changed elements
        diff_mask = (w1 != w2)
        changed = diff_mask.sum().item()
        total = w1.numel()
        
        total_params += total
        changed_params += changed
        
        # Rough estimate: store (index, new_value) for changed elements
        # index = 4 bytes (int32), value = 2 bytes (bf16)
        total_bytes += changed * 6
    
    return {
        "total_params": total_params,
        "changed_params": changed_params,
        "change_ratio": changed_params / max(total_params, 1),
        "estimated_diff_bytes": total_bytes,
        "estimated_diff_mb": total_bytes / (1024 * 1024),
    }

# ── Main ─────────────────────────────────────────────────
def main():
    print("=" * 60, flush=True)
    print("M216 — WAL Checkpoint Diff", flush=True)
    print("=" * 60, flush=True)
    
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare edit batches
    all_facts = FACTS_50[:N_EDITS * FACTS_PER_EDIT]
    batches = [all_facts[i*FACTS_PER_EDIT:(i+1)*FACTS_PER_EDIT] for i in range(N_EDITS)]
    
    # Load model and encode
    print(f"Loading {MODEL_ID}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model = model.to(device)
    
    print("Encoding base model...", flush=True)
    model = encode_model(model, K=K, iters=ITERS)
    
    # Snapshot v0
    print("Snapshot v0...", flush=True)
    snap_v0 = snapshot_checkpoint(model)
    
    # Sequential edits with snapshots
    snapshots = [snap_v0]
    edit_results = []
    
    for edit_idx in range(N_EDITS):
        batch = batches[edit_idx]
        print(f"\nEdit {edit_idx+1}/{N_EDITS}: {len(batch)} facts", flush=True)
        
        # Train
        model = train_lora(model, tokenizer, batch, steps=STEPS, rank=RANK,
                          target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES,
                          lr=LR, device=device)
        
        # Merge
        model = merge_lora(model)
        
        # Re-encode
        model = encode_model(model, K=K, iters=ITERS)
        
        # Snapshot
        snap = snapshot_checkpoint(model)
        snapshots.append(snap)
        
        # Compare with previous
        diff = compare_checkpoints(snapshots[-2], snap)
        binary = estimate_binary_diff_size(snapshots[-2], snap)
        
        print(f"  Changed modules: {diff['changed_modules']}/{diff['total_modules']} ({diff['change_fraction']*100:.1f}%)", flush=True)
        print(f"  Mean rel change: {diff['mean_relative_change']*100:.3f}%", flush=True)
        print(f"  Max rel change: {diff['max_relative_change']*100:.2f}%", flush=True)
        print(f"  Changed params: {binary['changed_params']:,} / {binary['total_params']:,} ({binary['change_ratio']*100:.3f}%)", flush=True)
        print(f"  Est. diff size: {binary['estimated_diff_mb']:.2f} MB", flush=True)
        
        edit_results.append({
            "edit": edit_idx + 1,
            "diff": diff,
            "binary": binary,
        })
    
    # Cumulative diff (v0 → v3)
    print(f"\n{'='*60}", flush=True)
    print("CUMULATIVE DIFF: v0 → v3", flush=True)
    print(f"{'='*60}", flush=True)
    cumul_diff = compare_checkpoints(snapshots[0], snapshots[-1])
    cumul_binary = estimate_binary_diff_size(snapshots[0], snapshots[-1])
    print(f"  Changed modules: {cumul_diff['changed_modules']}/{cumul_diff['total_modules']} ({cumul_diff['change_fraction']*100:.1f}%)", flush=True)
    print(f"  Mean rel change: {cumul_diff['mean_relative_change']*100:.3f}%", flush=True)
    print(f"  Changed params: {cumul_binary['changed_params']:,} / {cumul_binary['total_params']:,} ({cumul_binary['change_ratio']*100:.3f}%)", flush=True)
    print(f"  Est. diff size: {cumul_binary['estimated_diff_mb']:.2f} MB", flush=True)
    
    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Edit':>5} {'Changed %':>10} {'Mean Rel %':>12} {'Diff MB':>10}", flush=True)
    print("-" * 45, flush=True)
    for r in edit_results:
        print(f"{r['edit']:>5} {r['diff']['change_fraction']*100:>9.1f}% {r['diff']['mean_relative_change']*100:>11.3f}% {r['binary']['estimated_diff_mb']:>9.2f}", flush=True)
    print(f"{'Total':>5} {cumul_diff['change_fraction']*100:>9.1f}% {cumul_diff['mean_relative_change']*100:>11.3f}% {cumul_binary['estimated_diff_mb']:>9.2f}", flush=True)
    
    result = {
        "edit_diffs": edit_results,
        "cumulative": {
            "diff": cumul_diff,
            "binary": cumul_binary,
        },
    }
    
    with open("experiments/m216_results.json", "w") as f:
        json.dump(result, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print("\n✅ Saved to experiments/m216_results.json", flush=True)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
