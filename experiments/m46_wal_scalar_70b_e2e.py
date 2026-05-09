#!/usr/bin/env python3
"""M46: WAL Scalar end-to-end on Llama 3.3 70B — lmax=2, K=128, skip spiky."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from route_encoder import encode_routes
from codebook import build_codebook

DEVICE = torch.device("cuda:2")
model_name = "unsloth/Llama-3.3-70B-Instruct"
max_length = 2048
stride = 512
max_samples = 16

K_ATOMS = 128
L_MAX = 2
KMEANS_ITERS = 5
SAMPLE_SIZE = 1_000_000
SPIKY_THRESHOLD = 0.08


def kmeans_gpu(samples, K, iters=5):
    """Mini-batch k-means on GPU."""
    samples = samples.to(DEVICE)
    atoms = torch.zeros(K, device=DEVICE, dtype=torch.float32)
    atoms[0] = samples[torch.randint(0, samples.numel(), (1,), device=DEVICE)]
    for k in range(1, K):
        dists = (samples.unsqueeze(1) - atoms[:k].unsqueeze(0)).abs().min(dim=1)[0]
        probs = dists / dists.sum()
        cumprobs = probs.cumsum(dim=0)
        idx = torch.searchsorted(cumprobs, torch.rand(1, device=DEVICE))
        idx = idx.clamp_max(samples.numel() - 1)
        atoms[k] = samples[idx]
    for _ in range(iters):
        assignments = torch.empty(samples.numel(), dtype=torch.int64, device=DEVICE)
        batch = 262144
        for start in range(0, samples.numel(), batch):
            end = min(start + batch, samples.numel())
            assignments[start:end] = (samples[start:end].unsqueeze(1) - atoms.unsqueeze(0)).abs().argmin(dim=1)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                atoms[k] = samples[mask].mean()
    return atoms.cpu()


def encode_wal_scalar(w_norm, atoms, l_max):
    """Greedy ternary residual encoding. Returns programs and reconstruction."""
    N = w_norm.numel()
    programs = torch.zeros(N, l_max, 2, dtype=torch.int64)
    residual = w_norm.clone()
    atoms_gpu = atoms.to(w_norm.device)
    
    for step in range(l_max):
        best_ids = torch.empty(N, dtype=torch.int64, device=w_norm.device)
        best_signs = torch.empty(N, dtype=torch.int64, device=w_norm.device)
        batch = 262144
        for start in range(0, N, batch):
            end = min(start + batch, N)
            b = residual[start:end]
            sp = (b.unsqueeze(1) - atoms_gpu.unsqueeze(0)) ** 2
            sn = (b.unsqueeze(1) + atoms_gpu.unsqueeze(0)) ** 2
            mp, ip = sp.min(dim=1)
            mn, in_ = sn.min(dim=1)
            use_pos = mp < mn
            best_ids[start:end] = torch.where(use_pos, ip, in_)
            best_signs[start:end] = torch.where(use_pos, torch.tensor(1, device=w_norm.device), torch.tensor(-1, device=w_norm.device))
        
        programs[:, step, 0] = best_ids.cpu()
        programs[:, step, 1] = best_signs.cpu()
        residual -= atoms_gpu[best_ids] * best_signs.float()
    
    # Decode
    recon = torch.zeros_like(w_norm)
    for step in range(l_max):
        ids = programs[:, step, 0].to(w_norm.device)
        signs = programs[:, step, 1].float().to(w_norm.device)
        recon += atoms_gpu[ids] * signs
    
    return programs, recon


print(f"Loading {model_name}...")
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

print("Applying WAL Scalar v0.1, lmax=2, K=128, skip spiky...")
stats = {"encoded": 0, "skipped": 0}
for name, param in list(model.named_parameters()):
    if len(param.shape) != 2:
        stats["skipped"] += 1
        continue
    if "embed_tokens" in name or "lm_head" in name:
        stats["skipped"] += 1
        continue
    
    w = param.data.float()
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    std = w_norm.std().item()
    is_spiky = std < SPIKY_THRESHOLD
    
    if not is_spiky:
        # Sample for k-means
        flat = w_norm.reshape(-1)
        if flat.numel() > SAMPLE_SIZE:
            idx = torch.randperm(flat.numel())[:SAMPLE_SIZE]
            samples = flat[idx]
        else:
            samples = flat
        
        atoms = kmeans_gpu(samples, K_ATOMS, KMEANS_ITERS)
        _, recon = encode_wal_scalar(flat.to(DEVICE), atoms, L_MAX)
        w_hat = recon.reshape(w.shape) * row_scale
        param.data.copy_(w_hat.to(param.dtype))
        stats["encoded"] += 1
        if stats["encoded"] % 50 == 0:
            print(f"  Encoded {stats['encoded']} params...")
    else:
        stats["skipped"] += 1

print(f"Encoding done: {stats}")

# PPL
print("Loading WikiText-2...")
ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"][:])
encodings = tokenizer(text, return_tensors="pt")
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
num_steps = 0
for begin_loc in range(0, seq_len, stride):
    if num_steps >= max_samples:
        break
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc
    input_ids = encodings.input_ids[:, begin_loc:end_loc]
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids.to(model.device), labels=target_ids.to(model.device))
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)
    prev_end_loc = end_loc
    num_steps += 1
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(f"\nWAL Scalar lmax=2 K=128 PPL ({num_steps} steps, {end_loc} tokens): {ppl.item():.4f}")
print(f"Baseline was: 2.7805")
