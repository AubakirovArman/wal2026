#!/usr/bin/env python3
"""M57: WAL-0 with codebook layer — full 70B encode + PPL.

For each parameter:
  1. Row-normalize
  2. K-means atoms
  3. GPU-native encode (wal_encode_scalar_gpu)
  4. Build codebook (unique programs → precomputed recon values)
  5. Apply precomputed recon to model weights
  6. Run WikiText-2 PPL
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu

DEVICE = torch.device("cuda:3")
model_name = "unsloth/Llama-3.3-70B-Instruct"
max_length = 2048
stride = 512
max_samples = 16

K_ATOMS = 128
L_MAX = 2
KMEANS_ITERS = 5
SAMPLE_SIZE = 1_000_000
SPIKY_THRESHOLD = 0.08

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

print("Applying WAL-0 with CODEBOOK layer...")
stats = {"encoded": 0, "skipped": 0, "total_unique_programs": 0}

t0_total = time.time()
for idx, (name, param) in enumerate(list(model.named_parameters())):
    if len(param.shape) != 2:
        stats["skipped"] += 1
        continue
    if "embed_tokens" in name or "lm_head" in name:
        stats["skipped"] += 1
        continue
    
    w = param.data.float()
    std = (w / w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)).std().item()
    is_spiky = std < SPIKY_THRESHOLD
    
    if not is_spiky:
        param_device = param.device
        row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        w_norm = w / row_scale
        flat = w_norm.reshape(-1)
        
        # Sample for k-means
        if flat.numel() > SAMPLE_SIZE:
            idx_samp = torch.randperm(flat.numel(), device=param_device)[:SAMPLE_SIZE]
            samples = flat[idx_samp]
        else:
            samples = flat
        
        # Atoms on GPU
        atoms = build_atoms_kmeans(samples, K_ATOMS, KMEANS_ITERS, device=param_device)
        atoms = atoms.to(param_device)
        
        # Encode GPU-native
        indices, signs, recon = wal_encode_scalar_gpu(flat.to(param_device), atoms, L_MAX)
        
        # Build codebook on GPU
        def pack_programs(indices, signs):
            sign_map = signs.clone()
            sign_map[signs == -1] = 0
            sign_map[signs == 0] = 1
            sign_map[signs == 1] = 2
            packed = (sign_map[:, 0].long() << 16) | (indices[:, 0].long() << 9) | \
                     (sign_map[:, 1].long() << 7) | (indices[:, 1].long())
            return packed
        
        packed = pack_programs(indices, signs)
        unique_prog, inverse, counts = torch.unique(packed, return_inverse=True, return_counts=True)
        num_unique = unique_prog.numel()
        stats["total_unique_programs"] += num_unique
        
        # Precomputed recon values
        def unpack_program(packed_val):
            v = int(packed_val)
            atom_1 = v & 0x7F
            sign_1 = (v >> 7) & 0x3
            atom_0 = (v >> 9) & 0x7F
            sign_0 = (v >> 16) & 0x3
            sign_map_rev = {0: -1, 1: 0, 2: 1}
            return (atom_0, sign_map_rev[sign_0], atom_1, sign_map_rev[sign_1])
        
        codebook_recon = torch.zeros(num_unique, dtype=torch.float32, device=param_device)
        for i in range(num_unique):
            a0, s0, a1, s1 = unpack_program(int(unique_prog[i].item()))
            codebook_recon[i] = atoms[a0].item() * s0 + atoms[a1].item() * s1
        
        # Map packed programs to codebook IDs
        # Use inverse from torch.unique (already 0..num_unique-1, matches unique_prog order)
        program_ids = inverse
        
        # Apply precomputed recon
        recon_from_codebook = codebook_recon[program_ids]
        w_hat = recon_from_codebook.reshape(w.shape) * row_scale.to(param_device)
        param.data.copy_(w_hat.to(param.dtype))
        
        stats["encoded"] += 1
        if stats["encoded"] % 50 == 0:
            elapsed = time.time() - t0_total
            print(f"  Encoded {stats['encoded']} params... elapsed={elapsed:.0f}s, total_unique_programs={stats['total_unique_programs']:,}")
    else:
        stats["skipped"] += 1

print(f"\nEncoding done in {time.time() - t0_total:.0f}s")
print(f"  Encoded: {stats['encoded']}")
print(f"  Skipped: {stats['skipped']}")
print(f"  Total unique programs across all layers: {stats['total_unique_programs']:,}")
print(f"  Avg unique programs per layer: {stats['total_unique_programs'] / max(stats['encoded'], 1):.0f}")

# PPL
print("\nLoading WikiText-2...")
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
    if num_steps % 4 == 0:
        print(f"  Step {num_steps}/{max_samples} loss={outputs.loss.item():.4f}")
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(f"\nWAL-0 Codebook K={K_ATOMS} lmax={L_MAX} PPL ({num_steps} steps, {end_loc} tokens): {ppl.item():.4f}")
print(f"Baseline was: 2.7805")
print(f"M46 (PyTorch encode) was: 2.7821")
print(f"M53c (Fused Triton encode) was: 2.7858")
