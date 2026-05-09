#!/usr/bin/env python3
"""M53: WAL-0 compression + PPL validation on Llama 3.3 70B.
Optimized: no programs storage during encode, theoretical compression estimate."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wal import build_atoms_kmeans
from wal.compress import compute_compressed_size, compress_codebook

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
ENCODE_BATCH = 524288


def fast_encode(w_norm, atoms, l_max, batch=ENCODE_BATCH):
    """Fast encode without programs storage. Returns reconstruction only."""
    N = w_norm.numel()
    device = w_norm.device
    residual = w_norm.clone()
    atoms_gpu = atoms.to(device)
    recon = torch.zeros_like(w_norm)
    
    for step in range(l_max):
        best_ids = torch.empty(N, dtype=torch.int64, device=device)
        best_signs = torch.empty(N, dtype=torch.int64, device=device)
        for start in range(0, N, batch):
            end = min(start + batch, N)
            b = residual[start:end]
            sp = (b.unsqueeze(1) - atoms_gpu.unsqueeze(0)) ** 2
            sn = (b.unsqueeze(1) + atoms_gpu.unsqueeze(0)) ** 2
            mp, ip = sp.min(dim=1)
            mn, in_ = sn.min(dim=1)
            use_pos = mp < mn
            best_ids[start:end] = torch.where(use_pos, ip, in_)
            best_signs[start:end] = torch.where(use_pos, 1, -1)
        
        step_recon = atoms_gpu[best_ids] * best_signs.float()
        recon += step_recon
        residual -= step_recon
    
    return recon


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

print("Applying WAL Scalar...")
encoded_shapes = []  # list of (name, shape, numel, device)
stats = {"encoded": 0, "skipped": 0}

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
        
        if flat.numel() > SAMPLE_SIZE:
            idx_samp = torch.randperm(flat.numel())[:SAMPLE_SIZE]
            samples = flat[idx_samp]
        else:
            samples = flat
        
        atoms = build_atoms_kmeans(samples, K_ATOMS, KMEANS_ITERS, device=param_device)
        atoms = atoms.to(param_device)
        recon = fast_encode(flat.to(param_device), atoms, L_MAX)
        w_hat = recon.reshape(w.shape) * row_scale.to(param_device)
        param.data.copy_(w_hat.to(param.dtype))
        
        encoded_shapes.append((name, w.shape, w.numel(), param_device))
        stats["encoded"] += 1
        
        if stats["encoded"] % 50 == 0:
            elapsed = time.time() - t0_total
            print(f"  Encoded {stats['encoded']} params... elapsed={elapsed:.0f}s")
    else:
        stats["skipped"] += 1

print(f"\nEncoding done in {time.time() - t0_total:.0f}s")
print(f"  Encoded: {stats['encoded']}")
print(f"  Skipped: {stats['skipped']}")

# Theoretical compression analysis on subset (re-encode a few params for stats)
print(f"\nCompression analysis on {min(10, len(encoded_shapes))} representative params...")
subset = encoded_shapes[::max(1, len(encoded_shapes)//10)]
total_orig = 0
total_comp_raw = 0
total_comp_codebook = 0
total_comp_uint8 = 0
all_unique = []

for name, shape, numel, param_device in subset:
    param = dict(model.named_parameters())[name]
    w = param.data.float().to(param_device)
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    flat = w_norm.reshape(-1)
    
    # Re-encode just for stats (fast, no programs storage needed)
    samples = flat if flat.numel() <= SAMPLE_SIZE else flat[torch.randperm(flat.numel())[:SAMPLE_SIZE]]
    atoms = build_atoms_kmeans(samples, K_ATOMS, KMEANS_ITERS, device=param_device).to(param_device)
    recon = fast_encode(flat, atoms, L_MAX)
    
    # Build programs from recon for compression analysis
    from wal.isa import ProgramBuffer
    residual = flat.clone()
    indices = torch.zeros(flat.numel(), L_MAX, dtype=torch.uint8, device='cpu')
    signs = torch.zeros(flat.numel(), L_MAX, dtype=torch.int8, device='cpu')
    atoms_gpu = atoms.to(param_device)
    for step in range(L_MAX):
        sp = (residual.unsqueeze(1) - atoms_gpu.unsqueeze(0)) ** 2
        sn = (residual.unsqueeze(1) + atoms_gpu.unsqueeze(0)) ** 2
        mp, ip = sp.min(dim=1)
        mn, in_ = sn.min(dim=1)
        use_pos = mp < mn
        best_ids = torch.where(use_pos, ip, in_)
        best_signs = torch.where(use_pos, 1, -1)
        indices[:, step] = best_ids.cpu().to(torch.uint8)
        signs[:, step] = best_signs.cpu().to(torch.int8)
        residual -= atoms_gpu[best_ids] * best_signs.float()
    
    prog = ProgramBuffer(indices, signs, L_MAX)
    comp = compute_compressed_size(prog, K_ATOMS, row_scale.shape, use_codebook=True, row_scale_bits=16)
    
    total_orig += numel * 2
    total_comp_raw += numel * L_MAX * 2  # raw: uint8 idx + int8 sign
    total_comp_codebook += comp['total_bytes']
    total_comp_uint8 += numel * L_MAX * 1 + K_ATOMS * 4 + row_scale.numel() * 2  # uint8 pack + atoms + scales
    all_unique.append(comp['num_unique'])

ratio = len(encoded_shapes) / len(subset)
print(f"  Subset original:         {total_orig / 1e6:.1f} MB")
print(f"  Subset compressed (raw): {total_comp_raw / 1e6:.1f} MB  ({total_orig / total_comp_raw:.3f}x)")
print(f"  Subset compressed (cb):  {total_comp_codebook / 1e6:.1f} MB  ({total_orig / total_comp_codebook:.3f}x)")
print(f"  Subset compressed (u8):  {total_comp_uint8 / 1e6:.1f} MB  ({total_orig / total_comp_uint8:.3f}x)")

total_orig_all = total_orig * ratio
total_comp_raw_all = total_comp_raw * ratio
total_comp_cb_all = total_comp_codebook * ratio
total_comp_u8_all = total_comp_uint8 * ratio

print(f"\nExtrapolated for all {len(encoded_shapes)} encoded params:")
print(f"  Original:           {total_orig_all / 1e9:.2f} GB")
print(f"  Raw programs:       {total_comp_raw_all / 1e9:.2f} GB  ({total_orig_all / total_comp_raw_all:.3f}x)")
print(f"  Codebook dedup:     {total_comp_cb_all / 1e9:.2f} GB  ({total_orig_all / total_comp_cb_all:.3f}x)")
print(f"  Uint8 pack (K≤85):  {total_comp_u8_all / 1e9:.2f} GB  ({total_orig_all / total_comp_u8_all:.3f}x)")

if all_unique:
    unique_tensor = torch.tensor([u for u in all_unique if u is not None]).float()
    if len(unique_tensor) > 0:
        print(f"  Unique programs: mean={unique_tensor.mean():.0f}, max={unique_tensor.max():.0f}")
        print(f"  Params with ≤256 unique: {(unique_tensor <= 256).sum().item()}/{len(unique_tensor)}")

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
print(f"\nWAL Scalar K={K_ATOMS} lmax={L_MAX} PPL ({num_steps} steps, {end_loc} tokens): {ppl.item():.4f}")
print(f"Baseline was: 2.7805")
print(f"M46 (no compression analysis) was: 2.7821")
