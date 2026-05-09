"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M116 / Phase 16: Distribution-Level (Global) Atoms

Test whether one global atom table for the entire model can replace
560 per-layer tables without PPL degradation.

Approach:
1. Pool all linear weights across all layers
2. Build global atoms via k-means on pooled data (K=256)
3. Each layer gets its own coeff table but shares global atoms
4. Compare PPL vs per-layer baseline
"""
import torch
import torch.nn as nn
import sys, time
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from wal.v1 import replace_linear_with_wal
from wal.v1.nn import WALLinear, WALCachedLinear, replace_wal_with_linear, WALParameter
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.isa import AtomDef, AtomTableV1, CoeffTable

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
K_GLOBAL = 256
C_PER_LAYER = 16


def encode_with_global_atoms(weight, global_atoms, C=16):
    """Encode a single layer weight using global atoms + per-layer coeffs."""
    flat = weight.reshape(-1)
    atoms = global_atoms.to(flat.device)
    coeffs_tensor = build_coeff_table(flat, atoms, C=C, iters=3)
    prog, recon = wal_encode_v1(flat, atoms, coeffs_tensor, batch=262_144)
    
    atom_defs = [AtomDef(level=0, op="CONST") for _ in range(atoms.numel())]
    atom_table = AtomTableV1(base_atoms=atoms, atom_defs=atom_defs)
    coeff_table = CoeffTable(values=coeffs_tensor)
    
    return WALParameter(
        prog=prog, atom_table=atom_table, coeffs=coeff_table,
        shape=weight.shape, dtype=weight.dtype,
    )


def replace_linear_with_global_wal(model, global_atoms, C=16, cached=False):
    """Replace all nn.Linear with WALLinear using global atoms."""
    LinearClass = WALCachedLinear if cached else WALLinear
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            wal_param = encode_with_global_atoms(module.weight.data, global_atoms, C=C)
            new_layer = LinearClass(
                wal_weight=wal_param,
                bias=module.bias.data if module.bias is not None else None,
            )
            setattr(model, name, new_layer)
        else:
            replace_linear_with_global_wal(module, global_atoms, C=C, cached=cached)
    return model


def compute_ppl(model, tokenizer, texts, max_length=256):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            out = model(**inputs, labels=inputs["input_ids"])
            n = inputs["input_ids"].numel()
            total_loss += out.loss.item() * n
            total_tokens += n
    avg = total_loss / total_tokens
    return torch.exp(torch.tensor(avg)).item()


def main():
    print("=" * 70)
    print("M116 / Phase 16: Distribution-Level (Global) Atoms")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.strip()) > 20][:50]

    # ------------------------------------------------------------------
    # 1. Load dense model
    # ------------------------------------------------------------------
    print("\n[1] Loading dense model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )
    dense_ppl = compute_ppl(model, tokenizer, texts)
    print(f"    Dense PPL: {dense_ppl:.4f}", flush=True)

    # ------------------------------------------------------------------
    # 2. Collect all linear weights for global atom building
    # ------------------------------------------------------------------
    print("\n[2] Collecting all linear weights...", flush=True)
    all_weights = []
    total_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data.detach().clone()
            all_weights.append(w)
            total_elements += w.numel()
    print(f"    Total linear layers: {len(all_weights)}")
    print(f"    Total elements: {total_elements / 1e9:.2f}B")

    # ------------------------------------------------------------------
    # 3. Build global atoms from pooled weights
    # ------------------------------------------------------------------
    print(f"\n[3] Building global atoms (K={K_GLOBAL})...", flush=True)
    t0 = time.time()
    device = torch.device(DEVICE)
    
    # Sample from all weights (max 2M samples for speed)
    pooled_samples = []
    for w in all_weights:
        flat = w.reshape(-1)
        n_samp = min(flat.numel(), max(5000, flat.numel() // 100))
        idx = torch.randperm(flat.numel(), device=flat.device)[:n_samp]
        pooled_samples.append(flat[idx])
    
    pooled = torch.cat(pooled_samples)
    if pooled.numel() > 2_000_000:
        idx = torch.randperm(pooled.numel(), device=device)[:2_000_000]
        pooled = pooled[idx]
    
    global_atoms = build_l0_atoms(pooled, K=K_GLOBAL, iters=5, device=device)
    print(f"    Global atoms built in {time.time() - t0:.1f}s")
    print(f"    Atom range: [{global_atoms.min().item():.4f}, {global_atoms.max().item():.4f}]")

    # ------------------------------------------------------------------
    # 4. Encode entire model with global atoms
    # ------------------------------------------------------------------
    print("\n[4] Encoding model with global atoms...", flush=True)
    t0 = time.time()
    replace_linear_with_global_wal(model, global_atoms, C=C_PER_LAYER, cached=True)
    encode_time = time.time() - t0
    print(f"    Encode time: {encode_time:.1f}s")

    # ------------------------------------------------------------------
    # 5. Measure PPL with global atoms
    # ------------------------------------------------------------------
    print("\n[5] Global atoms PPL...", flush=True)
    global_ppl = compute_ppl(model, tokenizer, texts)
    print(f"    PPL: {global_ppl:.4f} (Δ vs dense: {global_ppl - dense_ppl:+.4f})")

    # ------------------------------------------------------------------
    # 6. Per-layer baseline for comparison
    # ------------------------------------------------------------------
    print("\n[6] Per-layer baseline (re-loading fresh model)...", flush=True)
    model2 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )
    t0 = time.time()
    replace_linear_with_wal(model2, K=256, C=16, build_hier=False, cached=True)
    perlayer_time = time.time() - t0
    perlayer_ppl = compute_ppl(model2, tokenizer, texts)
    print(f"    Encode time: {perlayer_time:.1f}s")
    print(f"    PPL: {perlayer_ppl:.4f} (Δ vs dense: {perlayer_ppl - dense_ppl:+.4f})")

    # ------------------------------------------------------------------
    # 7. Storage comparison
    # ------------------------------------------------------------------
    print("\n[7] Storage comparison...")
    n_layers = len(all_weights)
    per_layer_atom_bytes = n_layers * K_GLOBAL * 4  # 4 bytes per float32 atom
    global_atom_bytes = K_GLOBAL * 4
    
    print(f"    Per-layer atom tables: {per_layer_atom_bytes / 1e6:.1f} MB")
    print(f"    Global atom table:     {global_atom_bytes / 1e6:.1f} MB")
    print(f"    Savings:               {per_layer_atom_bytes / global_atom_bytes:.1f}×")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("M116 / Phase 16: SUMMARY")
    print("=" * 70)
    print(f"\n  PPL:")
    print(f"    Dense baseline:   {dense_ppl:.4f}")
    print(f"    Global atoms:     {global_ppl:.4f} (Δ={global_ppl - dense_ppl:+.4f})")
    print(f"    Per-layer atoms:  {perlayer_ppl:.4f} (Δ={perlayer_ppl - dense_ppl:+.4f})")
    print(f"\n  Timing:")
    print(f"    Global encode:    {encode_time:.1f}s")
    print(f"    Per-layer encode: {perlayer_time:.1f}s")
    print(f"\n  Storage (atom tables only):")
    print(f"    Per-layer: {per_layer_atom_bytes / 1e6:.1f} MB")
    print(f"    Global:    {global_atom_bytes / 1e6:.1f} MB")
    print(f"    Ratio:     {per_layer_atom_bytes / global_atom_bytes:.1f}×")

    ok_quality = abs(global_ppl - perlayer_ppl) < 0.5
    if ok_quality:
        print(f"\n  ✅ PASS: Global atoms match per-layer quality!")
    else:
        print(f"\n  🟡 PARTIAL: Global atoms work but PPL degraded vs per-layer.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
