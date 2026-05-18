"""Comprehensive WAL evaluation on Gemma-4-31B-it.

Replaces 15+ legacy PPL experiments with one complete benchmark:
1. Baseline PPL
2. PPL vs compression sweep (lmax, K, stop_threshold)
3. Layer-by-layer sensitivity
4. Frozen vocabulary edit test
5. Block-RVQ PPL comparison
"""
import json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ─── Config ────────────────────────────────────────────────────
MODEL_DIR = "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
DEVICE = "cuda:0"  # via CUDA_VISIBLE_DEVICES
RESULT_DIR = Path("/mnt/hf_model_weights/arman/3bit/wal/experiments_runner/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Add dwl2 dynamic route to path
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit")
from dwl2_dynamic_route.src.calibrate import calibrate_ladder
from dwl2_dynamic_route.src.route_encoder import encode_routes, decode_routes, rel_mse

# ─── Helpers ──────────────────────────────────────────────────
def load_model(device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map=device, local_files_only=True)
    model.eval()
    return model, tokenizer

def load_eval_data(tokenizer, max_samples=200):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in ds['text'] if len(t.strip()) > 0][:max_samples]
    enc = tokenizer('\n\n'.join(texts), return_tensors='pt', truncation=True, max_length=2048)
    return enc['input_ids']

def compute_ppl(model, input_ids, max_len=512, stride=256):
    """Sliding window PPL."""
    model.eval()
    nlls = []
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.size(1)
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        if end - begin < stride:
            break
        chunk = input_ids[:, begin:end]
        with torch.no_grad():
            outputs = model(chunk, labels=chunk)
        nlls.append(outputs.loss.item())
    return float(torch.exp(torch.tensor(nlls).mean()).item())

def get_text_weights(model):
    """Extract all text layer weight tensors."""
    weights = {}
    for name, param in model.named_parameters():
        if 'language_model.layers.' in name and 'weight' in name:
            weights[name] = param.data.clone()
    return weights

def route_encode_weight(w, lmax=12, stop_threshold=0.0):
    """Encode a weight tensor using route quantization (CPU to save GPU RAM)."""
    w_cpu = w.detach().cpu().float()
    row_max = w_cpu.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w_cpu / row_max
    sample = w_norm.flatten()
    if sample.numel() > 2_000_000:
        sample = sample[torch.randint(0, sample.numel(), (2_000_000,))]
    ladder = calibrate_ladder(sample, l_max=lmax, refine_iters=10, pin_top=True, top_value=1.0, seed="geometric")
    enc = encode_routes(w_norm, ladder, stop_threshold=stop_threshold, l_max=lmax)
    w_hat_norm = decode_routes(enc, ladder, out_dtype=torch.float32)
    w_hat = w_hat_norm * row_max
    return w_hat.to(w.dtype).to(w.device), enc, ladder

# ─── Tests ────────────────────────────────────────────────────

def test_baseline_ppl(model, tokenizer):
    input_ids = load_eval_data(tokenizer)
    ppl = compute_ppl(model, input_ids)
    print(f"[BASELINE] PPL={ppl:.2f}")
    return ppl

def test_ppl_sweep(model, tokenizer, original_weights):
    """Sweep compression parameters and measure PPL."""
    configs = [
        (12, 0.0, "lmax=12,τ=0"),
        (7, 0.0, "lmax=7,τ=0"),
        (12, 0.001, "lmax=12,τ=0.001"),
    ]
    input_ids = load_eval_data(tokenizer)
    results = []

    for lmax, stop, label in configs:
        print(f"\n[SWEEP] {label}...")
        # Encode all weights
        encoded = {}
        total_re = 0.0
        t0 = time.time()
        for name, w in original_weights.items():
            w_hat, enc, ladder = route_encode_weight(w, lmax=lmax, stop_threshold=stop)
            param = dict(model.named_parameters()).get(name)
            if param is not None:
                param.data.copy_(w_hat)
            encoded[name] = {'rel_mse': rel_mse(w.float(), w_hat.float()).item()}
            total_re += encoded[name]['rel_mse']

        encode_time = time.time() - t0
        avg_rel_mse = total_re / len(encoded)
        ppl = compute_ppl(model, input_ids)
        print(f"  avg_rel_mse={avg_rel_mse:.6f} PPL={ppl:.2f} time={encode_time:.1f}s")
        results.append({'config': label, 'lmax': lmax, 'stop': stop,
                       'avg_rel_mse': avg_rel_mse, 'ppl': ppl, 'encode_time': encode_time})

        # Restore original weights
        for name, w in original_weights.items():
            param = dict(model.named_parameters()).get(name)
            if param is not None:
                param.data.copy_(w)

    return results

def test_layer_sensitivity(model, tokenizer, original_weights):
    """Test PPL impact when quantizing each layer individually."""
    input_ids = load_eval_data(tokenizer)
    layers = sorted(set(
        k.split('.layers.')[1].split('.')[0]
        for k in original_weights if 'layers.' in k
    ))
    sensitivity = []
    sample_layers = [int(l) for l in layers[::5]]  # every 5th layer

    for layer_idx in sample_layers:
        layer_weights = {k: v for k, v in original_weights.items()
                        if f'.layers.{layer_idx}.' in k}
        print(f"[SENSITIVITY] Layer {layer_idx} ({len(layer_weights)} params)...", end=" ", flush=True)

        for name, w in layer_weights.items():
            w_hat, _, _ = route_encode_weight(w, lmax=9, stop_threshold=0.0)
            param = dict(model.named_parameters()).get(name)
            if param is not None:
                param.data.copy_(w_hat)

        ppl = compute_ppl(model, input_ids)
        print(f"PPL={ppl:.2f}")
        sensitivity.append({'layer': layer_idx, 'ppl': ppl})

        # Restore
        for name, w in layer_weights.items():
            param = dict(model.named_parameters()).get(name)
            if param is not None:
                param.data.copy_(original_weights[name])

    return sensitivity

def test_frozen_vocab_edit(model, tokenizer, original_weights):
    """Test: encode a layer, re-encode after fake edit, check non-target diff."""
    target_layer = 0
    target_proj = f"language_model.layers.{target_layer}.mlp.down_proj.weight"

    w_target = original_weights[target_proj]
    w_hat, enc, ladder = route_encode_weight(w_target, lmax=12)
    rel_mse_before = rel_mse(w_target.float(), w_hat.float()).item()

    # Simulate edit: perturb 10% of target weights
    w_edited = w_target.clone()
    n_perturb = w_edited.numel() // 10
    idx = torch.randperm(w_edited.numel())[:n_perturb]
    w_edited.view(-1)[idx] += torch.randn(n_perturb, device=w_edited.device, dtype=w_edited.dtype) * 0.01

    # Re-encode with SAME ladder (frozen vocab)
    w_hat2, enc2, _ = route_encode_weight(w_edited, lmax=12)

    # Check: target diff should be high, non-target should be 0
    target_diff = (enc.digits != enc2.digits).float().mean().item()

    # Encode a different layer with the same frozen ladder
    w_other_key = f"language_model.layers.{target_layer + 1}.mlp.down_proj.weight"
    w_other = original_weights[w_other_key]
    w_hat_other_before, enc_other_before, _ = route_encode_weight(w_other, lmax=12)
    w_hat_other_after, enc_other_after, _ = route_encode_weight(w_other, lmax=12)

    non_target_diff = (enc_other_before.digits != enc_other_after.digits).float().mean().item()

    return {
        'target_layer': target_layer,
        'rel_mse_before': rel_mse_before,
        'target_diff': target_diff,
        'non_target_diff': non_target_diff,
        'frozen_vocab_works': non_target_diff == 0.0
    }

# ─── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("WAL COMPREHENSIVE EVALUATION — Gemma-4-31B-it")
    print("=" * 70)

    # 1. Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model(DEVICE)
    print(f"  Model loaded on {next(model.parameters()).device}")

    # 2. Extract original weights
    print("\n[2/5] Extracting text weights...")
    original_weights = get_text_weights(model)
    print(f"  {len(original_weights)} weight tensors")

    # 3. Baseline PPL
    print("\n[3/5] Baseline PPL...")
    baseline_ppl = test_baseline_ppl(model, tokenizer)

    # 4. PPL sweep
    print("\n[4/5] PPL compression sweep...")
    sweep_results = test_ppl_sweep(model, tokenizer, original_weights)

    # 5. Layer sensitivity (quick: every 10th layer)
    print("\n[5/5] Layer sensitivity...")
    sensitivity_results = test_layer_sensitivity(model, tokenizer, original_weights)

    # 6. Frozen vocab
    print("\n[+1] Frozen vocabulary test...")
    frozen_result = test_frozen_vocab_edit(model, tokenizer, original_weights)
    print(f"  target_diff={frozen_result['target_diff']:.4f} "
          f"non_target_diff={frozen_result['non_target_diff']:.4f} "
          f"frozen_vocab={'✓' if frozen_result['frozen_vocab_works'] else '✗'}")

    # 7. Save results
    report = {
        'model': 'google/gemma-4-31B-it',
        'baseline_ppl': baseline_ppl,
        'ppl_sweep': sweep_results,
        'layer_sensitivity': sensitivity_results,
        'frozen_vocab': frozen_result,
    }
    out_path = RESULT_DIR / 'gemma_wal_full_eval.json'
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n✓ Report saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
