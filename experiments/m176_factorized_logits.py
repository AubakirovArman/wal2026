"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M176 — Factorized Logits Memory Test.

Tests whether factorized logits (atom_logits + coeff_logits) solve memory explosion.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, json, sys, time, gc
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from wal.v1 import build_l0_atoms, build_coeff_table


def gumbel_softmax(logits, temperature=1.0, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / temperature
    y_soft = gumbels.softmax(dim=-1)
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y


class FactorizedGumbelWALLinear(nn.Module):
    """Gumbel-WAL with factorized logits: [N, K] + [N, C] instead of [N, K*C]."""
    
    def __init__(self, in_features, out_features, K=64, C=8):
        super().__init__()
        self.K, self.C = K, C
        self.out_features, self.in_features = out_features, in_features
        self.N = out_features * in_features
        
        self.dense_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        with torch.no_grad():
            w = self.dense_weight.data.reshape(-1)
            atoms = build_l0_atoms(w, K=K, iters=1)
            coeffs = build_coeff_table(w, atoms, C=C, iters=1)
            self.register_buffer('atoms', atoms)
            self.register_buffer('coeffs', coeffs)
        
        # Factorized logits: [N, K] + [N, C] instead of [N, K*C]
        self.atom_logits = nn.Parameter(torch.zeros(self.N, K))
        self.coeff_logits = nn.Parameter(torch.zeros(self.N, C))
    
    @property
    def weight(self):
        # Gumbel select atoms and coeffs independently
        atom_prog = gumbel_softmax(self.atom_logits, temperature=0.5, hard=True)  # [N, K]
        coeff_prog = gumbel_softmax(self.coeff_logits, temperature=0.5, hard=True)  # [N, C]
        
        # Weighted sum: sum_k sum_c atom_k * coeff_c * atom_prog_k * coeff_prog_c
        # = (atom_prog @ atoms) * (coeff_prog @ coeffs)
        selected_atoms = atom_prog @ self.atoms  # [N]
        selected_coeffs = coeff_prog @ self.coeffs  # [N]
        w = selected_atoms * selected_coeffs  # [N]
        
        return w.reshape(self.out_features, self.in_features)
    
    def forward(self, x):
        return F.linear(x, self.weight)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, n_heads=4, d_ff=1024, n_layers=4, use_factorized=False, K=64, C=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(128, d_model)
        
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        
        LinearClass = FactorizedGumbelWALLinear if use_factorized else nn.Linear
        self.head = LinearClass(d_model, vocab_size, K=K, C=C) if use_factorized else nn.Linear(d_model, vocab_size)
        
        self._param_count = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        h = self.encoder(h)
        return self.head(h)
    
    def count_params(self):
        return self._param_count


def train_model(model, device, steps=20, batch_size=4, seq_len=16, vocab_size=10000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    
    for step in range(steps):
        data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), target.reshape(-1))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses


def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0


def main():
    print("=" * 60)
    print("M176 — Factorized Logits Memory Test")
    print("=" * 60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    configs = [
        ("10M", 256, 4, 1024, 4),
        ("30M", 512, 6, 2048, 8),
    ]
    
    results = {}
    
    for name, d_model, n_layers, d_ff, n_heads in configs:
        print(f"\n{'='*50}")
        print(f"Config: {name} (d={d_model}, L={n_layers}, heads={n_heads})")
        print(f"{'='*50}")
        
        # Dense baseline
        print("\n--- Dense Baseline ---")
        model_dense = TinyTransformer(10000, d_model, n_heads, d_ff, n_layers, use_factorized=False).to(device)
        print(f"  Params: {model_dense.count_params()/1e6:.1f}M")
        
        torch.cuda.empty_cache()
        mem_before = get_gpu_memory()
        
        start = time.time()
        losses_dense = train_model(model_dense, device, steps=20)
        time_dense = time.time() - start
        
        mem_after = get_gpu_memory()
        print(f"  Final loss: {losses_dense[-1]:.4f}, Time: {time_dense:.1f}s, GPU: {mem_after-mem_before:.2f}GB")
        
        del model_dense
        torch.cuda.empty_cache()
        gc.collect()
        
        # Factorized Gumbel-WAL
        print("\n--- Factorized Gumbel-WAL ---")
        model_fact = TinyTransformer(10000, d_model, n_heads, d_ff, n_layers, use_factorized=True, K=64, C=8).to(device)
        print(f"  Params: {model_fact.count_params()/1e6:.1f}M")
        
        torch.cuda.empty_cache()
        mem_before = get_gpu_memory()
        
        start = time.time()
        losses_fact = train_model(model_fact, device, steps=20)
        time_fact = time.time() - start
        
        mem_after = get_gpu_memory()
        print(f"  Final loss: {losses_fact[-1]:.4f}, Time: {time_fact:.1f}s, GPU: {mem_after-mem_before:.2f}GB")
        
        results[name] = {
            "dense_final": losses_dense[-1],
            "dense_time": time_dense,
            "factorized_final": losses_fact[-1],
            "factorized_time": time_fact,
            "params_M": model_fact.count_params() / 1e6,
        }
        
        del model_fact
        torch.cuda.empty_cache()
        gc.collect()
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Config':>8} {'Dense':>10} {'Factorized':>12} {'F/D':>8}")
    print("-" * 45)
    for name, r in results.items():
        fd = r['factorized_final'] / max(r['dense_final'], 1e-10)
        print(f"{name:>8} {r['dense_final']:>10.4f} {r['factorized_final']:>12.4f} {fd:>8.2f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m176_factorized_logits.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
