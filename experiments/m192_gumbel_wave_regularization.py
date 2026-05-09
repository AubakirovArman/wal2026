#!/usr/bin/env python3
"""M192 — Gumbel-WAL + Wave Regularization.

Integrates spectral wave penalty into Gumbel-WAL training loop.
Hypothesis: wave regularization improves spectral distribution
of learned weights, reducing risk of resonant perturbations.
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


def spectral_entropy(w):
    """Compute spectral entropy of flattened weight (higher = more uniform)."""
    w_flat = w.reshape(-1).float()
    fft = torch.fft.fft(w_flat)
    amps = fft.abs()
    probs = amps / (amps.sum() + 1e-10)
    return -(probs * torch.log(probs + 1e-10)).sum()


def top10_energy_ratio(w):
    """Fraction of energy in top 10 FFT components (lower = more uniform)."""
    w_flat = w.reshape(-1).float()
    fft = torch.fft.fft(w_flat)
    amps = fft.abs()
    sorted_amps = amps.sort(descending=True).values
    return sorted_amps[:10].sum() / (amps.sum() + 1e-10)


def spectral_norm(w):
    if w.dim() >= 2:
        return torch.linalg.matrix_norm(w, ord=2)
    return w.abs().max()


class FactorizedGumbelWALLinear(nn.Module):
    """Gumbel-WAL with factorized logits and optional wave regularization."""
    
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
        
        self.atom_logits = nn.Parameter(torch.zeros(self.N, K))
        self.coeff_logits = nn.Parameter(torch.zeros(self.N, C))
    
    @property
    def weight(self):
        atom_prog = gumbel_softmax(self.atom_logits, temperature=0.5, hard=True)
        coeff_prog = gumbel_softmax(self.coeff_logits, temperature=0.5, hard=True)
        selected_atoms = atom_prog @ self.atoms
        selected_coeffs = coeff_prog @ self.coeffs
        w = selected_atoms * selected_coeffs
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


def train_model(model, device, steps=50, batch_size=4, seq_len=16, vocab_size=10000, wave_lambda=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    task_losses = []
    wave_losses = []
    entropies = []
    top10s = []
    spec_norms = []
    
    for step in range(steps):
        data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        optimizer.zero_grad()
        out = model(data)
        task_loss = F.cross_entropy(out.reshape(-1, out.size(-1)), target.reshape(-1))
        
        # Wave regularization on head weight
        if wave_lambda > 0 and hasattr(model.head, 'weight'):
            w = model.head.weight
            wave_penalty = top10_energy_ratio(w)  # Minimize spectral concentration
            loss = task_loss + wave_lambda * wave_penalty
            wave_losses.append(wave_penalty.item())
        else:
            loss = task_loss
            wave_losses.append(0.0)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        task_losses.append(task_loss.item())
        
        # Track spectral stats
        with torch.no_grad():
            w = model.head.weight
            entropies.append(spectral_entropy(w).item())
            top10s.append(top10_energy_ratio(w).item())
            spec_norms.append(spectral_norm(w).item())
    
    return {
        'losses': losses,
        'task_losses': task_losses,
        'wave_losses': wave_losses,
        'entropies': entropies,
        'top10s': top10s,
        'spec_norms': spec_norms,
    }


def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0


def main():
    print("=" * 60)
    print("M192 — Gumbel-WAL + Wave Regularization")
    print("=" * 60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    d_model, n_layers, d_ff, n_heads = 256, 4, 1024, 4
    
    configs = [
        ("Baseline", 0.0),
        ("Wave_λ=0.1", 0.1),
        ("Wave_λ=1.0", 1.0),
    ]
    
    results = {}
    
    for name, wave_lambda in configs:
        print(f"\n{'='*50}")
        print(f"Config: {name}")
        print(f"{'='*50}")
        
        model = TinyTransformer(10000, d_model, n_heads, d_ff, n_layers, use_factorized=True, K=64, C=8).to(device)
        print(f"  Params: {model.count_params()/1e6:.1f}M")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        start = time.time()
        stats = train_model(model, device, steps=50, wave_lambda=wave_lambda)
        elapsed = time.time() - start
        
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Final task loss: {stats['task_losses'][-1]:.4f}")
        print(f"  Final wave loss: {stats['wave_losses'][-1]:.4f}")
        print(f"  Spectral entropy: {stats['entropies'][-1]:.4f}")
        print(f"  Top10 energy: {stats['top10s'][-1]:.4f}")
        print(f"  Spectral norm: {stats['spec_norms'][-1]:.2f}")
        
        # Trend: first 10 vs last 10
        early_task = sum(stats['task_losses'][:10]) / 10
        late_task = sum(stats['task_losses'][-10:]) / 10
        early_top10 = sum(stats['top10s'][:10]) / 10
        late_top10 = sum(stats['top10s'][-10:]) / 10
        early_ent = sum(stats['entropies'][:10]) / 10
        late_ent = sum(stats['entropies'][-10:]) / 10
        
        print(f"  Task loss: {early_task:.4f} → {late_task:.4f}")
        print(f"  Top10 energy: {early_top10:.4f} → {late_top10:.4f} (Δ={late_top10-early_top10:+.4f})")
        print(f"  Spectral entropy: {early_ent:.4f} → {late_ent:.4f} (Δ={late_ent-early_ent:+.4f})")
        
        results[name] = {
            'final_task_loss': stats['task_losses'][-1],
            'final_wave_loss': stats['wave_losses'][-1],
            'final_entropy': stats['entropies'][-1],
            'final_top10': stats['top10s'][-1],
            'final_spec_norm': stats['spec_norms'][-1],
            'early_task': early_task,
            'late_task': late_task,
            'early_top10': early_top10,
            'late_top10': late_top10,
            'early_entropy': early_ent,
            'late_entropy': late_ent,
            'time': elapsed,
        }
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Config':>12} {'TaskLoss':>10} {'Top10':>8} {'Entropy':>10} {'SpecNorm':>10}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:>12} {r['final_task_loss']:>10.4f} {r['final_top10']:>8.4f} "
              f"{r['final_entropy']:>10.4f} {r['final_spec_norm']:>10.2f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m192_gumbel_wave_regularization.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")


if __name__ == "__main__":
    main()
