#!/usr/bin/env python3
"""M177 — Gumbel-WAL Temperature Schedule Sweep.

Tests different temperature schedules to prevent program collapse.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, json, sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from wal.v1 import build_l0_atoms, build_coeff_table


def gumbel_softmax(logits, temperature=1.0, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / max(temperature, 0.01)
    y_soft = gumbels.softmax(dim=-1)
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y


class FactorizedGumbelWALLinear(nn.Module):
    def __init__(self, in_features, out_features, K=32, C=4):
        super().__init__()
        self.K, self.C = K, C
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
    
    def forward_with_temp(self, x, temperature=1.0, hard=False):
        atom_prog = gumbel_softmax(self.atom_logits, temperature, hard)
        coeff_prog = gumbel_softmax(self.coeff_logits, temperature, hard)
        selected_atoms = atom_prog @ self.atoms
        selected_coeffs = coeff_prog @ self.coeffs
        w = selected_atoms * selected_coeffs
        return F.linear(x, w.reshape(self.dense_weight.shape))
    
    def forward(self, x):
        return self.forward_with_temp(x, temperature=0.5, hard=True)


class TinyModel(nn.Module):
    def __init__(self, vocab=100, d=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.lin1 = FactorizedGumbelWALLinear(d, d * 2, K=32, C=4)
        self.lin2 = FactorizedGumbelWALLinear(d * 2, d, K=32, C=4)
        self.head = FactorizedGumbelWALLinear(d, vocab, K=32, C=4)
    
    def forward_with_temp(self, x, temperature=1.0, hard=False):
        h = self.embed(x)
        h = torch.relu(self.lin1.forward_with_temp(h, temperature, hard))
        h = torch.relu(self.lin2.forward_with_temp(h, temperature, hard))
        return self.head.forward_with_temp(h, temperature, hard)


def get_temperature(step, total_steps, start=2.0, end=0.1, schedule="linear"):
    progress = step / total_steps
    if schedule == "linear":
        return start + (end - start) * progress
    elif schedule == "cosine":
        return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2
    elif schedule == "exponential":
        return start * (end / start) ** progress
    else:
        return start


def train_with_schedule(model, device, steps=50, schedule="linear", start=2.0, end=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    entropies = []
    dead_atoms = []
    
    for step in range(steps):
        temp = get_temperature(step, steps, start, end, schedule)
        
        data = torch.randint(0, 100, (4, 16), device=device)
        target = torch.randint(0, 100, (4, 16), device=device)
        
        optimizer.zero_grad()
        out = model.forward_with_temp(data, temperature=temp, hard=True)
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), target.reshape(-1))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Metrics
        with torch.no_grad():
            probs = model.lin1.atom_logits.softmax(dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            entropies.append(entropy)
            
            # Dead atoms = never selected
            dead = (probs.max(dim=0).values < 0.01).sum().item()
            dead_atoms.append(dead)
    
    return losses, entropies, dead_atoms


def main():
    print("=" * 60)
    print("M177 — Gumbel-WAL Temperature Schedule Sweep")
    print("=" * 60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    schedules = [
        ("constant_high", 2.0, 2.0, "linear"),
        ("constant_low", 0.1, 0.1, "linear"),
        ("linear_decay", 2.0, 0.1, "linear"),
        ("cosine_decay", 2.0, 0.1, "cosine"),
        ("exp_decay", 2.0, 0.1, "exponential"),
        ("linear_sharp", 5.0, 0.03, "linear"),
    ]
    
    results = {}
    
    for name, start, end, sched in schedules:
        print(f"\n--- {name}: {start} → {end} ({sched}) ---")
        
        model = TinyModel(vocab=100, d=64).to(device)
        losses, entropies, dead_atoms = train_with_schedule(
            model, device, steps=50, schedule=sched, start=start, end=end
        )
        
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Final entropy: {entropies[-1]:.4f}")
        print(f"  Dead atoms: {dead_atoms[-1]}/{model.lin1.K}")
        print(f"  Loss stability (std): {torch.tensor(losses[-10:]).std().item():.4f}")
        
        results[name] = {
            "losses": losses,
            "entropies": entropies,
            "dead_atoms": dead_atoms,
            "final_loss": losses[-1],
            "final_entropy": entropies[-1],
            "final_dead": dead_atoms[-1],
            "start": start,
            "end": end,
            "schedule": sched,
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Schedule':>18} {'Loss':>10} {'Entropy':>10} {'Dead':>8} {'Stability':>10}")
    print("-" * 60)
    for name, r in results.items():
        stab = torch.tensor(r['losses'][-10:]).std().item()
        print(f"{name:>18} {r['final_loss']:>10.4f} {r['final_entropy']:>10.4f} {r['final_dead']:>8} {stab:>10.4f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m177_temperature_schedule.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
