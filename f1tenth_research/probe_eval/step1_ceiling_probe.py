"""
step1_ceiling_probe.py — Step 1: the DECODE CEILING test.

Question: is each physics parameter decodable from the latent it is ENCODED
INTO — i.e. from z = mu(e) itself? This needs no rollouts: we sample e
synthetically from PhysicsRandomizer, push it through the trained mu, and fit
probes z -> e_i.

Why this matters: phi is trained to reproduce mu(e). If a parameter is not
recoverable from mu(e) (mu collapsed it), then NO phi — with any window, any
architecture, any training time — can ever decode it. This closes the decode
question by construction, upstream of everything phi-related.

Two probes per parameter:
  - LINEAR (ridge): what the E5 linear probe can hope for.
  - NONLINEAR (2-layer MLP): the information-theoretic-ish ceiling. Since
    z = mu(e) is a deterministic function, a LOW nonlinear R^2 is strong
    evidence mu genuinely discarded that parameter.

Interpretation grid for grip_factor:
  nonlinear R^2 high, linear high  -> grip lives in z linearly; E5 probe is fine
  nonlinear high,   linear low     -> grip is in z but curved; use MLP probe in E5
  nonlinear low                    -> mu discarded grip; decoding phi was doomed
                                      from the start (report as a finding)

Run:
    python -m f1tenth_research.probe_eval.step1_ceiling_probe \
        --actor_critic checkpoints/phase1_lidar/final.pt \
        --config f1tenth_research/configs/rma_config.yaml \
        --n_samples 50000 --outdir probe_results/step1
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import yaml

from ..models import RMAActorCritic
from ..envs import PhysicsRandomizer
from .probe_utils import (PARAM_KEYS, ridge_fit, ridge_predict, r2_score)
from .collect_probe_dataset import get_encoder


class MLPProbe(nn.Module):
    """Small nonlinear probe: 8 -> 64 -> 64 -> 1."""

    def __init__(self, in_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp_probe(X_tr, y_tr, X_te, device, epochs=200, lr=1e-3, bs=1024):
    """Train an MLP probe; return predictions on X_te."""
    model = MLPProbe(X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X_tr).float().to(device)
    yt = torch.from_numpy(y_tr).float().to(device)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_te).float().to(device)).cpu().numpy()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actor_critic', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--n_samples', type=int, default=50000)
    ap.add_argument('--ridge_lam', type=float, default=1.0)
    ap.add_argument('--outdir', default='probe_results/step1')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- Load mu from the trained checkpoint ---------------------------------
    # obs_dim only affects the policy head; take it from config, fall back 41.
    obs_dim = 113  # v29/v30 real obs_dim (5 base + 108 lidar); policy head unused by this script
    actor_critic = RMAActorCritic(obs_dim=obs_dim).to(device)
    ckpt = torch.load(args.actor_critic, map_location=device)
    actor_critic.load_state_dict(ckpt['actor_critic'])
    actor_critic.eval()
    encoder = get_encoder(actor_critic)

    # --- Synthetic e sampling from the training distribution ----------------
    randomizer = PhysicsRandomizer(config['environment']['randomization'])
    E = np.zeros((args.n_samples, len(PARAM_KEYS)), dtype=np.float32)
    for i in range(args.n_samples):
        p = randomizer.sample_training()
        E[i] = [p[k] for k in PARAM_KEYS]

    with torch.no_grad():
        Z = encoder(torch.from_numpy(E).float().to(device)).cpu().numpy()
    print(f"[step1] sampled {args.n_samples} e vectors; z shape {Z.shape}")

    # --- 80/20 split (iid samples; no episode structure here) ---------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(args.n_samples)
    n_te = args.n_samples // 5
    te, tr = perm[:n_te], perm[n_te:]

    results = {}
    print(f"\n{'parameter':<24}{'linear R^2':>12}{'nonlinear R^2':>15}")
    print('-' * 51)
    for i, key in enumerate(PARAM_KEYS):
        y = E[:, i]
        # linear ridge probe
        lin = ridge_fit(Z[tr], y[tr], lam=args.ridge_lam)
        r2_lin = r2_score(y[te], ridge_predict(lin, Z[te]))
        # nonlinear MLP probe
        pred_nl = fit_mlp_probe(Z[tr], y[tr], Z[te], device)
        r2_nl = r2_score(y[te], pred_nl)

        results[key] = {'linear_r2': r2_lin, 'nonlinear_r2': r2_nl}
        print(f"{key:<24}{r2_lin:>12.4f}{r2_nl:>15.4f}")

    with open(os.path.join(args.outdir, 'step1_ceiling_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # --- Figure: per-parameter decode ceiling --------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(PARAM_KEYS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, [results[k]['linear_r2'] for k in PARAM_KEYS], w,
           label='linear probe (ridge)')
    ax.bar(x + w / 2, [results[k]['nonlinear_r2'] for k in PARAM_KEYS], w,
           label='nonlinear probe (MLP)')
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_KEYS, rotation=30, ha='right')
    ax.set_ylabel(r'held-out $R^2$')
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(r'Step 1 — decode ceiling: probes from true $z=\mu(e)$ to each parameter')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'step1_ceiling.png'), dpi=150)
    print(f"\n[step1] results -> {args.outdir}/step1_ceiling_results.json, step1_ceiling.png")

    # --- Verdict for grip ------------------------------------------------------
    g = results['grip_factor']
    if g['nonlinear_r2'] < 0.5:
        print("[step1] VERDICT: mu itself does not preserve grip cleanly "
              f"(nonlinear R^2={g['nonlinear_r2']:.3f}). Decoding grip from phi "
              "was impossible in principle — report as a mechanism finding.")
    elif g['linear_r2'] < 0.5 <= g['nonlinear_r2']:
        print("[step1] VERDICT: grip is in z but NOT linearly "
              f"(linear {g['linear_r2']:.3f} vs nonlinear {g['nonlinear_r2']:.3f}). "
              "Use the MLP probe, not a linear one, for E5.")
    else:
        print("[step1] VERDICT: grip is linearly decodable from mu(e) "
              f"(R^2={g['linear_r2']:.3f}). The ceiling is fine — the bottleneck "
              "is phi/observability. Proceed to Steps 2-3.")


if __name__ == '__main__':
    main()
