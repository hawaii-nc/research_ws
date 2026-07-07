"""
step2_observability.py — Step 2: is grip OBSERVABLE from history at all?

Forget phi entirely. Train a FRESH network with full supervision to predict
grip_factor directly from the same state-action history phi sees. This is the
single most informative experiment in the pipeline:

  - If this net CANNOT predict grip (held-out R^2 ~ 0), the information is not
    in the signal. No phi architecture change (dual windows, bigger CNN, more
    epochs) can fix that — the fix is upstream: excitation-rich data,
    slip-relevant features, or randomized disturbance magnitudes.
  - If this net CAN predict grip but phi's latent can't be probed for it
    (Step 3), the bottleneck is phi's training target (mu's latent), not
    observability.

The excitation-stratified R^2 is the mechanism figure: grip should be
predictable exactly when lateral demand is high (car near the friction limit)
and unpredictable when cruising — that is physics, not model failure.

Consumes the .npz from collect_probe_dataset.py. No env or package imports —
runs anywhere with numpy/torch/matplotlib.

Run:
    python -m f1tenth_research.probe_eval.step2_observability \
        --dataset probe_results/probe_dataset_w100.npz \
        --window 100 --outdir probe_results/step2
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn

from .probe_utils import (build_histories, split_by_episode, r2_score,
                          excitation, stratified_r2)

GRIP_IDX = 0        # grip_factor is index 0 of the canonical 7-D param vector
V_IDX = 0           # obs[0] = current_velocity   (base-state layout)
YAW_IDX = 4         # obs[4] = yaw_rate


class GripNet(nn.Module):
    """
    Fresh supervised regressor: (window, F) history -> grip scalar.
    Mirrors phi's 1D-CNN family but sized so the SAME net works for both
    window=10 and window=100 (padding-preserving convs + adaptive pooling).
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1, stride=2), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):                 # x: (B, window, F)
        h = self.conv(x.transpose(1, 2))  # -> (B, 32, 1)
        return self.head(h.squeeze(-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='.npz from collect_probe_dataset')
    ap.add_argument('--window', type=int, default=100)
    ap.add_argument('--add_alat', action='store_true', default=True,
                    help='append v*yaw_rate as an extra feature channel (default on)')
    ap.add_argument('--no_alat', dest='add_alat', action='store_false',
                    help='ablation: raw history only, no derived slip feature')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--n_bins', type=int, default=5)
    ap.add_argument('--test_frac', type=float, default=0.2)
    ap.add_argument('--outdir', default='probe_results/step2')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.dataset, allow_pickle=True)
    obs, act = d['obs'], d['act']
    grip = d['e'][:, GRIP_IDX]
    ep_id = d['episode_id']

    # --- feature matrix per timestep: (obs, act[, a_lat]) --------------------
    feats = [obs, act]
    if args.add_alat:
        a_lat = (obs[:, V_IDX] * obs[:, YAW_IDX]).reshape(-1, 1)
        feats.append(a_lat.astype(np.float32))
    F = np.concatenate(feats, axis=1)

    # standardize features globally (helps CNN training; stats saved for reuse)
    f_mu, f_sd = F.mean(0), F.std(0) + 1e-8
    F = (F - f_mu) / f_sd

    # --- histories + aligned targets/excitation ------------------------------
    H, tidx = build_histories(F, ep_id, args.window)
    y = grip[tidx]
    exc = excitation(obs[tidx, V_IDX], obs[tidx, YAW_IDX])
    ep_h = ep_id[tidx]
    print(f"[step2] {len(H)} histories (window={args.window}, feat_dim={H.shape[2]}), "
          f"grip range [{y.min():.2f}, {y.max():.2f}]")

    # --- split by EPISODE (never by timestep) ---------------------------------
    tr, te = split_by_episode(ep_h, test_frac=args.test_frac, seed=args.seed)
    print(f"[step2] train {tr.sum()} / test {te.sum()} timesteps "
          f"({len(np.unique(ep_h[tr]))} / {len(np.unique(ep_h[te]))} episodes)")

    # --- train the fresh supervised regressor --------------------------------
    model = GripNet(H.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    Htr = torch.from_numpy(H[tr]).float()
    ytr = torch.from_numpy(y[tr]).float()
    n = len(Htr)
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = Htr[idx].to(device)
            yb = ytr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[step2] epoch {epoch + 1}/{args.epochs}  mse={tot / max(nb, 1):.6f}")

    # --- evaluate on held-out episodes ----------------------------------------
    model.eval()

    def batched_predict(Hsplit):
        out = []
        with torch.no_grad():
            Ht = torch.from_numpy(Hsplit).float()
            for i in range(0, len(Ht), 4096):
                out.append(model(Ht[i:i + 4096].to(device)).cpu().numpy())
        return np.concatenate(out)

    yp = batched_predict(H[te])
    yt = y[te]
    train_r2 = r2_score(y[tr], batched_predict(H[tr]))

    overall = r2_score(yt, yp)
    strat = stratified_r2(yt, yp, exc[te], n_bins=args.n_bins)

    print(f"\n[step2] TRAIN R^2: {train_r2:.4f}   (must be high for a failure "
          f"verdict to mean anything — low train R^2 = underfit, not unobservable)")
    print(f"[step2] OVERALL held-out R^2 (history -> grip): {overall:.4f}")
    print(f"{'excitation bin':<28}{'n':>8}{'R^2':>10}")
    print('-' * 46)
    for row in strat:
        print(f"[{row['lo']:6.3f}, {row['hi']:6.3f}) m/s^2 {row['n']:>8}{row['r2']:>10.4f}")

    with open(os.path.join(args.outdir, 'step2_observability.json'), 'w') as f:
        json.dump({'overall_r2': overall, 'train_r2': train_r2,
                   'stratified': strat,
                   'window': args.window, 'add_alat': bool(args.add_alat),
                   'n_train': int(tr.sum()), 'n_test': int(te.sum())}, f, indent=2)

    # --- figures ---------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # (a) predicted vs true grip
    axes[0].scatter(yt, yp, s=2, alpha=0.15)
    lo, hi = yt.min(), yt.max()
    axes[0].plot([lo, hi], [lo, hi], 'k--', lw=1)
    axes[0].set_xlabel('true grip')
    axes[0].set_ylabel('predicted grip')
    axes[0].set_title(f'(a) fresh supervised net — held-out $R^2$={overall:.3f}')
    # (b) R^2 vs excitation bin — the observability mechanism figure
    centers = [(r['lo'] + r['hi']) / 2 for r in strat]
    r2s = [r['r2'] for r in strat]
    axes[1].plot(centers, r2s, 'o-')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_xlabel(r'lateral demand $|v\cdot\dot\psi|$ (m/s$^2$)')
    axes[1].set_ylabel(r'within-bin $R^2$')
    axes[1].set_title('(b) grip observability vs excitation')
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'step2_observability.png'), dpi=150)
    print(f"[step2] results -> {args.outdir}/")

    # --- verdict -----------------------------------------------------------------
    hi_bin = next((r['r2'] for r in reversed(strat) if np.isfinite(r['r2'])), np.nan)
    if train_r2 < 0.5 and overall < 0.2:
        print("[step2] VERDICT: INCONCLUSIVE — the net is UNDERFIT "
              f"(train R^2={train_r2:.3f}). Re-run with more --epochs / higher "
              "--lr before drawing any observability conclusion; a failure "
              "verdict is only valid once the training fit is good.")
    elif overall < 0.2 and (not np.isfinite(hi_bin) or hi_bin < 0.3):
        print("[step2] VERDICT: grip is NOT observable from history as collected — "
              "even a fully supervised net fails. Fix the DATA (excitation-rich "
              "rollouts, randomized disturbance magnitudes), not phi.")
    elif overall < 0.4 and np.isfinite(hi_bin) and hi_bin > overall + 0.15:
        print("[step2] VERDICT: grip is observable ONLY under high lateral demand — "
              "the excitation-gated story. phi's 'noise' at low excitation is "
              "expected physics; report the stratified figure as the mechanism.")
    else:
        print("[step2] VERDICT: grip IS observable from history "
              f"(R^2={overall:.3f}). If Step 3 still shows phi's latent can't be "
              "probed for grip, the bottleneck is the mu-latent target, and a "
              "direct system-ID head becomes the justified fix.")


if __name__ == '__main__':
    main()
