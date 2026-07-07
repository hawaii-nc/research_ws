"""
step3_excitation_probe.py — Step 3: the E5 probe, done properly.

Replaces the nearest-neighbor decode with the methodology your Paper 2 doc
already prescribes: a ridge probe from phi's latent zhat to each physics
parameter, with

  - EMA smoothing on zhat (instantaneous latents are always noisy; a ~0.2-1 s
    time constant is the standard treatment),
  - held-out evaluation split by EPISODE,
  - per-parameter R^2 (never a single-axis Pearson r),
  - excitation stratification for grip (the observability mechanism),
  - a matched reference probe from the TRUE z = mu(e) on the SAME rollout
    data — so "how much did phi lose" is separated from "how much was ever
    there" (Step 1 gives the iid ceiling; this gives the on-rollout ceiling).

Consumes the .npz from collect_probe_dataset.py. No env/package model imports.

Run:
    python -m f1tenth_research.probe_eval.step3_excitation_probe \
        --dataset probe_results/probe_dataset_w100.npz \
        --outdir probe_results/step3
"""

import argparse
import json
import os
import numpy as np

from .probe_utils import (PARAM_KEYS, ridge_fit, ridge_predict, r2_score,
                          split_by_episode, ema_per_episode, excitation,
                          stratified_r2)

GRIP_IDX = 0
V_IDX = 0     # obs[0] = current_velocity
YAW_IDX = 4   # obs[4] = yaw_rate


def probe_all_params(X_tr, E_tr, X_te, E_te, lam):
    """Ridge probe from latent X to every physics parameter. Returns dict."""
    out = {}
    for i, key in enumerate(PARAM_KEYS):
        m = ridge_fit(X_tr, E_tr[:, i], lam=lam)
        out[key] = {
            'r2': r2_score(E_te[:, i], ridge_predict(m, X_te)),
            'model': m,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--ema_beta', type=float, default=0.9,
                    help='EMA smoothing on zhat: 0.9 ~ 0.2 s @ 50 Hz, 0.98 ~ 1 s. '
                         '0 disables smoothing.')
    ap.add_argument('--ridge_lam', type=float, default=1.0)
    ap.add_argument('--n_bins', type=int, default=5)
    ap.add_argument('--test_frac', type=float, default=0.2)
    ap.add_argument('--outdir', default='probe_results/step3')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)

    d = np.load(args.dataset, allow_pickle=True)
    valid = d['zhat_valid']            # drop warm-up steps before phi's buffer fills
    obs = d['obs'][valid]
    E = d['e'][valid]
    Z = d['z'][valid]                  # true mu(e) on the same rollouts
    Zh = d['zhat'][valid]
    ep = d['episode_id'][valid]
    window = int(d['window'])
    print(f"[step3] {len(Zh)} valid timesteps (window={window}, "
          f"policy_z={str(d['policy_z'])})")

    # --- EMA smooth phi's latent (per episode) --------------------------------
    Zh_s = ema_per_episode(Zh, ep, beta=args.ema_beta) if args.ema_beta > 0 else Zh

    exc = excitation(obs[:, V_IDX], obs[:, YAW_IDX])
    tr, te = split_by_episode(ep, test_frac=args.test_frac, seed=args.seed)
    print(f"[step3] train {tr.sum()} / test {te.sum()} timesteps")

    # --- probes: phi's latent (raw + smoothed) and true z (reference) ---------
    probes = {
        'zhat_raw': probe_all_params(Zh[tr], E[tr], Zh[te], E[te], args.ridge_lam),
        'zhat_ema': probe_all_params(Zh_s[tr], E[tr], Zh_s[te], E[te], args.ridge_lam),
        'z_true':   probe_all_params(Z[tr], E[tr], Z[te], E[te], args.ridge_lam),
    }

    print(f"\n{'parameter':<24}{'zhat raw':>10}{'zhat EMA':>10}{'true z':>10}")
    print('-' * 54)
    summary = {}
    for key in PARAM_KEYS:
        row = {src: probes[src][key]['r2'] for src in probes}
        summary[key] = row
        print(f"{key:<24}{row['zhat_raw']:>10.4f}{row['zhat_ema']:>10.4f}"
              f"{row['z_true']:>10.4f}")

    # --- grip R^2 stratified by excitation (zhat EMA probe) -------------------
    grip_model = probes['zhat_ema']['grip_factor']['model']
    grip_pred = ridge_predict(grip_model, Zh_s[te])
    grip_true = E[te, GRIP_IDX]
    strat = stratified_r2(grip_true, grip_pred, exc[te], n_bins=args.n_bins)

    print(f"\ngrip R^2 by excitation bin (zhat EMA probe):")
    print(f"{'excitation bin':<28}{'n':>8}{'R^2':>10}")
    print('-' * 46)
    for row in strat:
        print(f"[{row['lo']:6.3f}, {row['hi']:6.3f}) m/s^2 {row['n']:>8}{row['r2']:>10.4f}")

    out = {
        'per_param_r2': summary,
        'grip_stratified_zhat_ema': strat,
        'ema_beta': args.ema_beta,
        'window': window,
        'policy_z': str(d['policy_z']),
    }
    with open(os.path.join(args.outdir, 'step3_probe_results.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # --- figures -----------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(PARAM_KEYS))
    w = 0.26
    axes[0].bar(x - w, [summary[k]['z_true'] for k in PARAM_KEYS], w,
                label=r'true $z=\mu(e)$ (on-rollout ceiling)')
    axes[0].bar(x, [summary[k]['zhat_ema'] for k in PARAM_KEYS], w,
                label=r'$\hat z=\phi$ (EMA)')
    axes[0].bar(x + w, [summary[k]['zhat_raw'] for k in PARAM_KEYS], w,
                label=r'$\hat z=\phi$ (raw)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(PARAM_KEYS, rotation=30, ha='right')
    axes[0].set_ylabel(r'held-out $R^2$')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_title('(a) E5 linear probe per parameter')
    axes[0].legend(fontsize=8)

    centers = [(r['lo'] + r['hi']) / 2 for r in strat]
    axes[1].plot(centers, [r['r2'] for r in strat], 'o-')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_xlabel(r'lateral demand $|v\cdot\dot\psi|$ (m/s$^2$)')
    axes[1].set_ylabel(r'within-bin grip $R^2$')
    axes[1].set_title(r'(b) grip decodability from $\hat z$ vs excitation')
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'step3_probe.png'), dpi=150)
    print(f"\n[step3] results -> {args.outdir}/")

    # --- verdict -------------------------------------------------------------------
    g = summary['grip_factor']
    if g['z_true'] < 0.3:
        print("[step3] VERDICT: even TRUE z can't be linearly probed for grip on "
              "rollout data — the mu latent entangles grip (consistent with a low "
              "Step 1 linear ceiling). Nonlinear probe or system-ID head needed; "
              "the NN decode never had a chance.")
    elif g['zhat_ema'] < 0.3 <= g['z_true']:
        print("[step3] VERDICT: grip is probeable from true z but NOT from phi's "
              "estimate — phi loses the grip information. Combine with Step 2: if "
              "Step 2's supervised net succeeded, retrain phi (excitation-rich "
              "data / direct system-ID auxiliary head); if Step 2 also failed, "
              "it's observability, and the excitation figure is your story.")
    else:
        print(f"[step3] VERDICT: grip IS probeable from phi's latent "
              f"(EMA R^2={g['zhat_ema']:.3f}). The earlier nearest-neighbor decode "
              "was the broken link, not phi. Report this probe as E5.")


if __name__ == '__main__':
    main()
