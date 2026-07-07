"""
collect_probe_dataset.py — Step 0: collect the shared dataset for Steps 2 & 3.

Rolls out the trained Phase 1 policy in f110_gym with domain randomization
(and mid-episode disturbances, so grip varies WITHIN episodes too) and logs,
per timestep:

    obs        (obs_dim,)  raw policy observation
    act        (2,)        action actually taken
    e          (7,)        TRUE physics params (grip, mass, ..., delays)
    z          (8,)        mu(e) — the Phase-2 regression target
    zhat       (8,)        phi(history) — live adaptation estimate
    zhat_valid (bool)      False until the history buffer is full
    episode_id (int)

Everything is saved to a single .npz. Step 1 does NOT need this dataset
(it samples e synthetically); Steps 2 and 3 both consume it.

Run as a module from the repo root (mirrors your other training scripts):

    python -m f1tenth_research.probe_eval.collect_probe_dataset \
        --actor_critic checkpoints/phase1_lidar/final.pt \
        --phase2 checkpoints/phase2/adaptation_100step.pt \
        --config f1tenth_research/configs/rma_config.yaml \
        --episodes 150 --window 100 \
        --out probe_results/probe_dataset_w100.npz

============================ INTEGRATION POINTS ==============================
Two small APIs vary across your checkpoint versions. Both are isolated in the
clearly-marked functions below so any fix is a one-line change:

  (1) get_encoder(actor_critic)   -> the intrinsics encoder mu module
  (2) get_action(actor_critic, .) -> deterministic action from (obs, z)

If either raises, the error message tells you exactly what to edit.
==============================================================================
"""

import argparse
import os
import numpy as np
import torch
import yaml
from collections import deque

from ..models import RMAActorCritic, AdaptationModule
from ..envs import F1TenthRMAEnv
from .probe_utils import PARAM_KEYS


# ---------------------------------------------------------------------------
# INTEGRATION POINT (1): locate the intrinsics encoder mu inside the AC model
# ---------------------------------------------------------------------------
def get_encoder(actor_critic):
    """Return the mu module (7-D physics params -> 8-D latent z)."""
    for name in ('intrinsics_encoder', 'encoder', 'mu', 'env_encoder'):
        if hasattr(actor_critic, name):
            return getattr(actor_critic, name)
    raise AttributeError(
        "Could not find the intrinsics encoder on RMAActorCritic. "
        "Edit get_encoder() in collect_probe_dataset.py to return the correct "
        "attribute (check models/__init__.py for how RMAActorCritic composes mu)."
    )


# ---------------------------------------------------------------------------
# INTEGRATION POINT (2): deterministic action from (obs, z)
# ---------------------------------------------------------------------------
def get_action(actor_critic, obs_t: torch.Tensor, z_t: torch.Tensor) -> np.ndarray:
    """Return a deterministic (mean) action as a numpy array of shape (2,)."""
    # Try a dedicated act() API first
    if hasattr(actor_critic, 'act'):
        try:
            out = actor_critic.act(obs_t, z_t, deterministic=True)
            a = out[0] if isinstance(out, (tuple, list)) else out
            return a.detach().cpu().numpy().reshape(-1)
        except TypeError:
            pass
    # Fall back to calling the Gaussian policy head directly (mean action)
    if hasattr(actor_critic, 'policy'):
        mean, _log_std = actor_critic.policy(obs_t, z_t)
        return mean.detach().cpu().numpy().reshape(-1)
    raise AttributeError(
        "Could not compute an action from RMAActorCritic. "
        "Edit get_action() in collect_probe_dataset.py to match your model's "
        "forward/act signature."
    )


def step_env(env, action):
    """Gym/Gymnasium compatibility: normalize step() to (obs, done, info)."""
    out = env.step(action)
    if len(out) == 5:                     # gymnasium: obs, r, term, trunc, info
        obs, _r, term, trunc, info = out
        return obs, bool(term or trunc), info
    obs, _r, done, info = out             # classic gym: obs, r, done, info
    return obs, bool(done), info


def params_to_vec(pdict: dict, last: np.ndarray) -> np.ndarray:
    """Convert the env's physics_params dict to the canonical 7-D vector.
    Falls back to the last known value if the env omits the dict this step
    (keeps mid-episode disturbance updates when they ARE reported)."""
    if not pdict:
        return last
    return np.array([float(pdict.get(k, last[i]))
                     for i, k in enumerate(PARAM_KEYS)], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actor_critic', required=True, help='Phase 1 checkpoint (.pt)')
    ap.add_argument('--phase2', required=True, help='Phase 2 phi checkpoint (.pt)')
    ap.add_argument('--config', required=True, help='rma_config.yaml')
    ap.add_argument('--episodes', type=int, default=150)
    ap.add_argument('--max_steps', type=int, default=3000, help='per-episode cap')
    ap.add_argument('--window', type=int, default=100, help='phi history window k')
    ap.add_argument('--policy_z', choices=['oracle', 'live'], default='oracle',
                    help="What z the POLICY is driven with during collection. "
                         "'oracle' (mu(e)) gives clean on-distribution driving; "
                         "'live' (phi) matches deployment closed-loop behavior. "
                         "Collect both if you want the distribution-shift comparison.")
    ap.add_argument('--out', required=True, help='output .npz path')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- Environment (same construction path as your training scripts) ------
    env = F1TenthRMAEnv(config)
    obs_dim = env.observation_space.shape[0]
    print(f"[collect] obs_dim={obs_dim}")

    # --- Phase 1 actor-critic + encoder mu -----------------------------------
    actor_critic = RMAActorCritic(obs_dim=obs_dim).to(device)
    ckpt = torch.load(args.actor_critic, map_location=device)
    actor_critic.load_state_dict(ckpt['actor_critic'])
    actor_critic.eval()
    encoder = get_encoder(actor_critic)

    # --- Phase 2 adaptation module phi ---------------------------------------
    ncfg = config.get('networks', {}).get('adaptation', {})
    state_action_dim = ncfg.get('state_action_dim', obs_dim + 2)
    adaptation = AdaptationModule(
        state_action_dim=state_action_dim,
        history_window=args.window,
        intrinsics_dim=ncfg.get('intrinsics_dim', 8),
    ).to(device)
    ckpt2 = torch.load(args.phase2, map_location=device)
    adaptation.load_state_dict(ckpt2['adaptation'])
    adaptation.eval()
    # How many obs dims phi consumes per step (state_action_dim - action_dim).
    # Handles both the obs[:41] era and the full-obs era automatically.
    phi_obs_dim = state_action_dim - 2
    print(f"[collect] phi: window={args.window}, state_action_dim={state_action_dim} "
          f"(obs slice [:{phi_obs_dim}] + action)")

    # --- Storage --------------------------------------------------------------
    L_obs, L_act, L_e, L_z, L_zhat, L_valid, L_ep = [], [], [], [], [], [], []

    with torch.no_grad():
        for ep in range(args.episodes):
            reset_out = env.reset(options={'training': True})
            obs, info = reset_out if isinstance(reset_out, tuple) else (reset_out, {})
            obs = np.asarray(obs, dtype=np.float32)

            e_vec = params_to_vec(info.get('physics_params', {}),
                                  np.ones(len(PARAM_KEYS), dtype=np.float32))
            history = deque(maxlen=args.window)
            done, step = False, 0

            while not done and step < args.max_steps:
                obs_t = torch.from_numpy(obs).float().to(device)

                # true z = mu(e) — the Phase-2 target, logged every step
                e_t = torch.from_numpy(e_vec).float().to(device)
                z_t = encoder(e_t)

                # phi estimate from history (valid once the buffer is full)
                if len(history) == args.window:
                    h = torch.from_numpy(
                        np.stack(history, axis=0)[None]  # (1, k, F)
                    ).float().to(device)
                    zhat_t = adaptation(h).reshape(-1)
                    zhat_valid = True
                else:
                    zhat_t = torch.zeros_like(z_t).reshape(-1)
                    zhat_valid = False

                # choose which z drives the policy this run
                z_for_policy = zhat_t if (args.policy_z == 'live' and zhat_valid) else z_t
                action = get_action(actor_critic, obs_t, z_for_policy)

                # log BEFORE stepping (obs/e/z/zhat all describe time t)
                L_obs.append(obs.copy())
                L_act.append(action.astype(np.float32))
                L_e.append(e_vec.copy())
                L_z.append(z_t.detach().cpu().numpy().reshape(-1))
                L_zhat.append(zhat_t.detach().cpu().numpy().reshape(-1))
                L_valid.append(zhat_valid)
                L_ep.append(ep)

                # update phi's history with THIS step's (obs slice, action)
                history.append(np.concatenate(
                    [obs[:phi_obs_dim], action]).astype(np.float32))

                obs, done, info = step_env(env, action)
                obs = np.asarray(obs, dtype=np.float32)
                # pick up mid-episode disturbances (grip drop) when reported
                e_vec = params_to_vec(info.get('physics_params', {}), e_vec)
                step += 1

            print(f"[collect] episode {ep + 1}/{args.episodes}: {step} steps "
                  f"(grip start->end: {L_e[-step][0]:.2f} -> {e_vec[0]:.2f})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        obs=np.asarray(L_obs, dtype=np.float32),
        act=np.asarray(L_act, dtype=np.float32),
        e=np.asarray(L_e, dtype=np.float32),
        z=np.asarray(L_z, dtype=np.float32),
        zhat=np.asarray(L_zhat, dtype=np.float32),
        zhat_valid=np.asarray(L_valid, dtype=bool),
        episode_id=np.asarray(L_ep, dtype=np.int64),
        param_keys=np.asarray(PARAM_KEYS),
        window=args.window,
        policy_z=args.policy_z,
    )
    n = len(L_obs)
    print(f"[collect] saved {n} timesteps from {args.episodes} episodes -> {args.out}")


if __name__ == '__main__':
    main()
