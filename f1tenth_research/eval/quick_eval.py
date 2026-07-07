"""
Quick Phase 1 evaluation: trained policy vs. random baseline.
=============================================================

Loads a checkpoint, runs N episodes using the policy's deterministic
mean action (with TRUE intrinsics zt from get_intrinsics), and reports:
  - Episode length (steps)
  - Total reward
  - Laps completed (from f110_gym's lap_counts)
  - Avg / max centerline deviation (meters, if centerline CSV available)

Centerline CSVs are loaded from /research_ws/maps/{track}_centerline.csv
(format: x, y, w_left, w_right). example_map has no centerline CSV so
deviation is skipped for that track.

Usage:
    python3 -m f1tenth_research.eval.quick_eval \\
        --config f1tenth_research/configs/rma_config.yaml \\
        --checkpoint checkpoints/phase1_lidar/final.pt \\
        --track aut --num_episodes 5
"""

import argparse
import numpy as np
import torch
import yaml
import os

from ..envs import F1TenthRMAEnv
from ..models import RMAActorCritic


def load_centerline(track):
    """Load centerline CSV for a track, returns Nx2 array or None."""
    path = f'/research_ws/maps/{track}_centerline.csv'
    if not os.path.exists(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=',')
        return data[:, 0:2]  # x, y columns
    except Exception:
        return None


def nearest_centerline_dist(x, y, centerline):
    """Compute distance from (x, y) to nearest centerline point."""
    if centerline is None:
        return None
    dx = centerline[:, 0] - x
    dy = centerline[:, 1] - y
    return float(np.min(np.sqrt(dx**2 + dy**2)))


def run_episodes(env, actor_critic, device, num_episodes,
                 centerline=None, random_policy=False,
                 adaptation=None, history_window=100):
    lengths = []
    rewards = []
    laps = []
    avg_devs = []
    max_devs = []

    from collections import deque
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        step_devs = []
        sa_history = deque(maxlen=history_window)

        while not done and step < env.max_episode_steps:
            if random_policy:
                action = env.action_space.sample()
            else:
                obs_tensor = torch.from_numpy(
                    obs if isinstance(obs, np.ndarray) else np.array(obs)
                ).float().to(device)

                with torch.no_grad():
                    if adaptation is not None:
                        if len(sa_history) < history_window:
                            pad_vec = np.zeros(obs.shape[0] + env.action_space.shape[0], dtype=np.float32)
                            padded = list(sa_history) + [pad_vec] * (history_window - len(sa_history))
                            hist_arr = np.array(padded, dtype=np.float32)
                        else:
                            hist_arr = np.array(list(sa_history), dtype=np.float32)
                        hist_tensor = torch.from_numpy(hist_arr).unsqueeze(0).float().to(device)
                        intrinsics = adaptation(hist_tensor).squeeze(0)
                    else:
                        env_params = info.get('physics_params', {})
                        env_params_tensor = torch.from_numpy(
                            np.array([env_params.get(k, 0) for k in
                                      ['grip_factor', 'mass_scale', 'inertia_scale',
                                       'motor_steering_scale', 'motor_drive_scale',
                                       'delay_steering', 'delay_drive']])
                        ).float().to(device)
                        intrinsics = actor_critic.get_intrinsics(env_params_tensor)
                    mean, _ = actor_critic.policy(obs_tensor, intrinsics)
                action = mean.cpu().numpy()
            sa_history.append(np.concatenate([obs, action]).astype(np.float32))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            # Centerline deviation
            if centerline is not None and 'poses_x' in info:
                d = nearest_centerline_dist(info['poses_x'], info['poses_y'],
                                            centerline)
                if d is not None:
                    step_devs.append(d)

        lengths.append(step)
        rewards.append(total_reward)
        laps.append(info.get('lap_counts', 0))
        if step_devs:
            avg_devs.append(float(np.mean(step_devs)))
            max_devs.append(float(np.max(step_devs)))

        lap_str = f"laps={info.get('lap_counts', '?')}"
        dev_str = f", avg_dev={np.mean(step_devs):.3f}m, max_dev={np.max(step_devs):.3f}m" if step_devs else ""
        print(f"  Episode {ep+1}/{num_episodes}: steps={step}, "
              f"reward={total_reward:.3f}, {lap_str}{dev_str}")

    return lengths, rewards, laps, avg_devs, max_devs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--adaptation', type=str, default=None,
                        help='Optional path to Phase 2 adaptation module checkpoint')
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--track', type=str, default='example_map',
                        help="Track to evaluate on (e.g. 'example_map', "
                             "'aut', 'esp', 'gbr', 'mco', 'CornerHall')")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    centerline = load_centerline(args.track)

    env = F1TenthRMAEnv(config=config, track=args.track, max_episode_steps=config.get("environment", {}).get("max_episode_steps", 5000))

    actor_critic = RMAActorCritic(obs_dim=env.observation_space.shape[0]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    actor_critic.load_state_dict(ckpt['actor_critic'])
    actor_critic.eval()

    adaptation = None
    if args.adaptation:
        from ..models import AdaptationModule
        action_dim = env.action_space.shape[0]
        adaptation = AdaptationModule(
            state_action_dim=env.observation_space.shape[0] + action_dim,  # full obs (113D), no raceline lookahead exists
            history_window=100,
            intrinsics_dim=8,
        ).to(device)
        adapt_ckpt = torch.load(args.adaptation, map_location=device, weights_only=False)
        adaptation.load_state_dict(adapt_ckpt['adaptation'])
        adaptation.eval()
        print(f'[Eval] Phase 2 adaptation module loaded from {args.adaptation}')

    if centerline is not None:
        print(f"Centerline loaded: {len(centerline)} points")
    else:
        print(f"No centerline CSV found for '{args.track}' -- skipping deviation metric")

    print(f"\n=== Trained Policy ({args.checkpoint}) ===")
    tl, tr, tlaps, tavg, tmax = run_episodes(
        env, actor_critic, device, args.num_episodes,
        centerline=centerline, random_policy=False,
        adaptation=adaptation
    )

    print(f"\n=== Random Baseline ===")
    rl, rr, rlaps, ravg, rmax = run_episodes(
        env, actor_critic, device, args.num_episodes,
        centerline=centerline, random_policy=True
    )

    print("\n=== Summary ===")
    print(f"Trained policy:  avg_length={np.mean(tl):.1f}, "
          f"avg_reward={np.mean(tr):.3f}, "
          f"avg_laps={np.mean(tlaps):.2f}")
    if tavg:
        print(f"  Centerline:    avg_dev={np.mean(tavg):.3f}m, "
              f"max_dev={np.mean(tmax):.3f}m")
    print(f"Random baseline: avg_length={np.mean(rl):.1f}, "
          f"avg_reward={np.mean(rr):.3f}, "
          f"avg_laps={np.mean(rlaps):.2f}")
    if ravg:
        print(f"  Centerline:    avg_dev={np.mean(ravg):.3f}m, "
              f"max_dev={np.mean(rmax):.3f}m")


if __name__ == '__main__':
    main()
