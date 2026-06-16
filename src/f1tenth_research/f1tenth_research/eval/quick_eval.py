"""
Quick Phase 1 sanity eval: trained policy vs. random baseline.
================================================================

Loads checkpoints/phase1/final.pt, runs N episodes using the policy's
deterministic mean action (with TRUE intrinsics zt from get_intrinsics,
since Phase 2's adaptation module isn't trained yet), and compares
episode length / total reward against a random-action baseline.

Usage:
    python3 -m f1tenth_research.eval.quick_eval --config f1tenth_research/configs/rma_config.yaml --checkpoint checkpoints/phase1/final.pt
"""

import argparse
import numpy as np
import torch
import yaml

from ..envs import F1TenthRMAEnv
from ..models import RMAActorCritic


def run_episodes(env, actor_critic, device, num_episodes, random_policy=False):
    lengths = []
    rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        while not done and step < env.max_episode_steps:
            if random_policy:
                action = env.action_space.sample()
            else:
                obs_tensor = torch.from_numpy(
                    obs if isinstance(obs, np.ndarray) else np.array(obs)
                ).float().to(device)

                env_params = info.get('physics_params', {})
                env_params_tensor = torch.from_numpy(
                    np.array([env_params.get(k, 0) for k in
                              ['grip_factor', 'mass_scale', 'inertia_scale',
                               'motor_steering_scale', 'motor_drive_scale',
                               'delay_steering', 'delay_drive']])
                ).float().to(device)

                with torch.no_grad():
                    intrinsics = actor_critic.get_intrinsics(env_params_tensor)
                    mean, _ = actor_critic.policy(obs_tensor, intrinsics)
                action = mean.cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

        lengths.append(step)
        rewards.append(total_reward)
        print(f"  Episode {ep+1}/{num_episodes}: steps={step}, total_reward={total_reward:.3f}")

    return lengths, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--track', type=str, default='example_map',
                       help="Track to evaluate on (e.g. 'example_map', 'aut', 'esp', 'gbr', 'mco', 'CornerHall')")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)

    env = F1TenthRMAEnv(config=config, track=args.track)

    actor_critic = RMAActorCritic(obs_dim=env.observation_space.shape[0]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    actor_critic.load_state_dict(ckpt['actor_critic'])
    actor_critic.eval()

    print(f"\n=== Trained Policy ({args.checkpoint}) ===")
    trained_lengths, trained_rewards = run_episodes(
        env, actor_critic, device, args.num_episodes, random_policy=False
    )

    print(f"\n=== Random Baseline ===")
    random_lengths, random_rewards = run_episodes(
        env, actor_critic, device, args.num_episodes, random_policy=True
    )

    print("\n=== Summary ===")
    print(f"Trained policy: avg_length={np.mean(trained_lengths):.1f}, avg_reward={np.mean(trained_rewards):.3f}")
    print(f"Random baseline: avg_length={np.mean(random_lengths):.1f}, avg_reward={np.mean(random_rewards):.3f}")


if __name__ == '__main__':
    main()
