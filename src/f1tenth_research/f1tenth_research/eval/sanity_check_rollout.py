"""Short Phase 1 PPO+IL sanity check.

Runs a brief rollout and one update step using the real Phase1Trainer so you
can catch broken sampling, log-prob, entropy, or PPO-update behavior quickly.
"""

import argparse

import numpy as np
import torch
import yaml

from ..training.phase1_ppo_il import Phase1Trainer


def _format_array_stats(name: str, values: np.ndarray) -> str:
    return (
        f"{name}: min={values.min():.6f}, max={values.max():.6f}, "
        f"mean={values.mean():.6f}, std={values.std():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 PPO+IL sanity check")
    parser.add_argument("--config", type=str, required=True, help="Path to rma_config.yaml")
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=100,
        help="Short rollout length for the sanity check",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for the trainer (default: cpu)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    # Keep this lightweight so it does not clutter the main training logs.
    trainer = Phase1Trainer(
        config,
        device=args.device,
        log_dir="logs/sanity_check",
        checkpoint_dir="checkpoints/sanity_check",
    )

    print("=" * 70)
    print("STAGE 1: Short rollout")
    print("=" * 70)

    rollout_data = trainer.rollout(args.rollout_steps)

    actions = np.array(rollout_data["actions"])
    log_probs = np.array(rollout_data["log_probs"])
    values = np.array(rollout_data["values"])
    rewards = np.array(rollout_data["rewards"])
    dones = np.array(rollout_data["dones"])
    advantages = np.array(rollout_data["advantages"])
    returns = np.array(rollout_data["returns"])

    print(f"Collected {len(actions)} steps")
    print(f"Episode endings (done=1): {np.where(dones == 1)[0].tolist()}")
    print()

    print("--- Actions ---")
    print(f"First 5 actions:\n{actions[:5]}")
    print(_format_array_stats("Action std across steps", actions.std(axis=0)))
    if np.all(actions.std(axis=0) < 1e-4):
        print("  !! WARNING: actions are nearly constant across steps.")
    else:
        print("  OK: actions vary across steps.")
    print(f"Action range: min={actions.min():.4f}, max={actions.max():.4f}")
    print()

    print("--- Log-probs ---")
    print(f"First 5 log_probs: {log_probs[:5]}")
    print(
        f"log_prob range: min={log_probs.min():.4f}, max={log_probs.max():.4f}, "
        f"mean={log_probs.mean():.4f}"
    )
    if not np.all(np.isfinite(log_probs)):
        print("  !! WARNING: non-finite log_probs detected.")
    else:
        print("  OK: all log_probs finite.")
    print()

    print("--- Values, rewards, advantages, returns ---")
    print(f"Values:     min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
    print(f"Rewards:    min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}")
    print(
        f"Advantages: min={advantages.min():.4f}, max={advantages.max():.4f}, "
        f"mean={advantages.mean():.4f}, std={advantages.std():.4f}"
    )
    print(f"Returns:    min={returns.min():.4f}, max={returns.max():.4f}, mean={returns.mean():.4f}")
    for name, arr in [("values", values), ("rewards", rewards), ("advantages", advantages), ("returns", returns)]:
        if not np.all(np.isfinite(arr)):
            print(f"  !! WARNING: non-finite values in {name}")
    if np.allclose(advantages, 0.0):
        print("  !! WARNING: advantages are all ~0.")
    else:
        print("  OK: advantages have non-trivial spread.")
    print()

    print("--- Entropy check (recomputed from current policy) ---")
    obs_tensor = torch.from_numpy(np.array(rollout_data["obs"][:5])).float().to(trainer.device)
    intrinsics_tensor = torch.from_numpy(np.array(rollout_data["intrinsics"][:5])).float().to(trainer.device)
    with torch.no_grad():
        mean, log_std = trainer.actor_critic.policy(obs_tensor, intrinsics_tensor)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1)
    print(f"log_std: {log_std.detach().cpu().numpy()}")
    print(f"std:     {std.detach().cpu().numpy()}")
    print(f"Entropy for first 5 obs: {entropy.detach().cpu().numpy()}")
    if torch.any(entropy < 0.1):
        print("  !! WARNING: entropy is very low after a short rollout.")
    else:
        print("  OK: entropy looks reasonable.")
    print()

    print("=" * 70)
    print("STAGE 2: One update() step")
    print("=" * 70)

    before = trainer.actor_critic.policy.network[0].weight.detach().clone()
    trainer.update(rollout_data, epoch=0)
    after = trainer.actor_critic.policy.network[0].weight.detach().clone()

    weight_change = (after - before).abs().mean().item()
    print(f"Mean abs weight change in policy's first layer: {weight_change:.8f}")
    if weight_change < 1e-10:
        print("  !! WARNING: weights barely changed.")
    else:
        print("  OK: weights updated.")
    print()

    print("--- Metrics from update() ---")
    for key, vals in trainer.metrics.items():
        if len(vals) > 0:
            print(f"{key}: mean={np.mean(vals):.6f}, min={np.min(vals):.6f}, max={np.max(vals):.6f}")

    clip_fracs = trainer.metrics.get("clip_fraction", [])
    approx_kls = trainer.metrics.get("approx_kl", [])
    if len(clip_fracs) > 0:
        if np.mean(clip_fracs) > 0.8:
            print("  !! WARNING: clip_fraction very high.")
        else:
            print("  OK: clip_fraction in a reasonable range.")
    if len(approx_kls) > 0:
        if np.mean(np.abs(approx_kls)) > 0.1:
            print("  !! WARNING: approx_kl is large for a single update.")
        else:
            print("  OK: approx_kl looks small.")

    print()
    print("=" * 70)
    print("Sanity check complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()