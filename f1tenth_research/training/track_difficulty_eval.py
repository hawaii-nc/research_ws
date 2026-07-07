"""
Fast per-track difficulty scoring, run periodically DURING training
(uses f110_gym, not Gazebo -- fast enough to run every few million steps).

Score = 1 - (avg episode length / max_episode_steps), i.e. how much of the
episode budget the car burns through before crashing. Higher score = harder.
This is a proxy for the Gazebo corner-study pass rate, cheap enough to run
inline without disrupting training throughput much.
"""
import numpy as np
import torch


def evaluate_track_difficulty(actor_critic, config, tracks_list, device,
                               episodes_per_track=3, max_steps=3000):
    """
    Returns dict {track_name: difficulty_score in [0,1]}, higher = harder.
    Uses the CURRENT policy with deterministic (mean) actions.
    """
    from ..envs import F1TenthRMAEnv

    difficulty = {}
    actor_critic.eval()

    for track in tracks_list:
        lengths = []
        for _ in range(episodes_per_track):
            env = F1TenthRMAEnv(config=config, track=track, max_episode_steps=max_steps)
            obs, info = env.reset()
            done, step = False, 0
            while not done and step < max_steps:
                o = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0).to(device)
                ep = info.get('physics_params', {})
                et = torch.from_numpy(np.array(
                    [ep.get(k, 0) for k in
                     ['grip_factor', 'mass_scale', 'inertia_scale',
                      'motor_steering_scale', 'motor_drive_scale',
                      'delay_steering', 'delay_drive']]
                )).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    intr = actor_critic.get_intrinsics(et)
                    mean, _ = actor_critic.policy(o, intr)
                action = mean.squeeze(0).cpu().numpy()
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc
                step += 1
            lengths.append(step)

        avg_len = np.mean(lengths)
        difficulty[track] = float(1.0 - min(avg_len / max_steps, 1.0))

    actor_critic.train()
    return difficulty
