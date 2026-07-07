"""Compare v19 (Phase 1 only) vs v19 + Phase 2 across all 4 tracks."""
import yaml, os, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from f1tenth_research.envs import F1TenthRMAEnv
from f1tenth_research.models import RMAActorCritic, AdaptationModule
import warnings
warnings.filterwarnings('ignore')

with open('f1tenth_research/configs/rma_config.yaml') as f:
    config = yaml.safe_load(f)

PHASE1_CKPT = 'checkpoints/phase1_lidar_v19/final.pt'
PHASE2_CKPT = 'checkpoints/phase2/final.pt'
TRACKS = ['aut', 'esp', 'gbr', 'mco']
TRACK_LEN = {'aut': 92, 'esp': 230, 'gbr': 195, 'mco': 172}

def run_episode(track, use_adaptation):
    env = F1TenthRMAEnv(config=config, track=track, max_episode_steps=7500)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ac = RMAActorCritic(obs_dim=obs_dim, env_params_dim=7, action_dim=action_dim)
    ckpt = torch.load(PHASE1_CKPT, map_location='cpu', weights_only=False)
    ac.load_state_dict(ckpt['actor_critic'])
    ac.eval()

    adaptation = None
    if use_adaptation:
        adaptation = AdaptationModule(
            state_action_dim=obs_dim + action_dim,
            history_window=10,
            intrinsics_dim=8,
        )
        adapt_ckpt = torch.load(PHASE2_CKPT, map_location='cpu', weights_only=False)
        adaptation.load_state_dict(adapt_ckpt['adaptation'])
        adaptation.eval()

    obs, info = env.reset()
    positions, speeds = [], []
    sa_history = deque(maxlen=10)
    done, step = False, 0
    while not done and step < 7500:
        o = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0)
        with torch.no_grad():
            if adaptation is not None:
                if len(sa_history) < 10:
                    pad = np.zeros(obs.shape[0] + action_dim, dtype=np.float32)
                    padded = list(sa_history) + [pad] * (10 - len(sa_history))
                    hist = np.array(padded, dtype=np.float32)
                else:
                    hist = np.array(list(sa_history), dtype=np.float32)
                hist_t = torch.from_numpy(hist).unsqueeze(0).float()
                intr = adaptation(hist_t)  # keep batch dim to match obs
            else:
                ep = info.get('physics_params', {})
                et = torch.from_numpy(np.array(
                    [ep.get(k, 0) for k in
                     ['grip_factor','mass_scale','inertia_scale',
                      'motor_steering_scale','motor_drive_scale',
                      'delay_steering','delay_drive']]
                )).float().unsqueeze(0)
                intr = ac.get_intrinsics(et)
            mean, _ = ac.policy(o, intr)
        action = mean.squeeze(0).numpy()
        sa_history.append(np.concatenate([obs, action]).astype(np.float32))
        obs, r, term, trunc, info = env.step(action)
        positions.append((info.get('poses_x', 0), info.get('poses_y', 0)))
        speeds.append(float(obs[0]))
        done = term or trunc
        step += 1
    return np.array(positions), np.array(speeds), step

fig, axes = plt.subplots(2, len(TRACKS), figsize=(20, 12))

for j, track in enumerate(TRACKS):
    for i, (label, use_phase2) in enumerate([
        ('v19 (Phase 1 only)', False),
        ('v19 + Phase 2', True),
    ]):
        ax = axes[i, j]
        try:
            positions, speeds, steps = run_episode(track, use_phase2)
            cl = np.loadtxt(f'/research_ws/maps/{track}_centerline.csv', delimiter=',')[:, :2]
            rl = np.loadtxt(f'/research_ws/maps/racelines/{track}_raceline.csv',
                            delimiter=',', skiprows=1)[:, 1:3]
            ax.plot(cl[:, 0], cl[:, 1], '--', color='gray', linewidth=0.7)
            ax.plot(rl[:, 0], rl[:, 1], '-', color='red', linewidth=1.0, alpha=0.5)
            sc = ax.scatter(positions[:, 0], positions[:, 1], c=speeds, cmap='viridis',
                            s=2, vmin=0, vmax=8)
            ax.scatter(positions[0, 0], positions[0, 1], color='green', s=40, marker='o', zorder=5)
            ax.scatter(positions[-1, 0], positions[-1, 1], color='black', s=40, marker='X', zorder=5)
            dist = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            laps_est = dist / TRACK_LEN[track]
            ax.set_title(f'{label} | {track.upper()}\n{steps} steps, {dist:.0f}m, ~{laps_est:.2f} laps',
                         fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            print(f"{label} on {track}: {steps} steps, {dist:.0f}m, {laps_est:.2f} laps")
        except Exception as e:
            ax.text(0.5, 0.5, f'FAILED:\n{str(e)[:60]}', ha='center', va='center',
                    transform=ax.transAxes)
            print(f"FAILED {label} on {track}: {e}")

plt.suptitle('RMA Comparison: Phase 1 only (ground-truth physics) vs Phase 1 + Phase 2 (adapted)\n'
             'Physics randomization ON. Red = optimal raceline. Color = speed (purple→yellow).',
             fontsize=12, y=1.0)
plt.tight_layout()
out = '/research_ws/v19_phase2_comparison.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f'\nSaved to {out}')
