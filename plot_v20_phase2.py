"""Compare v20 (Phase 1 only) vs v20 + Phase 2 across 4 tight tracks."""
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

PHASE1_CKPT = 'checkpoints/phase1_lidar_v20/final.pt'
PHASE2_CKPT = 'checkpoints/phase2/final.pt'
TRACKS = ['aut_tight', 'esp_tight', 'gbr_tight', 'mco_tight']
TRACK_LEN = {'aut_tight': 32, 'esp_tight': 80, 'gbr_tight': 68, 'mco_tight': 60}
MAX_V = 3.0
HISTORY_WINDOW = 10

def run_episode(track, use_adaptation):
    env = F1TenthRMAEnv(config=config, track=track, max_episode_steps=10000)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ac = RMAActorCritic(obs_dim=obs_dim, env_params_dim=7, action_dim=action_dim)
    ckpt = torch.load(PHASE1_CKPT, map_location='cpu', weights_only=False)
    ac.load_state_dict(ckpt['actor_critic'])
    ac.eval()

    adaptation = None
    if use_adaptation:
        adaptation = AdaptationModule(
            state_action_dim=41 + action_dim,
            history_window=HISTORY_WINDOW,
            intrinsics_dim=8,
        )
        adapt_ckpt = torch.load(PHASE2_CKPT, map_location='cpu', weights_only=False)
        adaptation.load_state_dict(adapt_ckpt['adaptation'])
        adaptation.eval()

    obs, info = env.reset()
    positions, speeds = [], []
    sa_history = deque(maxlen=HISTORY_WINDOW)
    done, step = False, 0

    while not done and step < 10000:
        o = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0)
        with torch.no_grad():
            if adaptation is not None:
                if len(sa_history) < HISTORY_WINDOW:
                    pad = np.zeros(41 + action_dim, dtype=np.float32)
                    padded = list(sa_history) + [pad] * (HISTORY_WINDOW - len(sa_history))
                    hist = np.array(padded, dtype=np.float32)
                else:
                    hist = np.array(list(sa_history), dtype=np.float32)
                hist_t = torch.from_numpy(hist).unsqueeze(0).float()
                intr = adaptation(hist_t)
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
        sa_history.append(np.concatenate([obs[:41], action]).astype(np.float32))
        obs, r, term, trunc, info = env.step(action)
        positions.append((info.get('poses_x', 0), info.get('poses_y', 0)))
        speeds.append(float(obs[0]))
        done = term or trunc
        step += 1
    return np.array(positions), np.array(speeds), step

fig, axes = plt.subplots(2, len(TRACKS), figsize=(20, 12))
MODES = [('v20 (Phase 1 only)', False), ('v20 + Phase 2', True)]

last_sc = None
for i, (label, use_phase2) in enumerate(MODES):
    for j, track in enumerate(TRACKS):
        ax = axes[i, j]
        try:
            positions, speeds, steps = run_episode(track, use_phase2)

            # Load map boundaries
            cl_full = np.loadtxt(f'/research_ws/maps/{track}_centerline.csv', delimiter=',')
            cl = cl_full[:, :2]
            rl = np.loadtxt(f'/research_ws/maps/racelines/{track}_raceline.csv',
                            delimiter=',', skiprows=1)[:, 1:3]

            # Draw walls using normals
            dxy = np.gradient(cl, axis=0)
            norms = np.stack([-dxy[:, 1], dxy[:, 0]], axis=1)
            norms /= np.linalg.norm(norms, axis=1, keepdims=True) + 1e-9
            left = cl + norms * cl_full[:, 2:3]
            right = cl - norms * cl_full[:, 3:4]

            ax.plot(left[:, 0], left[:, 1], '-', color='black', linewidth=0.8, alpha=0.5)
            ax.plot(right[:, 0], right[:, 1], '-', color='black', linewidth=0.8, alpha=0.5)
            ax.plot(cl[:, 0], cl[:, 1], '--', color='gray', linewidth=0.7, alpha=0.4)
            ax.plot(rl[:, 0], rl[:, 1], '-', color='red', linewidth=1.2, alpha=0.5)
            sc = ax.scatter(positions[:, 0], positions[:, 1], c=speeds, cmap='viridis',
                            s=3, vmin=0, vmax=MAX_V)
            ax.scatter(positions[0, 0], positions[0, 1], color='green', s=60,
                       marker='o', zorder=5)
            ax.scatter(positions[-1, 0], positions[-1, 1], color='black', s=60,
                       marker='X', zorder=5)

            dist = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            laps_est = dist / TRACK_LEN[track]
            avg_speed = np.mean(speeds)
            ax.set_title(f'{label} | {track.upper()}\n'
                         f'{steps} steps  |  {dist:.0f}m  |  ~{laps_est:.2f} laps\n'
                         f'avg speed: {avg_speed:.2f} m/s  |  ',
                         fontsize=9)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            last_sc = sc
            print(f"{label} on {track}: {steps} steps, {dist:.1f}m, "
                  f"{laps_est:.2f} laps, avg {avg_speed:.2f} m/s")
        except Exception as e:
            ax.text(0.5, 0.5, f'FAILED:\n{str(e)[:60]}',
                    ha='center', va='center', transform=ax.transAxes)
            print(f"FAILED {label} on {track}: {e}")

if last_sc:
    cbar = fig.colorbar(last_sc, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.02)
    cbar.set_label('Speed (m/s)', fontsize=10)

plt.suptitle('v20 RMA Comparison: Phase 1 only vs Phase 1 + Phase 2\n'
             'Tight tracks (35% scale, 3 m/s max)  |  Physics randomization ON  |  '
             'Red = optimal raceline  |  Black = track walls',
             fontsize=12, y=1.01)

out = '/research_ws/v20_phase2_comparison.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f'\nSaved to {out}')
