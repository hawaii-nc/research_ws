import yaml, os, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from f1tenth_research.envs import F1TenthRMAEnv
from f1tenth_research.models import RMAActorCritic
import warnings
warnings.filterwarnings('ignore')

with open('f1tenth_research/configs/rma_config.yaml') as f:
    config = yaml.safe_load(f)

CKPT = 'checkpoints/phase1_lidar_v27/final.pt'
TRACKS = ['aut_tight', 'esp_tight', 'gbr_tight', 'mco_tight']
MAX_V = 3.0

def run_episode(track):
    env = F1TenthRMAEnv(config=config, track=track, max_episode_steps=16500)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ac = RMAActorCritic(obs_dim=obs_dim, env_params_dim=7, action_dim=action_dim)
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    ac.load_state_dict(ckpt['actor_critic'])
    ac.eval()
    obs, info = env.reset()
    positions, speeds = [], []
    done, step = False, 0
    while not done and step < 16500:
        o = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0)
        ep = info.get('physics_params', {})
        et = torch.from_numpy(np.array(
            [ep.get(k, 0) for k in
             ['grip_factor','mass_scale','inertia_scale',
              'motor_steering_scale','motor_drive_scale',
              'delay_steering','delay_drive']]
        )).float().unsqueeze(0)
        with torch.no_grad():
            intr = ac.get_intrinsics(et)
            mean, _ = ac.policy(o, intr)
        action = mean.squeeze(0).numpy()
        obs, r, term, trunc, info = env.step(action)
        positions.append((info.get('poses_x', 0), info.get('poses_y', 0)))
        speeds.append(float(obs[0]))
        done = term or trunc
        step += 1
    return np.array(positions), np.array(speeds), step

fig, axes = plt.subplots(1, len(TRACKS), figsize=(22, 6))

for j, track in enumerate(TRACKS):
    ax = axes[j]
    try:
        positions, speeds, steps = run_episode(track)
        cl_full = np.loadtxt(f'/research_ws/maps/{track}_centerline.csv', delimiter=',')
        rl = np.loadtxt(f'/research_ws/maps/racelines/{track}_raceline.csv',
                        delimiter=',', skiprows=1)[:, 1:3]
        dxy = np.gradient(cl_full[:, :2], axis=0)
        norms = np.stack([-dxy[:, 1], dxy[:, 0]], axis=1)
        norms /= np.linalg.norm(norms, axis=1, keepdims=True) + 1e-9
        left = cl_full[:, :2] + norms * cl_full[:, 2:3]
        right = cl_full[:, :2] - norms * cl_full[:, 3:4]
        ax.plot(left[:, 0], left[:, 1], '-', color='black', linewidth=0.8, alpha=0.5)
        ax.plot(right[:, 0], right[:, 1], '-', color='black', linewidth=0.8, alpha=0.5)
        ax.plot(rl[:, 0], rl[:, 1], '-', color='red', linewidth=1.0, alpha=0.5)
        sc = ax.scatter(positions[:, 0], positions[:, 1], c=speeds, cmap='viridis', s=2, vmin=0, vmax=MAX_V)
        ax.scatter(positions[0, 0], positions[0, 1], color='green', s=50, marker='o', zorder=5)
        ax.scatter(positions[-1, 0], positions[-1, 1], color='black', s=50, marker='X', zorder=5)
        dist = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        ax.set_title(f'{track.upper()}\n{steps} steps | {dist:.0f}m', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        print(f"{track}: {steps} steps, {dist:.1f}m")
    except Exception as e:
        ax.text(0.5, 0.5, f'FAILED:\n{str(e)[:60]}', ha='center', va='center', transform=ax.transAxes)

cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.03, pad=0.05)
cbar.set_label('Speed (m/s)')
plt.suptitle('v27: Real-car-scale policy, best run to date', fontsize=12, y=1.02)
plt.savefig('/research_ws/v27_trajectories.png', dpi=110, bbox_inches='tight')
print('Saved to /research_ws/v27_trajectories.png')
