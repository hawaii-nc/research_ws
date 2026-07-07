"""
Does Phase 2 (phi) actually detect physics changes as they happen?

Runs an episode with a mid-episode friction disturbance (already built into
the env's randomizer) and logs phi's ESTIMATED grip_factor at every step
alongside the TRUE grip_factor the simulator is actually using.

If phi is working, its estimate should track/follow the true value,
including dipping when the disturbance hits partway through the episode.
"""
import yaml, numpy as np, torch
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

PHASE1_CKPT = 'checkpoints/phase1_lidar_v28/final.pt'
PHASE2_CKPT = 'checkpoints/phase2/final.pt'
TRACK = 'sepang_tight'
MAX_STEPS = 5000

# Force the mid-episode disturbance ON for this test
config['environment']['randomization']['enabled'] = True
config['environment']['randomization']['mid_episode_disturbance'] = True

env = F1TenthRMAEnv(config=config, track=TRACK, max_episode_steps=MAX_STEPS)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ac = RMAActorCritic(obs_dim=obs_dim, env_params_dim=7, action_dim=action_dim)
ckpt = torch.load(PHASE1_CKPT, map_location='cpu', weights_only=False)
ac.load_state_dict(ckpt['actor_critic'])
ac.eval()

adaptation = AdaptationModule(state_action_dim=obs_dim+action_dim, history_window=10, intrinsics_dim=8)
adapt_ckpt = torch.load(PHASE2_CKPT, map_location='cpu', weights_only=False)
adaptation.load_state_dict(adapt_ckpt['adaptation'])
adaptation.eval()

obs, info = env.reset()
sa_history = deque(maxlen=10)

true_grip = []
est_grip = []      # phi's estimated grip_factor, decoded via nearest-neighbor lookup against mu's known mapping
raw_intrinsics_log = []
steps = []

# To interpret phi's output (a latent z, not directly "grip_factor"), we decode it
# by comparing against mu's encoding of many candidate grip values -- find the
# grip value whose mu-encoding is closest to phi's estimate in latent space.
candidate_grips = np.linspace(0.3, 1.3, 101)
candidate_encodings = []
for g in candidate_grips:
    params = torch.tensor([[g, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]]).float()
    with torch.no_grad():
        z = ac.get_intrinsics(params)
    candidate_encodings.append(z.squeeze(0).numpy())
candidate_encodings = np.array(candidate_encodings)

def decode_grip_from_latent(z):
    """Find which candidate grip value's mu-encoding is closest to z."""
    dists = np.linalg.norm(candidate_encodings - z, axis=1)
    return candidate_grips[np.argmin(dists)]

done, step = False, 0
while not done and step < MAX_STEPS:
    o = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0)
    ep = info.get('physics_params', {})
    true_g = ep.get('grip_factor', 1.0)

    with torch.no_grad():
        if len(sa_history) >= 10:
            hist = np.array(list(sa_history), dtype=np.float32)
            hist_t = torch.from_numpy(hist).unsqueeze(0).float()
            z_est = adaptation(hist_t)
        else:
            # not enough history yet, use nominal
            nominal = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]]).float()
            z_est = ac.get_intrinsics(nominal)
        mean, _ = ac.policy(o, z_est)

    action = mean.squeeze(0).numpy()
    sa_history.append(np.concatenate([obs, action]).astype(np.float32))

    z_np = z_est.squeeze(0).numpy()
    decoded_grip = decode_grip_from_latent(z_np)

    true_grip.append(true_g)
    est_grip.append(decoded_grip)
    raw_intrinsics_log.append(z_np.copy())
    steps.append(step)

    obs, r, term, trunc, info = env.step(action)
    done = term or trunc
    step += 1

true_grip = np.array(true_grip)
est_grip = np.array(est_grip)
steps = np.array(steps)

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(steps, true_grip, label='TRUE grip_factor (ground truth)', color='black', linewidth=2)
ax.plot(steps, est_grip, label='Phase 2 (phi) ESTIMATED grip_factor', color='red', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Timestep')
ax.set_ylabel('Grip Factor')
ax.set_title(f'Does Phase 2 detect friction changes? ({TRACK}, {step} steps)\n'
             f'Correlation: {np.corrcoef(true_grip, est_grip)[0,1]:.3f}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/research_ws/phi_friction_detection_test.png', dpi=120)

print(f"Episode length: {step} steps")
print(f"True grip range: {true_grip.min():.3f} to {true_grip.max():.3f}")
print(f"Estimated grip range: {est_grip.min():.3f} to {est_grip.max():.3f}")
print(f"Correlation (true vs estimated): {np.corrcoef(true_grip, est_grip)[0,1]:.3f}")
print(f"Mean absolute error: {np.mean(np.abs(true_grip - est_grip)):.3f}")
print(f"\nSaved plot to /research_ws/phi_friction_detection_test.png")
