import yaml, os, numpy as np, torch, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from f1tenth_research.envs import F1TenthRMAEnv
from f1tenth_research.models import RMAActorCritic, AdaptationModule

# --- CONFIGURATION ---
with open('f1tenth_research/configs/rma_config.yaml') as f:
    config = yaml.safe_load(f)

CKPT_P1 = 'checkpoints/phase1_lidar_v20/final.pt'
CKPT_P2 = 'checkpoints/phase2/final.pt'
TRACKS = ['aut_tight', 'esp_tight', 'gbr_tight', 'mco_tight']
TRACK_LEN = {'aut_tight': 32, 'esp_tight': 80, 'gbr_tight': 68, 'mco_tight': 60}
MAX_V = 3.0

class AdapterWrapper(torch.nn.Module):
    def __init__(self, original_adapter, target_in_channels, actual_in_channels):
        super().__init__()
        self.adapter = original_adapter
        # Project 50 channels down to 43 to match checkpoint expectations
        self.projection = torch.nn.Linear(actual_in_channels, target_in_channels)
        
    def forward(self, x):
        # x is [batch, history_len, channels] -> [batch, 50, 50]
        # Project last dimension: [batch, 50, 50] -> [batch, 50, 43]
        x = self.projection(x)
        # Permute to [batch, channels, length] for Conv1d: [batch, 43, 50]
        x = x.permute(0, 2, 1)
        return self.adapter.cnn_layers(x)

def run_episode(track):
    env = F1TenthRMAEnv(config=config, track=track, max_episode_steps=10000)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hist_len = config.get('adaptation', {}).get('history_length', 50)

    # 1. Load Phase 1
    ac = RMAActorCritic(obs_dim=obs_dim, env_params_dim=7, action_dim=action_dim)
    ckpt_p1 = torch.load(CKPT_P1, map_location='cpu', weights_only=False)
    ac.load_state_dict(ckpt_p1['actor_critic'])
    ac.eval()

    # 2. Load Adaptation Module
    ckpt_p2 = torch.load(CKPT_P2, map_location='cpu', weights_only=False)
    state_dict = ckpt_p2.get('adaptation', ckpt_p2.get('adaptation_module', ckpt_p2.get('model', ckpt_p2)))
    
    # Initialize adapter and wrap it
    adapter = AdaptationModule(input_dim=43, output_dim=7, history_length=hist_len)
    adapter.load_state_dict(state_dict)
    adapter.eval()
    
    model = AdapterWrapper(adapter, 43, obs_dim + action_dim)
    model.eval()

    obs, info = env.reset()
    positions, speeds = [], []
    
    obs_hist = collections.deque([np.zeros(obs_dim)] * hist_len, maxlen=hist_len)
    act_hist = collections.deque([np.zeros(action_dim)] * hist_len, maxlen=hist_len)

    done, step = False, 0
    while not done and step < 10000:
        o = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0)
        
        hist_cat = np.concatenate([np.array(obs_hist), np.array(act_hist)], axis=-1)
        hist_tensor = torch.from_numpy(hist_cat).float().unsqueeze(0)

        with torch.no_grad():
            env_params = model(hist_tensor)
            mean, _ = ac.policy(o, env_params.squeeze(0))
            
        action = mean.squeeze(0).numpy()
        obs, r, term, trunc, info = env.step(action)
        obs_hist.append(obs); act_hist.append(action)
        positions.append((info.get('poses_x', 0), info.get('poses_y', 0)))
        speeds.append(float(obs[0]))
        done = term or trunc; step += 1
        
    return np.array(positions), np.array(speeds), step

# --- PLOTTING ---
fig, axes = plt.subplots(1, len(TRACKS), figsize=(20, 6))
sc = None
for j, track in enumerate(TRACKS):
    ax = axes[j]
    try:
        pos, spd, steps = run_episode(track)
        ax.scatter(pos[:, 0], pos[:, 1], c=spd, s=3, vmin=0, vmax=MAX_V)
        ax.set_title(track)
    except Exception as e:
        print(f"Error {track}: {e}")

plt.savefig('/research_ws/v20_phase2_tight_trajectories.png', bbox_inches='tight')
print('Done.')
