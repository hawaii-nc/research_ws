"""
Shared harness for Paper 2 safety experiments E1, E2, E3, E5.

Implements three-controller switching via zhat_mode:
  - 'oracle': ground-truth physics params -> mu(params) -> z  (upper bound)
  - 'phi_rma': adaptation module estimates z from state-action history (the actual system)
  - 'blind': frozen nominal z = mu(nominal_params)  (lower bound / no-adaptation baseline)

Also computes friction-circle utilization (rho) per step:
  rho = sqrt(a_lat^2 + a_long^2) / (mu * g)
This measures how close to the friction limit the car is actually operating --
NOT whether it decodes mu correctly, but whether its behavior respects the
true physical limit. This is the primary safety metric for Paper 2.
"""
import numpy as np
import torch
from collections import deque


GRAVITY = 9.81


class SafetyEvalController:
    """Wraps a Phase1 policy + optional Phase2 adaptation module with
    controllable zhat_mode for the three-controller comparison."""

    def __init__(self, actor_critic, adaptation=None, history_window=10,
                 obs_dim=113, action_dim=2, device='cpu'):
        self.actor_critic = actor_critic
        self.adaptation = adaptation
        self.history_window = history_window
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.sa_history = deque(maxlen=history_window)
        self.nominal_params = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]], device=device
        ).float()

    def reset_history(self):
        self.sa_history.clear()

    def get_z(self, mode: str, obs: np.ndarray, true_physics_params: dict = None):
        """Return the intrinsics vector z according to zhat_mode."""
        with torch.no_grad():
            if mode == 'oracle':
                assert true_physics_params is not None, "oracle mode needs true physics params"
                p = torch.tensor([[
                    true_physics_params.get('grip_factor', 1.0),
                    true_physics_params.get('mass_scale', 1.0),
                    true_physics_params.get('inertia_scale', 1.0),
                    true_physics_params.get('motor_steering_scale', 1.0),
                    true_physics_params.get('motor_drive_scale', 1.0),
                    true_physics_params.get('delay_steering', 0.0),
                    true_physics_params.get('delay_drive', 0.0),
                ]], device=self.device).float()
                z = self.actor_critic.get_intrinsics(p)
            elif mode == 'phi_rma':
                if self.adaptation is None:
                    raise ValueError("phi_rma mode requires an adaptation module")
                if len(self.sa_history) < self.history_window:
                    z = self.actor_critic.get_intrinsics(self.nominal_params)
                else:
                    hist = np.array(list(self.sa_history), dtype=np.float32)
                    hist_t = torch.from_numpy(hist).unsqueeze(0).float().to(self.device)
                    z = self.adaptation(hist_t)
            elif mode == 'blind':
                z = self.actor_critic.get_intrinsics(self.nominal_params)
            else:
                raise ValueError(f"Unknown zhat_mode: {mode}")
        return z

    def act(self, obs: np.ndarray, mode: str, true_physics_params: dict = None):
        o = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0).to(self.device)
        z = self.get_z(mode, obs, true_physics_params)
        with torch.no_grad():
            mean, _ = self.actor_critic.policy(o, z)
        action = mean.squeeze(0).cpu().numpy()
        self.sa_history.append(np.concatenate([obs, action]).astype(np.float32))
        return action, z.squeeze(0).cpu().numpy()


def compute_rho(v, yaw_rate, prev_v, dt, mu, wheelbase=0.313):
    """
    Friction-circle utilization rho = ||a_lateral, a_longitudinal|| / (mu * g)
    rho -> 0: driving far below the limit (overly cautious)
    rho -> 1: driving AT the limit (using full available grip)
    rho > 1: exceeding the modeled limit (about to lose control)

    a_lateral approximated from yaw_rate * v (centripetal accel for bicycle model)
    a_longitudinal approximated from (v - prev_v) / dt
    """
    a_lat = abs(yaw_rate * v)
    a_long = abs((v - prev_v) / dt) if dt > 0 else 0.0
    a_total = np.sqrt(a_lat**2 + a_long**2)
    denom = max(mu * GRAVITY, 1e-6)
    return a_total / denom


def run_episode(env, controller, mode, max_steps=5000, dt=0.02):
    """
    Run one episode with a given zhat_mode. Returns a log dict with
    per-step trajectories needed for E1/E2/E3/E5 analysis.
    """
    obs, info = env.reset()
    controller.reset_history()

    log = {
        'v': [], 'yaw_rate': [], 'rho': [], 'true_mu': [], 'z_est': [],
        'min_lidar': [], 'step': [], 'crashed': False, 'steps_survived': 0,
    }

    prev_v = 0.0
    done, step = False, 0
    while not done and step < max_steps:
        physics = info.get('physics_params', {})
        true_mu = physics.get('grip_factor', 1.0)

        action, z = controller.act(obs, mode, physics)
        obs, r, term, trunc, info = env.step(action)

        v = float(obs[0])
        yaw_rate = float(obs[4])
        rho = compute_rho(v, yaw_rate, prev_v, dt, true_mu)
        min_lidar = float(np.min(obs[5:113])) * 10.0  # denormalize (obs beams are /10)

        log['v'].append(v)
        log['yaw_rate'].append(yaw_rate)
        log['rho'].append(rho)
        log['true_mu'].append(true_mu)
        log['z_est'].append(z.tolist())
        log['min_lidar'].append(min_lidar)
        log['step'].append(step)

        prev_v = v
        done = term or trunc
        step += 1

    log['crashed'] = bool(term) if 'term' in dir() else False
    log['steps_survived'] = step
    return log
