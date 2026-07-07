"""
F1Tenth Gymnasium Environment Wrapper for RMA
==============================================

Wraps f1tenth_gym simulator with domain randomization, reward function, and
state/action space specifications for Zhang et al. (2025) RMA training.

Observation space (xt):
  - current_velocity: longitudinal velocity (m/s)
  - current_steering_angle: steering angle command (rad)
  - desired_velocity: velocity setpoint (m/s)
  - desired_steering_angle: steering setpoint (rad)
  - yaw_rate: angular velocity around z-axis (rad/s)
  - [additional sensors as needed: lidar, IMU, etc.]

Action space (at):
  - steering_angle_cmd: steering command (rad)
  - throttle_cmd: throttle/velocity command

Returns:
  - observation: state vector xt
  - reward: composite reward signal (Zhang Section II-C)
  - terminated: episode end flag
  - truncated: time limit flag
  - info: diagnostics (randomized params, reward breakdown)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import warnings

try:
    from .real_env import RealF110Wrapper
    F1TENTH_GYM_AVAILABLE = True
except ImportError:
    F1TENTH_GYM_AVAILABLE = False
    warnings.warn(
        "f110_gym not available - using mock environment."
    )

from .randomization import PhysicsRandomizer, SampleMode
from .reward import RewardComputer


class F1TenthRMAEnv(gym.Env):
    """
    F1Tenth Gymnasium environment for RMA training.
    
    Integrates:
    - Domain randomization (physics parameter sampling)
    - Composite reward function
    - State/action spaces
    - Episode termination logic
    
    Reference: Zhang et al. (2025) Section II - System Setup and Randomization
    """
    
    metadata = {'render_modes': [None, 'human']}
    
    def __init__(
        self,
        config: Dict = None,
        render_mode: str = None,
        track: str = 'aut',
        tracks: list = None,
        track_weights: dict = None,  # optional {track_name: weight} for prioritized sampling
        max_episode_steps: int = 1000,
    ):
        """
        Initialize F1Tenth RMA environment.
        
        Args:
            config: Dictionary with environment hyperparameters
            render_mode: Rendering mode ('human' or None)
            track: Track name ('aut' for aut_centerline.csv)
            max_episode_steps: Maximum steps per episode
        """
        super().__init__()
        
        # config is the full YAML; environment-specific settings (reward,
        # randomization, observation) live under config['environment'].
        # Use that sub-dict as self.config so existing self.config.get(...)
        # calls throughout this class resolve correctly.
        full_config = config or {}
        self.config = full_config.get('environment', None) or self._default_config()
        self.render_mode = render_mode
        self.tracks_list = tracks if tracks else None
        if self.tracks_list:
            self.track_weights = track_weights or {t: 1.0 for t in self.tracks_list}
        else:
            self.track_weights = None
        self.track = track if not self.tracks_list else self.tracks_list[0]
        self.max_episode_steps = max_episode_steps
        self._map_cache = {}  # cached centerline+raceline per map
        
        # Initialize components
        self.randomizer = PhysicsRandomizer(self.config.get('randomization', {}))
        self.reward_computer = RewardComputer(self.config.get('reward', {}))

        # Track-specific data loaded by _load_track_data
        self._centerline_loaded = False
        self._raceline = None
        self._raceline_s = None
        self._raceline_speeds = None
        self._raceline_points = None
        self._raceline_total_length = None
        import os
        self._load_track_data(self.track, init_base_env=True)
        # State-action space definition
        self._setup_spaces()
        
        # Episode state
        self.current_physics_params = None
        self.current_il_weight = 1.0  # updated by trainer; controls lookahead dropout
        self.prev_action = None
        self.current_state = None
        self.episode_step = 0
        self.total_episode_reward = 0.0
    
    def _resolve_map_name(self, track: str) -> str:
        """
        Map a track identifier to its f110_gym map path (no extension --
        F110Env appends .png/.yaml itself).

        'example_map' is special-cased to f1tenth_gym's bundled example,
        used for the original Phase 1 validation run. All other track
        names (aut, esp, gbr, mco, CornerHall) resolve to the
        BDEvan5-based benchmark maps copied into /research_ws/maps/.
        """
        if track == 'example_map':
            return '/f1tenth_gym/examples/example_map'
        return f'/research_ws/maps/{track}'

    def _default_config(self) -> Dict:
        """Default environment configuration."""
        return {
            'randomization': {},
            'reward': {},
            'observation': {
                'include_lidar': False,  # LiDAR observations (if available)
                'lidar_beams': 36,
                'include_imu': True,  # IMU (yaw rate)
                'normalize_obs': True,
            },
        }
    

    def set_track_weights(self, weights: dict, floor: float = 0.05):
        """Update per-track sampling weights. Applies a minimum floor so no
        track is ever fully excluded (prevents catastrophic forgetting)."""
        if not self.tracks_list:
            return
        floored = {t: max(weights.get(t, floor), floor) for t in self.tracks_list}
        total = sum(floored.values())
        self.track_weights = {t: w / total for t, w in floored.items()}

    def _load_track_data(self, track_name: str, init_base_env: bool = False):
        """Load centerline + raceline for a given track. Recreates base_env on switch."""
        import os
        self.track = track_name

        # Centerline
        centerline_path = f'/research_ws/maps/{track_name}_centerline.csv'
        if os.path.exists(centerline_path):
            try:
                cl = np.loadtxt(centerline_path, delimiter=',')
                self.reward_computer.set_centerline(cl[:, 0:2])
                self._centerline_loaded = True
            except Exception:
                self._centerline_loaded = False
        else:
            self._centerline_loaded = False

        # Raceline
        self._raceline = None
        self._raceline_s = None
        self._raceline_speeds = None
        self._raceline_points = None
        self._raceline_total_length = None
        raceline_path = f'/research_ws/maps/racelines/{track_name}_raceline.csv'
        if os.path.exists(raceline_path):
            try:
                rl = np.loadtxt(raceline_path, delimiter=',', skiprows=1)
                rl_points = rl[:, 1:3]
                rl_speeds = rl[:, 5]
                self._raceline = rl
                self._raceline_s = rl[:, 0]
                self._raceline_speeds = rl_speeds
                self._raceline_points = rl_points
                self._raceline_total_length = float(rl[-1, 0])
                self.reward_computer.set_raceline(rl_points, rl_speeds)
            except Exception as e:
                print(f"[Env] Raceline load failed for {track_name}: {e}")

        # Base env recreation only when switching tracks (or initial load)
        if init_base_env and F1TENTH_GYM_AVAILABLE:
            try:
                map_name = self._resolve_map_name(track_name)
                self.base_env = RealF110Wrapper(map_name=map_name, timestep=0.01, track=track_name)
            except Exception as e:
                warnings.warn(f"Failed to initialize f110_gym wrapper for {track_name}: {e}")
                self.base_env = None
        elif F1TENTH_GYM_AVAILABLE and hasattr(self, 'base_env'):
            # Switching mid-training: recreate base_env on the new map
            try:
                map_name = self._resolve_map_name(track_name)
                self.base_env = RealF110Wrapper(map_name=map_name, timestep=0.01, track=track_name)
            except Exception as e:
                warnings.warn(f"Failed to switch f110_gym to {track_name}: {e}")

    def _setup_spaces(self):
        """
        Define observation and action spaces.
        
        Observation (xt): [v_current, steering_current, v_desired, steering_desired, yaw_rate, ...]
        Action (at): [steering_cmd, throttle_cmd]
        """
        obs_config = self.config.get('observation', {})
        
        # Base observation dimension: 5 core signals
        obs_dim = 5  # v, steering, v_des, steering_des, yaw_rate
        
        # Add lidar if enabled
        if obs_config.get('include_lidar', False):
            obs_dim += obs_config.get('lidar_beams', 36)
        

        # Observation space: all continuous, unbounded (relative to nominal ranges)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: steering angle + throttle/velocity command
        # Steering: [-0.4189, 0.4189] rad (typical F1Tenth limits)
        # Throttle: [-1, 1] normalized (maps to actual accel/brake)
        self.action_space = spaces.Box(
            low=np.array([-0.4189, -1.0], dtype=np.float32),
            high=np.array([0.4189, 1.0], dtype=np.float32),
            dtype=np.float32
        )
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Dict = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to start of episode.
        
        Samples new randomized physics parameters (et) and resets base environment.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of:
            - observation: initial state vector xt
            - info: dict with episode metadata (includes et and zt)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Sample new physics parameters (Zhang Section II-A)
        mode = SampleMode.TRAIN if options is None or options.get('training', True) else SampleMode.GENERALIZATION
        delta = options.get('delta', 0.5) if options else 0.5
        

        # Multi-map training: sample a random track from tracks_list each episode
        if self.tracks_list and len(self.tracks_list) > 1:
            if self.track_weights:
                probs = [self.track_weights.get(t, 1.0) for t in self.tracks_list]
                probs = np.array(probs) / sum(probs)
                new_track = str(np.random.choice(self.tracks_list, p=probs))
            else:
                new_track = str(np.random.choice(self.tracks_list))
            if new_track != self.track:
                self._load_track_data(new_track, init_base_env=True)
        self.current_physics_params = self.randomizer.sample(mode=mode, delta=delta)
        
        # Reset base environment with randomized physics (Zhang Section II-A: et)
        physics_overrides = self._map_physics_params(self.current_physics_params)
        # Randomize spawn position along raceline for robustness (item #4)
        custom_pose = None
        if self._raceline is not None and self._raceline_points is not None:
            random_spawn = self.config.get('randomize_spawn', True)
            if random_spawn:
                # Spawn on CENTERLINE (not raceline offset) with tighter heading perturbation
                # Load centerline for spawn — it's guaranteed to be inside the track
                import os as _os
                _cl_path = f'/research_ws/maps/{self.track}_centerline.csv'
                if _os.path.exists(_cl_path):
                    _cl = np.loadtxt(_cl_path, delimiter=',')
                    spawn_idx = int(np.random.randint(0, len(_cl)))
                    sx = float(_cl[spawn_idx, 0])
                    sy = float(_cl[spawn_idx, 1])
                    # Heading from consecutive centerline points
                    next_idx = (spawn_idx + 3) % len(_cl)
                    dx = float(_cl[next_idx, 0] - sx)
                    dy = float(_cl[next_idx, 1] - sy)
                    base_heading = np.arctan2(dy, dx)
                    heading_noise = np.random.uniform(-0.087, 0.087)  # +/- 5 deg (was 15)
                    custom_pose = [sx, sy, float(base_heading + heading_noise)]
        if self.base_env is None:
            raw_obs_vec = np.zeros(3, dtype=np.float32)
            raw_obs_dict = None
        else:
            raw_obs_vec, raw_obs_dict = self.base_env.reset(physics_overrides, custom_pose=custom_pose)
        
        # Reset episode state
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.current_state = self._process_observation(raw_obs_vec, self.prev_action, raw_obs_dict)
        self.episode_step = 0
        self.total_episode_reward = 0.0
        self.reward_computer.reset_episode()
        
        # Prepare info dict with physics parameters
        # Note: zt (intrinsics) would be computed by encoder μ during training
        info = {
            'physics_params': self.current_physics_params,
            'physics_params_description': self.randomizer.params_to_description(self.current_physics_params),
            'episode_num': 0,
        }
        if raw_obs_dict is not None:
            info['poses_x'] = float(raw_obs_dict['poses_x'][0])
            info['poses_y'] = float(raw_obs_dict['poses_y'][0])
            info['poses_theta'] = float(raw_obs_dict['poses_theta'][0])
        
        return self.current_state, info
    
    def _map_physics_params(self, p: Dict) -> Dict:
        """Map randomizer output (Zhang et's et) to f110_gym params dict overrides."""
        overrides = {}
        if p is None:
            return overrides
        grip = p.get('grip_factor', None)
        if grip is not None:
            overrides['mu'] = 1.0489 * float(grip)
        mass_scale = p.get('mass_scale', None)
        if mass_scale is not None:
            overrides['m'] = 3.74 * float(mass_scale)
        inertia_scale = p.get('inertia_scale', None)
        if inertia_scale is not None:
            overrides['I'] = 0.04712 * float(inertia_scale)
        return overrides


    def _compute_raceline_lookahead(self, raw_obs_dict, current_velocity):
        """Return [v_des_now, v_des_1s_ahead] normalized to [0, 1] (divide by 8 m/s).
        Returns None if raceline or position unavailable."""
        if self._raceline is None or raw_obs_dict is None:
            return None
        if 'poses_x' not in raw_obs_dict or 'poses_y' not in raw_obs_dict:
            return None
        x = float(raw_obs_dict['poses_x'][0])
        y = float(raw_obs_dict['poses_y'][0])
        # Find nearest raceline point to current position
        dx = self._raceline_points[:, 0] - x
        dy = self._raceline_points[:, 1] - y
        nearest_idx = int(np.argmin(dx**2 + dy**2))
        v_des_now = float(self._raceline_speeds[nearest_idx])
        # Lookahead: project arc-length 1 second ahead at current velocity
        s_current = float(self._raceline_s[nearest_idx])
        s_lookahead = (s_current + max(current_velocity, 0.5) * 1.0) % self._raceline_total_length
        lookahead_idx = int(np.argmin(np.abs(self._raceline_s - s_lookahead)))
        v_des_ahead = float(self._raceline_speeds[lookahead_idx])
        # Normalize by 8 m/s (max speed) for consistency with LiDAR scale
        max_v = self.config.get("max_velocity", 8.0)
        lookahead = np.array([v_des_now / max_v, v_des_ahead / max_v], dtype=np.float32)
        # Privileged info distillation: dropout lookahead proportional to IL weight
        # When il_weight=1.0 (early): always show lookahead
        # When il_weight=0.0 (late): never show lookahead
        # Forces policy to learn map-agnostic behavior as IL decays
        if np.random.random() > self.current_il_weight:
            lookahead = np.zeros(2, dtype=np.float32)
        return lookahead

    def _process_observation(self, raw_obs_vec: np.ndarray, last_action: np.ndarray,
                              raw_obs_dict: Dict = None) -> np.ndarray:
        """Build state vector xt = [v, steering, v_des, steering_des, yaw_rate, (lidar...)].
        No raceline lookahead (map-agnostic).
        Adds Gaussian noise + random beam dropout for sim-to-real robustness."""
        v, steering, yaw_rate = float(raw_obs_vec[0]), float(raw_obs_vec[1]), float(raw_obs_vec[2])
        steering_des = float(last_action[0])
        v_des = (float(last_action[1]) + 1.0) / 2.0 * self.config.get('max_velocity', 8.0)
        base_obs = np.array([v, steering, v_des, steering_des, yaw_rate], dtype=np.float32)
        obs_config = self.config.get('observation', {})
        if not obs_config.get('include_lidar', False):
            return base_obs
        num_beams = obs_config.get('lidar_beams', 36)
        max_range = obs_config.get('lidar_max_range', 10.0)
        if raw_obs_dict is not None and 'scans' in raw_obs_dict:
            full_scan = np.asarray(raw_obs_dict['scans'][0], dtype=np.float32)
            indices = np.linspace(0, len(full_scan) - 1, num_beams).astype(int)
            lidar = full_scan[indices]
            lidar = np.clip(lidar, 0.0, max_range) / max_range
            noise_std = obs_config.get('lidar_noise_std', 0.02)
            dropout_prob = obs_config.get('lidar_dropout_prob', 0.05)
            if noise_std > 0:
                lidar = lidar + np.random.normal(0, noise_std, size=lidar.shape).astype(np.float32)
                lidar = np.clip(lidar, 0.0, 1.0)
            if dropout_prob > 0:
                mask = np.random.random(size=lidar.shape) < dropout_prob
                lidar[mask] = 1.0
        else:
            lidar = np.zeros(num_beams, dtype=np.float32)
        return np.concatenate([base_obs, lidar]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of environment.
        
        Args:
            action: Action vector [steering_cmd, throttle_cmd]
            
        Returns:
            Tuple of:
            - observation: Next state vector xt+1
            - reward: Composite reward (Zhang Section II-C)
            - terminated: Episode termination flag (off-track, etc.)
            - truncated: Time limit flag
            - info: Diagnostics (reward breakdown, etc.)
        """
        self.episode_step += 1
        
        # Clip action to bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Map action -> f110_gym control input [steer_angle, velocity_cmd]
        steer_cmd = float(action[0])
        throttle_cmd = float(action[1])
        velocity_cmd = (throttle_cmd + 1.0) / 2.0 * self.config.get('max_velocity', 8.0)
        sim_action = np.array([steer_cmd, velocity_cmd], dtype=np.float32)
        
        # Step base environment
        if self.base_env is None:
            raw_obs_vec = np.zeros(3, dtype=np.float32)
            raw_obs_dict = None
            terminated = False
        else:
            raw_obs_vec, raw_obs_dict, done, _ = self.base_env.step(sim_action)
            # Terminate immediately on collision -- prevents wall-phasing exploit
            collision_flag = bool(raw_obs_dict['collisions'][0]) if raw_obs_dict is not None else False
            terminated = bool(done) or collision_flag
        truncated = False
        
        # Process observation -> xt
        obs = self._process_observation(raw_obs_vec, action, raw_obs_dict)
        
        # Extract state variables for reward computation (Zhang Section II-C)
        current_velocity = float(raw_obs_vec[0])
        # Use external v_des from Pure Pursuit speed profile (Zhang et al.)
        # NOT the policy's own velocity command (circular dependency)
        if (hasattr(self.reward_computer, '_v_des_profile')
                and self.reward_computer._v_des_profile is not None
                and raw_obs_dict is not None):
            _px = float(raw_obs_dict['poses_x'][0])
            _py = float(raw_obs_dict['poses_y'][0])
            _dx = self.reward_computer.centerline[:, 0] - _px
            _dy = self.reward_computer.centerline[:, 1] - _py
            _cl_idx = int(np.argmin(_dx**2 + _dy**2))
            desired_velocity = float(self.reward_computer._v_des_profile[_cl_idx])
        else:
            desired_velocity = velocity_cmd
        current_yaw_rate = float(raw_obs_vec[2])
        # Bicycle-model derivation: yaw_rate_des = v_des * tan(steer_des) / wheelbase
        # wheelbase = lf + lr = 0.15875 + 0.17145 = 0.3302 m (f110_gym default)
        wheelbase = 0.3302
        desired_yaw_rate = velocity_cmd * np.tan(steer_cmd) / wheelbase
        
        # Compute composite reward (Zhang Section II-C)
        # Extract position and collision for new reward terms
        _poses_x = float(raw_obs_dict['poses_x'][0]) if raw_obs_dict is not None else None
        _poses_y = float(raw_obs_dict['poses_y'][0]) if raw_obs_dict is not None else None
        _collision = bool(raw_obs_dict['collisions'][0]) if raw_obs_dict is not None else False
        _lidar_obs = obs[5:] if len(obs) > 5 else None

        step_reward, reward_breakdown = self.reward_computer.compute_step_reward(
            action=action,
            prev_action=self.prev_action,
            current_velocity=current_velocity,
            desired_velocity=desired_velocity,
            current_yaw_rate=current_yaw_rate,
            desired_yaw_rate=desired_yaw_rate,
            poses_x=_poses_x,
            poses_y=_poses_y,
            lidar_obs=_lidar_obs,
            collision=_collision,
        )
        
        # Add mid-episode disturbance if configured
        episode_progress = self.episode_step / self.max_episode_steps
        self.current_physics_params = self.randomizer.add_mid_episode_disturbance(
            self.current_physics_params, episode_progress
        )
        
        # Check termination conditions
        terminated_rma, termination_reason = self.reward_computer.compute_episode_termination(
            state={'velocity': current_velocity},
            track_error=None,
            lateral_accel=None,
        )
        terminated = terminated or terminated_rma
        
        # Check time limit
        truncated = truncated or (self.episode_step >= self.max_episode_steps)
        
        # Update state
        self.prev_action = action
        self.current_state = obs
        self.total_episode_reward += step_reward
        
        # Prepare info dict
        info = {
            'reward_breakdown': reward_breakdown,
            'episode_step': self.episode_step,
            'total_episode_reward': self.total_episode_reward,
            'physics_params': self.current_physics_params,
        }
        # Pass through position and lap data from f110_gym for lap completion
        # and centerline deviation metrics in evaluation
        if raw_obs_dict is not None:
            info['poses_x'] = float(raw_obs_dict['poses_x'][0])
            info['poses_y'] = float(raw_obs_dict['poses_y'][0])
            info['poses_theta'] = float(raw_obs_dict['poses_theta'][0])
            info['lap_counts'] = int(raw_obs_dict['lap_counts'][0])
        if termination_reason is not None:
            info['termination_reason'] = termination_reason
        
        return obs, step_reward, terminated, truncated, info
    
    def render(self):
        """Render environment (delegate to base_env if available)."""
        if self.base_env is not None and self.render_mode == 'human':
            return self.base_env.render()
        return None
    
    def close(self):
        """Clean up environment."""
        if self.base_env is not None:
            self.base_env.close()
    
    def get_physics_params_description(self) -> str:
        """Get human-readable description of current physics parameters."""
        if self.current_physics_params is None:
            return "No physics parameters sampled yet"
        return self.randomizer.params_to_description(self.current_physics_params)
