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
        
        self.config = config or self._default_config()
        self.render_mode = render_mode
        self.track = track
        self.max_episode_steps = max_episode_steps
        
        # Initialize components
        self.randomizer = PhysicsRandomizer(self.config.get('randomization', {}))
        self.reward_computer = RewardComputer(self.config.get('reward', {}))
        
        # Underlying f1tenth_gym environment
        self.base_env = None
        if F1TENTH_GYM_AVAILABLE:
            try:
                map_name = self._resolve_map_name(self.track)
                self.base_env = RealF110Wrapper(map_name=map_name, timestep=0.01)
            except Exception as e:
                warnings.warn(f"Failed to initialize f110_gym wrapper: {e}")
                self.base_env = None
        
        # State-action space definition
        self._setup_spaces()
        
        # Episode state
        self.current_physics_params = None
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
        
        self.current_physics_params = self.randomizer.sample(mode=mode, delta=delta)
        
        # Reset base environment with randomized physics (Zhang Section II-A: et)
        physics_overrides = self._map_physics_params(self.current_physics_params)
        if self.base_env is None:
            raw_obs_vec = np.zeros(3, dtype=np.float32)
        else:
            raw_obs_vec, _ = self.base_env.reset(physics_overrides)
        
        # Reset episode state
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.current_state = self._process_observation(raw_obs_vec, self.prev_action)
        self.episode_step = 0
        self.total_episode_reward = 0.0
        
        # Prepare info dict with physics parameters
        # Note: zt (intrinsics) would be computed by encoder μ during training
        info = {
            'physics_params': self.current_physics_params,
            'physics_params_description': self.randomizer.params_to_description(self.current_physics_params),
            'episode_num': 0,  # Updated by caller
        }
        
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

    def _process_observation(self, raw_obs_vec: np.ndarray, last_action: np.ndarray) -> np.ndarray:
        """
        Build state vector xt = [v, steering, v_des, steering_des, yaw_rate]
        from real_env's [v, steering_proxy, yaw_rate] and last commanded action.
        """
        v, steering, yaw_rate = float(raw_obs_vec[0]), float(raw_obs_vec[1]), float(raw_obs_vec[2])
        steering_des = float(last_action[0])
        v_des = (float(last_action[1]) + 1.0) / 2.0 * 8.0  # match step()'s velocity_cmd mapping
        return np.array([v, steering, v_des, steering_des, yaw_rate], dtype=np.float32)
    
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
        velocity_cmd = (throttle_cmd + 1.0) / 2.0 * 8.0  # [-1,1] -> [0, 8] m/s (realistic F1Tenth range)
        sim_action = np.array([steer_cmd, velocity_cmd], dtype=np.float32)
        
        # Step base environment
        if self.base_env is None:
            raw_obs_vec = np.zeros(3, dtype=np.float32)
            terminated = False
        else:
            raw_obs_vec, _, done, _ = self.base_env.step(sim_action)
            terminated = bool(done)
        truncated = False
        
        # Process observation -> xt
        obs = self._process_observation(raw_obs_vec, action)
        
        # Extract state variables for reward computation (Zhang Section II-C)
        current_velocity = float(raw_obs_vec[0])
        desired_velocity = velocity_cmd
        current_yaw_rate = float(raw_obs_vec[2])
        # Bicycle-model derivation: yaw_rate_des = v_des * tan(steer_des) / wheelbase
        # wheelbase = lf + lr = 0.15875 + 0.17145 = 0.3302 m (f110_gym default)
        wheelbase = 0.3302
        desired_yaw_rate = velocity_cmd * np.tan(steer_cmd) / wheelbase
        
        # Compute composite reward (Zhang Section II-C)
        step_reward, reward_breakdown = self.reward_computer.compute_step_reward(
            action=action,
            prev_action=self.prev_action,
            current_velocity=current_velocity,
            desired_velocity=desired_velocity,
            current_yaw_rate=current_yaw_rate,
            desired_yaw_rate=desired_yaw_rate,
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
