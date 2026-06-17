"""
F1Tenth Reward Function for RMA Training
==========================================

Two variants for Paper 1 / Paper 2 comparison:

Baseline RMA (Paper 1):
    R = 2.0*progress - 0.1*smoothness + 0.05*alive - 20*collision

Physics-Limit-Aware RMA (Paper 2):
    R = 2.0*progress - 1.0*wall_proximity - 0.1*smoothness + 0.05*alive - 20*collision

Progress = arc-length advancement along centerline this step.
Wall proximity = 1 - min(lidar_36_beams) -- close walls -> high penalty.
Collision = detected by f110_gym collisions flag.

Key design principle: reward depends ONLY on environment state,
not on policy outputs. Eliminates the circular dependency that
caused the near-zero throttle degenerate solution.
"""

from typing import Dict, Tuple, Optional
import numpy as np


class RewardComputer:
    """
    Computes composite reward for F1Tenth RMA training.
    """

    def __init__(self, config: Dict = None, centerline: np.ndarray = None):
        """
        Args:
            config: reward weights and parameters
            centerline: Nx2 array of (x, y) centerline points for progress computation
        """
        self.config = config or self._default_config()
        self.centerline = centerline  # set after init if available
        self._prev_centerline_idx = None

    def _default_config(self) -> Dict:
        return {
            # Variant selection
            'use_wall_proximity': False,   # False = Baseline RMA, True = Physics-Limit-Aware RMA

            # Term weights
            'weight_progress': 2.0,        # Arc-length progress along centerline per step
            'weight_wall_proximity': 1.0,  # LiDAR-based safety penalty (Paper 2 only)
            'weight_smoothness': 0.1,      # Action jerk penalty
            'weight_alive': 0.05,          # Small survival bonus
            'weight_collision': 20.0,      # Explicit collision penalty

            # Progress scaling
            'max_progress_per_step': 0.5,  # Cap meters/step to avoid teleport spikes

            # Smoothness
            'max_action_diff': 0.4,

            # Legacy (kept for compatibility with existing code)
            'max_velocity_error': 2.0,
            'max_yaw_rate_error': 4.0,
            'max_track_error': 0.5,
            'min_speed': -0.5,
            'max_lateral_accel': 10.0,
        }

    def set_centerline(self, centerline: np.ndarray):
        """Set centerline array (Nx2, x/y columns) for progress computation."""
        self.centerline = centerline
        self._prev_centerline_idx = None

    def reset_episode(self):
        """Call at episode start to reset progress tracking."""
        self._prev_centerline_idx = None

    def _nearest_centerline_idx(self, x: float, y: float) -> int:
        """Find index of nearest centerline point to (x, y)."""
        dx = self.centerline[:, 0] - x
        dy = self.centerline[:, 1] - y
        return int(np.argmin(dx**2 + dy**2))

    def compute_progress(self, poses_x: float, poses_y: float) -> float:
        """
        Arc-length progress along centerline this step.

        Projects car position onto nearest centerline point, computes
        how many points forward it moved since last step, converts to
        meters using average point spacing.

        Returns 0 if centerline not set or first step of episode.
        """
        if self.centerline is None or poses_x is None:
            return 0.0

        idx = self._nearest_centerline_idx(poses_x, poses_y)

        if self._prev_centerline_idx is None:
            self._prev_centerline_idx = idx
            return 0.0

        n = len(self.centerline)
        prev_idx = self._prev_centerline_idx

        # Forward progress (handles wraparound)
        delta_idx = (idx - prev_idx) % n

        # Ignore large jumps (teleport/reset artifacts)
        if delta_idx > n // 4:
            delta_idx = 0

        # Convert index delta to meters using average spacing
        if delta_idx > 0:
            # Average spacing of centerline points
            total_len = np.sum(np.sqrt(
                np.diff(self.centerline[:, 0])**2 +
                np.diff(self.centerline[:, 1])**2
            ))
            avg_spacing = total_len / (n - 1)
            progress_m = delta_idx * avg_spacing
        else:
            progress_m = 0.0

        # Cap to avoid spikes
        progress_m = min(progress_m, self.config['max_progress_per_step'])

        self._prev_centerline_idx = idx
        return progress_m

    def compute_wall_proximity_penalty(self, lidar_obs: np.ndarray) -> float:
        """
        LiDAR-based wall proximity penalty (Paper 2 term).

        min(lidar) near 0 = wall very close = high penalty.
        min(lidar) near 1 = open space = no penalty.

        lidar_obs: normalized [0,1] array (36 beams from obs[5:41])
        """
        if lidar_obs is None or len(lidar_obs) == 0:
            return 0.0
        min_dist = float(np.min(lidar_obs))
        # Penalty increases as walls get closer
        proximity = 1.0 - min_dist
        return -self.config['weight_wall_proximity'] * proximity

    def compute_smoothness_penalty(
        self,
        action: np.ndarray,
        prev_action: Optional[np.ndarray]
    ) -> float:
        if prev_action is None:
            return 0.0
        action_diff = np.linalg.norm(action - prev_action)
        normalized = action_diff / self.config['max_action_diff']
        return -self.config['weight_smoothness'] * normalized

    def compute_step_reward(
        self,
        action: np.ndarray,
        prev_action: Optional[np.ndarray],
        current_velocity: float = 0.0,
        desired_velocity: float = 0.0,
        current_yaw_rate: float = 0.0,
        desired_yaw_rate: float = 0.0,
        poses_x: float = None,
        poses_y: float = None,
        lidar_obs: np.ndarray = None,
        collision: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute composite reward.

        New parameters vs old signature:
            poses_x, poses_y: car world position (from info dict)
            lidar_obs: normalized 36-beam LiDAR array (obs[5:41])
            collision: True if f110_gym reports collision this step
        """
        progress = self.config['weight_progress'] * self.compute_progress(poses_x, poses_y)
        smoothness = self.compute_smoothness_penalty(action, prev_action)
        alive = self.config['weight_alive']
        collision_penalty = -self.config['weight_collision'] if collision else 0.0

        wall = 0.0
        if self.config.get('use_wall_proximity', False) and lidar_obs is not None:
            wall = self.compute_wall_proximity_penalty(lidar_obs)

        total = progress + smoothness + alive + collision_penalty + wall

        breakdown = {
            'progress': progress,
            'smoothness': smoothness,
            'alive': alive,
            'collision': collision_penalty,
            'wall_proximity': wall,
            'total': total,
        }
        return total, breakdown

    def compute_episode_termination(
        self,
        state: Dict,
        track_error: Optional[float] = None,
        lateral_accel: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        velocity = state.get('velocity', 0.0)
        if velocity < self.config['min_speed']:
            return True, f"stuck_reversing (v={velocity:.2f})"
        if track_error is not None and track_error > self.config['max_track_error']:
            return True, f"off_track (error={track_error:.2f}m)"
        return False, None

    def config_summary(self) -> str:
        variant = "Physics-Limit-Aware RMA" if self.config.get('use_wall_proximity') else "Baseline RMA"
        return f"Reward variant: {variant}"
