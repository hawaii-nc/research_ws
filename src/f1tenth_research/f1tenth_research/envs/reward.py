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
        self.config = config or self._default_config()
        self.centerline = centerline
        self._prev_centerline_idx = None

    def _default_config(self) -> Dict:
        return {
            'use_wall_proximity': False,

            'weight_progress': 2.0,
            'weight_wall_proximity': 1.0,
            'weight_smoothness': 0.1,
            'weight_alive': 0.05,
            'weight_collision': 20.0,

            'max_progress_per_step': 0.5,

            'max_action_diff': 0.4,

            'max_velocity_error': 2.0,
            'max_yaw_rate_error': 4.0,
            'max_track_error': 0.5,
            'min_speed': -0.5,
            'max_lateral_accel': 10.0,
        }

    def set_centerline(self, centerline: np.ndarray):
        self.centerline = centerline
        self._prev_centerline_idx = None

    def reset_episode(self):
        self._prev_centerline_idx = None

    def _nearest_centerline_idx(self, x: float, y: float) -> int:
        dx = self.centerline[:, 0] - x
        dy = self.centerline[:, 1] - y
        return int(np.argmin(dx**2 + dy**2))

    def compute_progress(self, poses_x: float, poses_y: float) -> float:
        if self.centerline is None or poses_x is None:
            return 0.0

        idx = self._nearest_centerline_idx(poses_x, poses_y)

        if self._prev_centerline_idx is None:
            self._prev_centerline_idx = idx
            return 0.0

        n = len(self.centerline)
        prev_idx = self._prev_centerline_idx

        delta_idx = (idx - prev_idx) % n

        if delta_idx > n // 4:
            delta_idx = 0

        if delta_idx > 0:
            total_len = np.sum(
                np.sqrt(
                    np.diff(self.centerline[:, 0])**2 +
                    np.diff(self.centerline[:, 1])**2
                )
            )
            avg_spacing = total_len / (n - 1)
            progress_m = delta_idx * avg_spacing
        else:
            progress_m = 0.0

        progress_m = min(progress_m, self.config['max_progress_per_step'])

        # DEBUG PROGRESS
        if np.random.rand() < 0.001:
            print(
                f"[PROGRESS DEBUG] "
                f"prev_idx={prev_idx}, "
                f"idx={idx}, "
                f"delta_idx={delta_idx}, "
                f"progress_m={progress_m:.5f}"
            )

        self._prev_centerline_idx = idx
        return progress_m

    def compute_wall_proximity_penalty(self, lidar_obs: np.ndarray) -> float:
        if lidar_obs is None or len(lidar_obs) == 0:
            return 0.0

        min_dist = float(np.min(lidar_obs))
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

        progress = (
            self.config['weight_progress']
            * self.compute_progress(poses_x, poses_y)
        )

        smoothness = self.compute_smoothness_penalty(
            action,
            prev_action
        )

        alive = self.config['weight_alive']

        collision_penalty = (
            -self.config['weight_collision']
            if collision
            else 0.0
        )

        wall = 0.0

        if (
            self.config.get('use_wall_proximity', False)
            and lidar_obs is not None
        ):
            wall = self.compute_wall_proximity_penalty(lidar_obs)

        total = (
            progress
            + smoothness
            + alive
            + collision_penalty
            + wall
        )

        # DEBUG REWARD
        if np.random.rand() < 0.001:
            print(
                f"[REWARD DEBUG] "
                f"progress={progress:.4f}, "
                f"smooth={smoothness:.4f}, "
                f"alive={alive:.4f}, "
                f"collision={collision_penalty:.4f}, "
                f"wall={wall:.4f}, "
                f"total={total:.4f}"
            )

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

        if (
            track_error is not None
            and track_error > self.config['max_track_error']
        ):
            return True, f"off_track (error={track_error:.2f}m)"

        return False, None

    def config_summary(self) -> str:
        variant = (
            "Physics-Limit-Aware RMA"
            if self.config.get('use_wall_proximity')
            else "Baseline RMA"
        )

        return f"Reward variant: {variant}"
