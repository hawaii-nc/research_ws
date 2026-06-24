"""
F1Tenth Reward Function for RMA Training
==========================================

Two variants for Paper 1 / Paper 2 comparison:

Baseline RMA (Paper 1):
    R = 5.0*progress
      - 0.05*smoothness
      + 0.01*alive
      - 1.0*collision
      - 0.5*speed_turn
      - 0.1*close_wall
      - weight_speed_limit * max(0, v - safe_speed)   [curvature-aware]

Physics-Limit-Aware RMA (Paper 2, use_wall_proximity=true):
    R = above + (-0.2*wall_proximity)

Key design principles:
1. Progress is non-exploitable (arc-length along centerline, not self-commanded velocity)
2. Curvature-aware speed limit: penalizes exceeding physically-justified safe speed
   BEFORE the car reaches the wall -- proactive rather than reactive
3. Wall proximity is reactive (already near wall); speed_limit is proactive (too fast for geometry)
4. Collision termination is the hard constraint; speed_limit is the soft learning signal

Why curvature-aware speed differs from increasing collision penalties:
- Collision penalty fires AFTER the car hits the wall (reactive, too late)
- Speed limit penalty fires BEFORE corners based on track geometry (proactive)
- Collision teaches "don't touch walls"; speed limit teaches "understand your limits"
- This is more aligned with the lab vision: a car that understands safety-speed tradeoffs
  from the geometry of the track, not just from crashing
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

        # Precompute centerline curvatures if centerline is available
        self._curvatures = None
        if self.centerline is not None:
            self._precompute_curvatures()

    def _default_config(self) -> Dict:
        return {
            # Variant selection
            'use_wall_proximity': False,

            # Term weights
            'weight_progress': 5.0,
            'weight_wall_proximity': 0.2,
            'weight_smoothness': 0.05,
            'weight_alive': 0.01,
            'weight_collision': 1.0,
            'weight_speed_turn': 0.5,
            'weight_close_wall': 0.1,

            # Curvature-aware speed limit
            # Justification for default 0.3:
            #   At a hard turn (curvature ~2.0 rad/m), safe_speed ~2.8 m/s
            #   Excess speed penalty at 8 m/s: -0.3 * (8 - 2.8) = -1.56/step
            #   This is comparable to ~1.5 steps of progress reward (5.0 * 0.2m = 1.0)
            #   So the policy must genuinely slow down to maintain positive reward
            'weight_speed_limit': 0.3,

            # Safe speed formula: safe_speed = k / sqrt(curvature + epsilon)
            # k scales the maximum safe speed. At curvature=0 (straight): safe_speed = k/sqrt(eps) = max_speed
            # k=4.0 gives safe_speed ~4.0 m/s at curvature=1.0, ~2.8 m/s at curvature=2.0
            'speed_limit_k': 4.0,
            'speed_limit_epsilon': 0.01,
            'speed_limit_min': 1.0,   # floor: never require below 1 m/s
            'speed_limit_max': 8.0,   # ceiling: never require above max speed

            # Curvature lookahead: use N points ahead/behind for curvature estimate
            'curvature_window': 3,

            # Progress scaling
            'max_progress_per_step': 0.5,

            # Smoothness
            'max_action_diff': 0.4,

            # Speed-turn coupling
            'speed_turn_steer_threshold': 0.15,

            # Close wall
            'close_wall_threshold': 0.05,

            # Legacy compatibility
            'max_velocity_error': 2.0,
            'max_yaw_rate_error': 4.0,
            'max_track_error': 0.5,
            'min_speed': -0.1,
            'max_lateral_accel': 10.0,
        }

    def set_centerline(self, centerline: np.ndarray):
        """Set centerline (Nx2 array) and precompute curvatures."""
        self.centerline = centerline
        self._prev_centerline_idx = None
        self._precompute_curvatures()

    def _precompute_curvatures(self):
        """
        Precompute curvature at every centerline point.

        Uses the Menger curvature formula for three consecutive points:
            curvature = 2 * |cross(B-A, C-A)| / (|B-A| * |C-B| * |C-A|)

        Stores result as self._curvatures (N,) array in rad/m.
        """
        cl = self.centerline
        n = len(cl)
        curvatures = np.zeros(n)

        for i in range(1, n - 1):
            A = cl[i - 1]
            B = cl[i]
            C = cl[i + 1]

            AB = B - A
            BC = C - B
            AC = C - A

            # Cross product magnitude (2D: scalar)
            cross = abs(AB[0] * BC[1] - AB[1] * BC[0])

            dAB = np.linalg.norm(AB)
            dBC = np.linalg.norm(BC)
            dAC = np.linalg.norm(AC)

            denom = dAB * dBC * dAC
            if denom > 1e-10:
                curvatures[i] = 2.0 * cross / denom
            else:
                curvatures[i] = 0.0

        # Endpoints: copy neighbor
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]

        self._curvatures = curvatures

    def _get_local_curvature(self, cl_idx: int) -> float:
        """
        Get maximum curvature in a window ahead of current position.

        Uses a lookahead window so the car is penalized for upcoming
        corners, not just the current position. This is what enables
        proactive speed reduction BEFORE the corner.
        """
        if self._curvatures is None:
            return 0.0

        n = len(self._curvatures)
        window = self.config.get('curvature_window', 3)

        # Look ahead: current position + window points forward
        indices = [
            (cl_idx + i) % n
            for i in range(0, window + 1)
        ]
        return float(np.max(self._curvatures[indices]))

    def _compute_safe_speed(self, curvature: float) -> float:
        """
        Compute physically-justified safe speed from curvature.

        Formula: safe_speed = k / sqrt(curvature + epsilon)

        Physical justification:
        - From circular motion: centripetal acceleration a = v^2 / r
        - Curvature = 1/r, so r = 1/curvature
        - Maximum safe speed given friction limit mu*g = v^2 * curvature
        - v_safe = sqrt(mu * g / curvature)
        - We use k = sqrt(mu * g) ≈ sqrt(1.0 * 9.81) ≈ 3.13
        - k=4.0 is slightly generous (allows slightly higher speed than pure friction limit)

        Examples:
            curvature=0.0 (straight):  safe_speed = 8.0 m/s (capped at max)
            curvature=0.5 (gentle):    safe_speed = 5.66 m/s
            curvature=1.0 (medium):    safe_speed = 4.00 m/s
            curvature=2.0 (hard):      safe_speed = 2.83 m/s
            curvature=4.0 (very hard): safe_speed = 2.00 m/s
        """
        k = self.config.get('speed_limit_k', 4.0)
        eps = self.config.get('speed_limit_epsilon', 0.01)
        v_min = self.config.get('speed_limit_min', 1.0)
        v_max = self.config.get('speed_limit_max', 8.0)

        safe = k / np.sqrt(curvature + eps)
        return float(np.clip(safe, v_min, v_max))

    def reset_episode(self):
        """Call at episode start to reset progress tracking."""
        self._prev_centerline_idx = None

    def _nearest_centerline_idx(self, x: float, y: float) -> int:
        """Find index of nearest centerline point to (x, y)."""
        dx = self.centerline[:, 0] - x
        dy = self.centerline[:, 1] - y
        return int(np.argmin(dx**2 + dy**2))

    def compute_progress(self, poses_x: float, poses_y: float) -> float:
        """Arc-length progress along centerline this step."""
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
            total_len = np.sum(np.sqrt(
                np.diff(self.centerline[:, 0])**2 +
                np.diff(self.centerline[:, 1])**2
            ))
            avg_spacing = total_len / (n - 1)
            progress_m = delta_idx * avg_spacing
        else:
            progress_m = 0.0

        progress_m = min(progress_m, self.config['max_progress_per_step'])

        if np.random.rand() < 0.001:
            print(
                f"[PROGRESS DEBUG] "
                f"prev_idx={prev_idx}, idx={idx}, "
                f"delta_idx={delta_idx}, "
                f"progress_m={progress_m:.5f}"
            )

        self._prev_centerline_idx = idx
        return progress_m

    def compute_wall_proximity_penalty(self, lidar_obs: np.ndarray) -> float:
        """LiDAR-based wall proximity penalty (Paper 2 term)."""
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
        """Action jerk penalty."""
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

        # ── Core terms ──────────────────────────────────────────────────────
        progress = self.config['weight_progress'] * self.compute_progress(
            poses_x, poses_y
        )
        smoothness = self.compute_smoothness_penalty(action, prev_action)
        alive = self.config['weight_alive']
        collision_penalty = (
            -self.config['weight_collision'] if collision else 0.0
        )

        # ── Wall proximity (Paper 2) ─────────────────────────────────────────
        wall = 0.0
        if (
            self.config.get('use_wall_proximity', False)
            and lidar_obs is not None
        ):
            wall = self.compute_wall_proximity_penalty(lidar_obs)

        # ── Speed-turn coupling ──────────────────────────────────────────────
        speed_turn_penalty = 0.0
        _steer_threshold = self.config.get('speed_turn_steer_threshold', 0.15)
        _weight_st = self.config.get('weight_speed_turn', 0.5)
        if action is not None and current_velocity is not None:
            _steer_mag = abs(float(action[0])) if hasattr(action, '__len__') else 0.0
            _v_norm = min(abs(float(current_velocity)) / 8.0, 1.0)
            _steer_excess = max(0.0, _steer_mag - _steer_threshold)
            speed_turn_penalty = -_weight_st * _steer_excess * _v_norm

        # ── Close wall penalty ───────────────────────────────────────────────
        close_wall_penalty = 0.0
        _close_threshold = self.config.get('close_wall_threshold', 0.05)
        _weight_cw = self.config.get('weight_close_wall', 0.1)
        if lidar_obs is not None:
            _min_lidar = float(np.min(lidar_obs))
            if _min_lidar < _close_threshold:
                close_wall_penalty = -_weight_cw * (
                    1.0 - _min_lidar / _close_threshold
                )

        # ── Curvature-aware speed limit ──────────────────────────────────────
        # This is the proactive safety term. It fires BEFORE the car reaches
        # the wall, based on track geometry at the car's current position.
        #
        # How it differs from collision penalty:
        #   - Collision: fires AFTER contact (reactive, too late to learn from)
        #   - Speed limit: fires when speed EXCEEDS what geometry allows (proactive)
        #
        # Why this aligns with "understanding physical limits":
        #   The safe speed is derived from centripetal acceleration physics.
        #   A car that learns to respect this limit has genuinely internalized
        #   the relationship between track geometry and vehicle dynamics --
        #   exactly the safety-speed tradeoff the lab vision requires.
        speed_limit_penalty = 0.0
        curvature = 0.0
        safe_speed = self.config.get('speed_limit_max', 8.0)

        if (
            self.centerline is not None
            and self._curvatures is not None
            and poses_x is not None
            and poses_y is not None
            and current_velocity is not None
        ):
            cl_idx = self._nearest_centerline_idx(poses_x, poses_y)
            curvature = self._get_local_curvature(cl_idx)
            safe_speed = self._compute_safe_speed(curvature)

            # Only penalize exceeding safe speed (not being too slow)
            speed_excess = max(0.0, abs(float(current_velocity)) - safe_speed)
            speed_limit_penalty = (
                -self.config.get('weight_speed_limit', 0.3) * speed_excess
            )

        # ── Reverse penalty ──────────────────────────────────────────────────
        reverse_penalty = (
            -0.5
            if (current_velocity is not None and float(current_velocity) < -0.05)
            else 0.0
        )

        # ── Total ────────────────────────────────────────────────────────────
        total = (
            progress
            + smoothness
            + alive
            + collision_penalty
            + wall
            + speed_turn_penalty
            + close_wall_penalty
            + speed_limit_penalty
            + reverse_penalty
        )

        # ── Debug (0.1% of steps) ────────────────────────────────────────────
        if np.random.rand() < 0.001:
            print(
                f"[REWARD DEBUG] "
                f"progress={progress:.4f}, "
                f"smooth={smoothness:.4f}, "
                f"alive={alive:.4f}, "
                f"collision={collision_penalty:.4f}, "
                f"wall={wall:.4f}, "
                f"speed_turn={speed_turn_penalty:.4f}, "
                f"close_wall={close_wall_penalty:.4f}, "
                f"speed_limit={speed_limit_penalty:.4f} "
                f"[curv={curvature:.3f}, safe={safe_speed:.2f}m/s, "
                f"actual={float(current_velocity) if current_velocity else 0:.2f}m/s], "
                f"total={total:.4f}"
            )

        breakdown = {
            'progress': progress,
            'smoothness': smoothness,
            'alive': alive,
            'collision': collision_penalty,
            'wall_proximity': wall,
            'speed_turn': speed_turn_penalty,
            'close_wall': close_wall_penalty,
            'speed_limit': speed_limit_penalty,
            'reverse': reverse_penalty,
            'total': total,
            # Diagnostic values
            'curvature': curvature,
            'safe_speed': safe_speed,
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
        variant = (
            "Physics-Limit-Aware RMA"
            if self.config.get('use_wall_proximity')
            else "Baseline RMA"
        )
        return f"Reward variant: {variant}"
