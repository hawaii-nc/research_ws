"""
F1Tenth Reward Function - Zhang et al. (2025) faithful replication
===================================================================

Replicates Zhang et al. Section II-C composite reward:
  R = -w_v * |v_current - v_des|      velocity tracking (EXTERNAL reference)
    - w_yaw * |yaw_rate - yaw_des|    yaw rate tracking
    - w_smooth * |a_t - a_{t-1}|      smoothness
    + w_alive * survival bonus
    - w_collision * collision penalty

Added (Hell et al. / Sun et al.):
  + w_progress * arc_length_advancement   forward progress along centerline
  - w_centerline_error * lateral_error    deviation from centerline (when > threshold)

Paper 2 adds wall_proximity term on top of this baseline.
"""

from typing import Dict, Tuple, Optional
import numpy as np


class RewardComputer:

    def __init__(self, config: Dict = None, centerline: np.ndarray = None):
        self.config = config or self._default_config()
        self.centerline = centerline
        self._prev_centerline_idx = None
        self._v_des_profile = None

        # Progress tracking state (Hell et al. / Sun et al.)
        self._prev_s = None
        self._centerline_s = None
        self._track_length = None

        if self.centerline is not None:
            self._precompute_speed_profile()
            self._precompute_arc_length()

    def _default_config(self) -> Dict:
        return {
            'weight_velocity_tracking': 0.0,
            'weight_yaw_rate_tracking': 0.3,
            'weight_smoothness': 0.005,
            'weight_alive': 0.01,
            'weight_collision': 0.5,
            'weight_progress': 0.5,
            'weight_centerline_error': 0.1,
            'centerline_error_threshold': 0.1,
            'use_wall_proximity': False,
            'weight_wall_proximity': 0.2,
            'vdes_base_speed': 4.0,
            'vdes_lookahead_steps': 8,
            'vdes_min_speed': 1.0,
            'wheelbase': 0.33,
            'lookahead_distance': 1.5,
            'max_velocity_error': 4.0,
            'max_yaw_rate_error': 4.0,
            'max_action_diff': 0.4,
            'max_track_error': 0.5,
            'min_speed': -0.1,
            'max_lateral_accel': 10.0,
        }

    def set_centerline(self, centerline: np.ndarray):
        self.centerline = centerline
        self._prev_centerline_idx = None
        self._prev_s = None
        self._precompute_speed_profile()
        self._precompute_arc_length()


    def set_raceline(self, raceline_points, raceline_speeds):
        """Override v_des_profile with raceline speed column.
        raceline_points: (N, 2) array of [x, y] raceline waypoints.
        raceline_speeds: (N,) array of optimal velocities at each point."""
        self._raceline_points = np.asarray(raceline_points, dtype=np.float64)
        self._raceline_speeds = np.asarray(raceline_speeds, dtype=np.float64)
        if self.centerline is not None:
            mapped_v_des = np.zeros(len(self.centerline))
            for i, (cx, cy) in enumerate(self.centerline):
                dists = np.linalg.norm(self._raceline_points - np.array([cx, cy]), axis=1)
                nearest_rl_idx = int(np.argmin(dists))
                mapped_v_des[i] = float(self._raceline_speeds[nearest_rl_idx])
            self._v_des_profile = mapped_v_des

    def reset_episode(self):
        self._prev_centerline_idx = None
        self._prev_s = None

    def _precompute_arc_length(self):
        if self.centerline is None or len(self.centerline) < 2:
            self._centerline_s = None
            self._track_length = None
            return
        seg_lengths = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
        self._centerline_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self._track_length = float(self._centerline_s[-1])

    def _nearest_cl_idx(self, x, y):
        dx = self.centerline[:, 0] - x
        dy = self.centerline[:, 1] - y
        return int(np.argmin(dx**2 + dy**2))

    def _nearest_cl_idx_and_dist(self, x, y):
        dx = self.centerline[:, 0] - x
        dy = self.centerline[:, 1] - y
        dist_sq = dx**2 + dy**2
        idx = int(np.argmin(dist_sq))
        return idx, float(np.sqrt(dist_sq[idx]))

    def _compute_progress(self, x, y):
        if self.centerline is None or self._centerline_s is None:
            return 0.0
        idx = self._nearest_cl_idx(x, y)
        curr_s = float(self._centerline_s[idx])
        if self._prev_s is None:
            self._prev_s = curr_s
            return 0.0
        delta = curr_s - self._prev_s
        if delta < -(self._track_length / 2.0):
            delta += self._track_length
        self._prev_s = curr_s
        return float(np.clip(delta, 0.0, 1.0))

    def _compute_centerline_error(self, x, y):
        if self.centerline is None:
            return 0.0
        _, dist = self._nearest_cl_idx_and_dist(x, y)
        threshold = float(self.config.get('centerline_error_threshold', 0.1))
        return max(0.0, dist - threshold)

    def _compute_speed_turn_penalty(self, action, current_velocity):
        """Penalize cornering at high speed.
        Returns 0 when |steering| <= threshold; otherwise scales with steer*velocity."""
        if action is None:
            return 0.0
        steer_abs = abs(float(action[0]))
        threshold = float(self.config.get('speed_turn_steer_threshold', 0.15))
        if steer_abs <= threshold:
            return 0.0
        v = max(0.0, float(current_velocity))
        # Penalty = (steer_excess) * velocity; normalized roughly to [0, 1] range
        steer_excess = steer_abs - threshold
        return float(steer_excess * v / 2.0)

    def _precompute_speed_profile(self):
        cl = self.centerline
        n = len(cl)
        lookahead_steps = int(self.config.get('vdes_lookahead_steps', 8))
        base_speed = float(self.config.get('vdes_base_speed', 4.0))
        min_speed = float(self.config.get('vdes_min_speed', 1.0))
        wheelbase = float(self.config.get('wheelbase', 0.33))
        lookahead_dist = float(self.config.get('lookahead_distance', 1.5))
        v_des = np.zeros(n)
        for i in range(n):
            target_idx = (i + lookahead_steps) % n
            dx = cl[target_idx, 0] - cl[i, 0]
            dy = cl[target_idx, 1] - cl[i, 1]
            dx_h = cl[(i+1)%n, 0] - cl[(i-1)%n, 0]
            dy_h = cl[(i+1)%n, 1] - cl[(i-1)%n, 1]
            heading = float(np.arctan2(dy_h, dx_h))
            alpha = float(np.arctan2(dy, dx)) - heading
            while alpha > np.pi:  alpha -= 2*np.pi
            while alpha < -np.pi: alpha += 2*np.pi
            steering = np.arctan2(2.0 * wheelbase * np.sin(alpha), lookahead_dist)
            steering = float(np.clip(steering, -0.4, 0.4))
            speed = base_speed * (1.0 - 0.5 * abs(steering) / 0.4)
            v_des[i] = max(speed, min_speed)
        self._v_des_profile = v_des

    def compute_smoothness_penalty(self, action, prev_action):
        if prev_action is None:
            return 0.0
        diff = np.linalg.norm(np.array(action) - np.array(prev_action))
        normalized = diff / self.config['max_action_diff']
        return -self.config['weight_smoothness'] * normalized

    def compute_wall_proximity_penalty(self, lidar_obs):
        if lidar_obs is None or len(lidar_obs) == 0:
            return 0.0
        min_dist = float(np.min(lidar_obs))
        return -self.config['weight_wall_proximity'] * (1.0 - min_dist)

    def compute_step_reward(
        self,
        action,
        prev_action,
        current_velocity: float = 0.0,
        desired_velocity: float = 0.0,
        current_yaw_rate: float = 0.0,
        desired_yaw_rate: float = 0.0,
        poses_x: float = None,
        poses_y: float = None,
        lidar_obs=None,
        collision: bool = False,
    ) -> Tuple[float, Dict]:

        v_des = desired_velocity
        if (self._v_des_profile is not None
                and poses_x is not None and poses_y is not None):
            cl_idx = self._nearest_cl_idx(poses_x, poses_y)
            v_des = float(self._v_des_profile[cl_idx])

        # 1. Velocity tracking
        vel_error = abs(float(current_velocity) - v_des)
        vel_error_norm = np.clip(vel_error / self.config['max_velocity_error'], 0, 1)
        velocity_tracking = -self.config['weight_velocity_tracking'] * vel_error_norm

        # 2. Yaw rate tracking
        yaw_error = abs(float(current_yaw_rate) - float(desired_yaw_rate))
        yaw_error_norm = np.clip(yaw_error / self.config['max_yaw_rate_error'], 0, 1)
        yaw_tracking = -self.config['weight_yaw_rate_tracking'] * yaw_error_norm

        # 3. Smoothness
        smoothness = self.compute_smoothness_penalty(action, prev_action)

        # 4. Survival
        alive = self.config['weight_alive']

        # 5. Collision
        collision_penalty = -self.config['weight_collision'] if collision else 0.0

        # 6. Progress along centerline (Hell et al.)
        progress = 0.0
        if poses_x is not None and poses_y is not None:
            progress = self._compute_progress(poses_x, poses_y)
        progress_term = self.config.get('weight_progress', 0.5) * progress

        # 7. Centerline error penalty (Sun et al.)
        centerline_err = 0.0
        if poses_x is not None and poses_y is not None:
            centerline_err = self._compute_centerline_error(poses_x, poses_y)
        centerline_term = -self.config.get('weight_centerline_error', 0.1) * centerline_err

        # 9. Speed-turn coupling penalty (corner safety)
        speed_turn = self._compute_speed_turn_penalty(action, current_velocity)
        speed_turn_term = -self.config.get('weight_speed_turn', 0.5) * speed_turn

        # 8. Wall proximity (Paper 2)
        wall = 0.0
        if self.config.get('use_wall_proximity', False) and lidar_obs is not None:
            wall = self.compute_wall_proximity_penalty(lidar_obs)

        total = (velocity_tracking + yaw_tracking + smoothness
                 + alive + collision_penalty
                 + progress_term + centerline_term + speed_turn_term + wall)

        if np.random.rand() < 0.001:
            print(
                f"[REWARD DEBUG] "
                f"vel_track={velocity_tracking:.4f} "
                f"[v={float(current_velocity):.2f} v_des={v_des:.2f}], "
                f"yaw={yaw_tracking:.4f}, "
                f"smooth={smoothness:.4f}, "
                f"alive={alive:.4f}, "
                f"collision={collision_penalty:.4f}, "
                f"progress={progress_term:.4f} [d={progress:.3f}m], "
                f"ce_err={centerline_term:.4f} [d={centerline_err:.3f}m], "
                f"wall={wall:.4f}, "
                f"speed_turn={speed_turn_term:.4f} [steer*v={speed_turn:.3f}], "
                f"total={total:.4f}"
            )

        breakdown = {
            'velocity_tracking': velocity_tracking,
            'yaw_rate_tracking': yaw_tracking,
            'smoothness': smoothness,
            'alive': alive,
            'collision': collision_penalty,
            'progress': progress_term,
            'centerline_error': centerline_term,
            'wall_proximity': wall,
            'speed_turn': speed_turn_term,
            'v_des': v_des,
            'total': total,
        }
        return total, breakdown

    def compute_episode_termination(self, state, track_error=None, lateral_accel=None):
        velocity = state.get('velocity', 0.0)
        if velocity < self.config['min_speed']:
            return True, f"stuck_reversing (v={velocity:.2f})"
        return False, None

    def config_summary(self):
        variant = "Physics-Limit-Aware RMA" if self.config.get('use_wall_proximity') else "Baseline RMA"
        return f"Reward: Zhang et al. + {variant} + progress + centerline_error"
