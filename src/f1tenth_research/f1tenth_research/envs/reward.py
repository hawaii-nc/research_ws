"""
F1Tenth Reward Function for RMA Training
==========================================

Implements Zhang et al. (2025) Section II-C: composite reward function with four terms:
1. Output smoothness penalty: -||a_t - a_{t-1}||
2. Survival reward: δt (constant per-timestep)
3. Velocity tracking deviation: -||v_t - v_des||
4. Yaw-rate tracking deviation: -||ω_t - ω_des||

All weights are configurable via config dict.
"""

from typing import Dict, Tuple, Optional
import numpy as np


class RewardComputer:
    """
    Computes composite reward signal for F1Tenth RMA training.
    Maps (state, action, physics_params) -> scalar reward.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize reward computer with configurable weights.
        
        Args:
            config: Dictionary with reward weights and scaling parameters
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """
        Default reward weights and parameters.
        Zhang et al. (2025) uses these in ablation studies.
        """
        return {
            # Term weights (sum should be ~1.0 or normalized separately)
            'weight_smoothness': 0.1,           # Output smoothness penalty
            'weight_survival': 1.0,             # Per-timestep survival bonus
            'weight_velocity_tracking': 0.5,    # Velocity tracking error penalty
            'weight_yaw_rate_tracking': 0.3,    # Yaw rate tracking error penalty
            'weight_progress': 0.8,             # Forward progress reward (prevents near-zero throttle exploit)
            'min_progress_speed': 0.5,          # Speed below which progress reward is zero
            'max_progress_speed': 4.0,          # Speed at which progress reward is maximal
            
            # Scaling parameters
            'max_action_diff': 0.4,             # Max steering angle for normalization
            'max_velocity_error': 2.0,          # Max v error for normalization
            'max_yaw_rate_error': 1.0,          # Max ω error for normalization
            
            # Episode termination conditions (hard constraints)
            'max_track_error': 0.5,             # Off-track threshold (meters)
            'min_speed': -0.5,                  # Minimum reverse speed before stopping
            'max_lateral_accel': 10.0,          # Max lateral acceleration (m/s^2)
        }
    
    def compute_smoothness_penalty(
        self,
        action: np.ndarray,
        prev_action: Optional[np.ndarray]
    ) -> float:
        """
        Output smoothness penalty: -||a_t - a_{t-1}||
        
        Encourages smooth control outputs to avoid jerkiness.
        
        Args:
            action: Current action [steering, throttle] shape (2,)
            prev_action: Previous action [steering, throttle] or None if first step
            
        Returns:
            Penalty term (negative value)
        """
        if prev_action is None:
            return 0.0
        
        # Compute L2 norm of action difference
        action_diff = np.linalg.norm(action - prev_action)
        
        # Normalize by max action magnitude for scale-invariance
        max_diff = self.config['max_action_diff']
        normalized_diff = action_diff / max_diff
        
        # Apply weight and return negative (penalty)
        penalty = -self.config['weight_smoothness'] * normalized_diff
        return penalty
    
    def compute_survival_reward(self) -> float:
        """
        Survival reward: δt (constant per-timestep bonus)
        
        Zhang et al. (2025): encourages agent to stay on track.
        In physical racing, trajectory length (time) is critical.
        
        Returns:
            Constant reward per timestep
        """
        return self.config['weight_survival']
    
    def compute_velocity_tracking_penalty(
        self,
        current_velocity: float,
        desired_velocity: float
    ) -> float:
        """
        Velocity tracking deviation penalty: -||v_t - v_des||
        
        Penalizes deviation from desired velocity setpoint.
        
        Args:
            current_velocity: Current longitudinal velocity (m/s)
            desired_velocity: Desired velocity command (m/s)
            
        Returns:
            Penalty term (negative value)
        """
        velocity_error = abs(current_velocity - desired_velocity)
        
        # Normalize by max error threshold
        max_error = self.config['max_velocity_error']
        normalized_error = velocity_error / max_error
        
        # Clip to [0, 1] to avoid unbounded penalties
        normalized_error = np.clip(normalized_error, 0, 1)
        
        penalty = -self.config['weight_velocity_tracking'] * normalized_error
        return penalty
    
    def compute_yaw_rate_tracking_penalty(
        self,
        current_yaw_rate: float,
        desired_yaw_rate: float
    ) -> float:
        """
        Yaw-rate tracking deviation penalty: -||ω_t - ω_des||
        
        Zhang et al. (2025) uses torque tracking over angular velocity.
        For F1Tenth, yaw rate (ω) is the most direct low-level signal available.
        
        Args:
            current_yaw_rate: Current yaw rate (rad/s)
            desired_yaw_rate: Desired yaw rate (rad/s)
            
        Returns:
            Penalty term (negative value)
        """
        yaw_rate_error = abs(current_yaw_rate - desired_yaw_rate)
        
        # Normalize by max error threshold
        max_error = self.config['max_yaw_rate_error']
        normalized_error = yaw_rate_error / max_error
        
        # Clip to [0, 1]
        normalized_error = np.clip(normalized_error, 0, 1)
        
        penalty = -self.config['weight_yaw_rate_tracking'] * normalized_error
        return penalty
    
    def compute_progress_reward(self, current_velocity: float) -> float:
        """
        Forward progress reward: incentivizes actual movement.

        Prevents the degenerate solution where the policy commands
        near-zero throttle (desired_velocity ~ 0, current_velocity ~ 0,
        velocity_error ~ 0) to exploit perfect velocity-tracking scores
        while barely moving.

        Returns reward proportional to forward speed, capped at
        max_progress_speed. Zero below min_progress_speed.
        """
        min_speed = self.config.get('min_progress_speed', 0.5)
        max_speed = self.config.get('max_progress_speed', 4.0)
        weight = self.config.get('weight_progress', 0.8)

        if current_velocity < min_speed:
            return 0.0
        normalized = min((current_velocity - min_speed) / (max_speed - min_speed), 1.0)
        return weight * normalized

    def compute_step_reward(
        self,
        action: np.ndarray,
        prev_action: Optional[np.ndarray],
        current_velocity: float,
        desired_velocity: float,
        current_yaw_rate: float,
        desired_yaw_rate: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute composite reward for a single timestep.
        
        Zhang et al. (2025) Section II-C: R(at) = weighted sum of four terms.
        
        Args:
            action: Current action [steering_cmd, throttle_cmd]
            prev_action: Previous action (for smoothness penalty)
            current_velocity: Measured longitudinal velocity
            desired_velocity: Velocity setpoint
            current_yaw_rate: Measured yaw rate (rad/s)
            desired_yaw_rate: Yaw rate setpoint (rad/s)
            
        Returns:
            Tuple of:
            - total_reward: scalar reward for this step
            - reward_breakdown: dict with individual term contributions
        """
        # Compute each term
        smoothness = self.compute_smoothness_penalty(action, prev_action)
        survival = self.compute_survival_reward()
        velocity_track = self.compute_velocity_tracking_penalty(
            current_velocity, desired_velocity
        )
        yaw_track = self.compute_yaw_rate_tracking_penalty(
            current_yaw_rate, desired_yaw_rate
        )
        progress = self.compute_progress_reward(current_velocity)

        # Composite reward (sum of weighted terms)
        total_reward = smoothness + survival + velocity_track + yaw_track + progress
        
        # Return breakdown for logging
        breakdown = {
            'smoothness': smoothness,
            'survival': survival,
            'velocity_tracking': velocity_track,
            'yaw_rate_tracking': yaw_track,
            'progress': progress,
            'total': total_reward,
        }
        
        return total_reward, breakdown
    
    def compute_episode_termination(
        self,
        state: Dict,
        track_error: Optional[float] = None,
        lateral_accel: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if episode should terminate (hard constraints).
        
        Args:
            state: Current state dict
            track_error: Lateral deviation from centerline (meters)
            lateral_accel: Computed lateral acceleration (m/s^2)
            
        Returns:
            Tuple of:
            - done: bool, whether episode is finished
            - reason: str, termination reason (or None if continuing)
        """
        # Check off-track
        if track_error is not None:
            max_error = self.config['max_track_error']
            if track_error > max_error:
                return True, f"off_track (error={track_error:.2f}m)"
        
        # Check min speed (reverse limit)
        velocity = state.get('velocity', 0.0)
        if velocity < self.config['min_speed']:
            return True, f"stuck_reversing (v={velocity:.2f})"
        
        # Check lateral acceleration (rollover/loss of traction)
        if lateral_accel is not None:
            max_accel = self.config['max_lateral_accel']
            if lateral_accel > max_accel:
                return True, f"excessive_lateral_accel ({lateral_accel:.1f}m/s²)"
        
        return False, None
    
    def config_summary(self) -> str:
        """
        Return human-readable summary of reward configuration.
        """
        lines = [
            "=== Reward Function Configuration ===",
            f"Smoothness weight: {self.config['weight_smoothness']:.3f}",
            f"Survival weight: {self.config['weight_survival']:.3f}",
            f"Velocity tracking weight: {self.config['weight_velocity_tracking']:.3f}",
            f"Yaw rate tracking weight: {self.config['weight_yaw_rate_tracking']:.3f}",
            f"Max track error: {self.config['max_track_error']:.2f}m",
            f"Max velocity error: {self.config['max_velocity_error']:.2f}m/s",
            f"Max yaw rate error: {self.config['max_yaw_rate_error']:.2f}rad/s",
        ]
        return "\n".join(lines)
