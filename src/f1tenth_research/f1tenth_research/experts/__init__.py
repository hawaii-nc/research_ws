"""
Expert Controller Module for F1Tenth RMA
=========================================

Implements expert controllers (IL target) that take ground-truth randomized physics
parameters as input and adapt their control law accordingly.

This mirrors Zhang et al. (2025) PD* expert, which had access to ground-truth
model parameters of each sampled quadcopter. For F1Tenth, we implement:
- Pure Pursuit with grip-aware speed modulation
- Future: MPCC (Model Predictive Contouring Control)

Reference: Zhang et al. (2025) Section II-B - Expert Policy PD*
"""

import numpy as np
from typing import Dict, Tuple
import warnings


class F1TenthExpertController:
    """
    Base class for F1Tenth expert controllers.
    
    Takes ground-truth physics parameters et and adapts control law.
    Outputs: expert action a_exp = [steering_cmd, throttle_cmd]
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize expert controller.
        
        Args:
            config: Dictionary with controller hyperparameters
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration - override in subclasses."""
        return {}
    
    def compute_action(
        self,
        state: Dict,
        physics_params: Dict,
    ) -> np.ndarray:
        """
        Compute expert action given state and physics parameters.
        
        Args:
            state: Current state dict containing:
              - 'position': (x, y) coordinates
              - 'yaw': heading angle (rad)
              - 'velocity': longitudinal velocity (m/s)
              - 'yaw_rate': angular velocity (rad/s)
            physics_params: Randomized physics parameters et containing:
              - 'grip_factor': tire friction scaling
              - 'mass_scale': chassis mass scaling
              - 'motor_steering_scale': steering effectiveness
              - 'motor_drive_scale': drive motor effectiveness
              - etc.
              
        Returns:
            Expert action [steering_cmd, throttle_cmd]
        """
        raise NotImplementedError("Subclasses must implement compute_action()")


class PurePursuitExpert(F1TenthExpertController):
    """
    Pure Pursuit expert controller with grip-aware speed modulation.
    
    Extends standard Pure Pursuit with adaptation to grip_factor:
    - Higher grip (c closer to 1.0): enable aggressive steering, maintain higher speed
    - Lower grip (c closer to 0.4): reduce speed, use gentler steering to maintain traction
    
    Reference: Pure Pursuit control law (Coulter, 1992) + Zhang et al. adaptation
    """
    
    def __init__(self, config: Dict = None, waypoints: np.ndarray = None):
        """
        Initialize Pure Pursuit expert.
        
        Args:
            config: Controller config dict
            waypoints: Nx2 array of reference waypoints (x, y)
        """
        super().__init__(config)
        self.waypoints = waypoints if waypoints is not None else np.array([])
        
        if len(self.waypoints) == 0:
            warnings.warn("PurePursuitExpert initialized without waypoints")
    
    def _default_config(self) -> Dict:
        """Default Pure Pursuit parameters."""
        return {
            'wheelbase': 0.33,              # F1Tenth wheelbase (meters)
            'lookahead_distance_base': 1.5, # Base lookahead distance (meters)
            'nominal_speed': 2.0,           # Nominal cruising speed (m/s)
            'max_steering': 0.4189,         # Max steering angle (rad)
            'min_speed': 0.5,               # Minimum speed (m/s)
            
            # Grip-aware modulation
            'grip_aware': True,
            'grip_speed_scale': {
                # grip_factor -> speed multiplier
                # At grip_factor=1.0: full nominal speed
                # At grip_factor=0.4: reduced speed for traction
                0.4: 0.6,   # Low grip: 60% nominal speed
                0.7: 0.85,  # Medium grip: 85% nominal speed
                1.0: 1.0,   # High grip: 100% nominal speed
            },
            'grip_steering_scale': {
                # grip_factor -> steering angle multiplier
                # Gentler steering on low grip to avoid skidding
                0.4: 0.7,
                0.7: 0.85,
                1.0: 1.0,
            },
        }
    
    def compute_action(
        self,
        state: Dict,
        physics_params: Dict,
    ) -> np.ndarray:
        """
        Compute Pure Pursuit action with grip-aware modulation.
        
        Args:
            state: Current state {position, yaw, velocity, yaw_rate, ...}
            physics_params: Physics params including grip_factor
            
        Returns:
            [steering_cmd, throttle_cmd]
        """
        if len(self.waypoints) == 0:
            # Fallback: no waypoints, return neutral action
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # Extract state
        x, y = state.get('position', (0, 0))
        yaw = state.get('yaw', 0.0)
        current_velocity = state.get('velocity', 0.0)
        
        # Find lookahead point
        target = self._find_lookahead_point(x, y)
        
        # Compute steering command via Pure Pursuit
        steering = self._compute_pure_pursuit_steering(
            x, y, yaw, target
        )
        
        # Adapt for grip factor
        grip_factor = physics_params.get('grip_factor', 1.0)
        
        if self.config['grip_aware']:
            # Scale steering angle based on grip
            steering_scale = self._interpolate_grip_scale(
                grip_factor, self.config['grip_steering_scale']
            )
            steering *= steering_scale
            
            # Compute speed command based on grip
            speed_scale = self._interpolate_grip_scale(
                grip_factor, self.config['grip_speed_scale']
            )
            desired_speed = self.config['nominal_speed'] * speed_scale
        else:
            desired_speed = self.config['nominal_speed']
        
        # Clamp steering to physical limits
        steering = np.clip(steering, -self.config['max_steering'], self.config['max_steering'])
        
        # Compute throttle command to reach desired speed
        # Simple: throttle = (desired_speed - current_velocity) * gain
        # Negative throttle = brake/coast
        speed_error = desired_speed - current_velocity
        throttle = np.clip(0.5 * speed_error, -1.0, 1.0)
        
        return np.array([steering, throttle], dtype=np.float32)
    
    def _find_lookahead_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Find the lookahead point on the reference trajectory.
        
        Searches forward along waypoints to find one at lookahead_distance.
        
        Args:
            x, y: Current position
            
        Returns:
            (target_x, target_y) coordinates
        """
        lookahead = self.config['lookahead_distance_base']
        
        # Compute distances to all waypoints
        dx = self.waypoints[:, 0] - x
        dy = self.waypoints[:, 1] - y
        distances = np.sqrt(dx**2 + dy**2)
        
        # Find closest waypoint
        closest_idx = np.argmin(distances)
        
        # Search forward to find one at lookahead distance
        n = len(self.waypoints)
        for i in range(n):
            idx = (closest_idx + i) % n
            if distances[idx] >= lookahead:
                return tuple(self.waypoints[idx])
        
        # Fallback: return closest
        return tuple(self.waypoints[closest_idx])
    
    def _compute_pure_pursuit_steering(
        self,
        x: float,
        y: float,
        yaw: float,
        target: Tuple[float, float],
    ) -> float:
        """
        Compute steering command via Pure Pursuit control law.
        
        ψ = atan2(2 * L * sin(α) / ℓ)
        where:
          L: wheelbase
          α: angle to target relative to heading
          ℓ: lookahead distance
        
        Args:
            x, y: Current position
            yaw: Current heading (rad)
            target: Target point (tx, ty)
            
        Returns:
            Steering angle command (rad)
        """
        tx, ty = target
        
        # Angle to target in world frame
        angle_to_target = np.arctan2(ty - y, tx - x)
        
        # Angle relative to current heading
        alpha = angle_to_target - yaw
        
        # Normalize to [-π, π]
        while alpha > np.pi:
            alpha -= 2 * np.pi
        while alpha < -np.pi:
            alpha += 2 * np.pi
        
        # Pure Pursuit formula
        wheelbase = self.config['wheelbase']
        lookahead = self.config['lookahead_distance_base']
        
        steering = np.arctan2(2.0 * wheelbase * np.sin(alpha), lookahead)
        return steering
    
    def _interpolate_grip_scale(
        self,
        grip_factor: float,
        scale_table: Dict[float, float],
    ) -> float:
        """
        Interpolate speed/steering scale factor based on grip_factor.
        
        Linear interpolation between defined points.
        
        Args:
            grip_factor: Current grip factor (typically 0.4-1.0)
            scale_table: {grip_value: scale_factor} mapping
            
        Returns:
            Interpolated scale factor
        """
        # Sort table by grip factor
        sorted_grips = sorted(scale_table.keys())
        
        # Clamp to range
        if grip_factor <= sorted_grips[0]:
            return scale_table[sorted_grips[0]]
        if grip_factor >= sorted_grips[-1]:
            return scale_table[sorted_grips[-1]]
        
        # Find bracketing points
        for i in range(len(sorted_grips) - 1):
            g1, g2 = sorted_grips[i], sorted_grips[i+1]
            if g1 <= grip_factor <= g2:
                # Linear interpolation
                frac = (grip_factor - g1) / (g2 - g1)
                scale1 = scale_table[g1]
                scale2 = scale_table[g2]
                return scale1 + frac * (scale2 - scale1)
        
        # Shouldn't reach here
        return 1.0


class MPCCExpert(F1TenthExpertController):
    """
    MPCC (Model Predictive Contouring Control) expert.
    
    More advanced than Pure Pursuit: uses explicit model of vehicle dynamics
    and solves finite-horizon optimal control problem.
    
    TODO: Implement full MPC formulation with:
    - Kinematic bicycle model
    - QP-based trajectory optimization
    - Grip-aware cost function weights
    
    For now, placeholder that falls back to Pure Pursuit.
    """
    
    def __init__(self, config: Dict = None, waypoints: np.ndarray = None):
        super().__init__(config)
        self.waypoints = waypoints
        self._pure_pursuit_fallback = PurePursuitExpert(config, waypoints)
    
    def compute_action(
        self,
        state: Dict,
        physics_params: Dict,
    ) -> np.ndarray:
        """Fallback to Pure Pursuit for now."""
        return self._pure_pursuit_fallback.compute_action(state, physics_params)


__all__ = [
    'F1TenthExpertController',
    'PurePursuitExpert',
    'MPCCExpert',
]
