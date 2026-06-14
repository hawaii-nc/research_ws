"""
Domain Randomization Module for F1Tenth RMA
============================================

Implements Zhang et al. (2025) Section II-A: domain randomization with size factor c.
Maps physical parameters (et) -> intrinsic parameters (zt) via μ encoder.

Key components:
- Grip factor c ∈ [0,1]: scales tire friction coefficient μ
- Motor effectiveness factors: independent scaling per actuator (steering, drive)
- Mass/inertia scaling: correlated chassis parameter variation
- External disturbances: mid-episode randomized perturbations

Training range (δ=0.5 reference) stored as config. Supports sampling at arbitrary δ.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum


class SampleMode(Enum):
    """Sampling mode for physics parameters."""
    TRAIN = "train"           # Sample from δ=0.5 training range
    GENERALIZATION = "generalization"  # Sample at arbitrary δ


class PhysicsRandomizer:
    """
    Samples randomized physics parameters (et) for F1Tenth domain randomization.
    
    References: Zhang et al. (2025) Table I - quadcopter randomization ranges.
    Adapted for F1Tenth with analogous scaling factors.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize randomizer with training range (δ=0.5).
        
        Args:
            config: Dictionary with hyperparameters. If None, use defaults.
        """
        self.config = config or self._default_config()
        
        # Training range (δ=0.5 reference) - normalized nominal values
        # These define what "1.0" scaling factor means
        self.nominal_params = {
            'grip_factor': 1.0,              # μ tire friction nominal
            'mass_scale': 1.0,               # chassis mass scaling
            'inertia_scale': 1.0,            # rotational inertia scaling
            'motor_steering_scale': 1.0,     # steering servo effectiveness
            'motor_drive_scale': 1.0,        # drive motor effectiveness
            'delay_steering': 0.0,           # steering command delay (seconds)
            'delay_drive': 0.0,              # drive command delay (seconds)
        }
        
        # Training range bounds (Zhang's δ=0.5 reference)
        self.train_range = {
            'grip_factor': (0.4, 1.0),                      # c ∈ [0.4, 1.0] 
            'mass_scale': (0.8, 1.2),                       # ±20%
            'inertia_scale': (0.8, 1.2),                    # ±20%
            'motor_steering_scale': (0.8, 1.2),             # ±20%
            'motor_drive_scale': (0.8, 1.2),                # ±20%
            'delay_steering': (0.0, 0.05),                  # 0-50ms jitter
            'delay_drive': (0.0, 0.05),                     # 0-50ms jitter
        }
        
        # Correlation: if mass scales up, inertia scales similarly
        # (preserves rotational dynamics)
        self.correlated_params = {
            'mass_inertia': True,  # mass and inertia scale together
        }
    
    def _default_config(self) -> Dict:
        """Default hyperparameters."""
        return {
            'apply_post_scale_noise': True,  # ±20% uniform noise after scaling
            'noise_fraction': 0.20,           # ±20% noise magnitude
            'mid_episode_disturbance': True,  # randomly timed friction change
            'disturbance_timing': (0.3, 0.9),  # when (frac of episode) disturbance occurs
            'disturbance_magnitude': 0.3,     # friction reduction magnitude
        }
    
    def sample_training(self) -> Dict[str, float]:
        """
        Sample physics parameters from training distribution (δ=0.5).
        
        Returns:
            Dictionary of sampled parameters et = {grip_factor, mass_scale, ...}
        """
        params = {}
        
        # Sample base parameters independently
        for key, (lo, hi) in self.train_range.items():
            params[key] = np.random.uniform(lo, hi)
        
        # Apply correlations: mass and inertia should scale together
        if self.correlated_params['mass_inertia']:
            # Use the sampled mass_scale for inertia_scale as well,
            # with slight decorrelation (±5%)
            base = params['mass_scale']
            params['inertia_scale'] = base * np.random.uniform(0.95, 1.05)
        
        # Apply post-sampling noise (±20% uniform)
        if self.config['apply_post_scale_noise']:
            noise_mag = self.config['noise_fraction']
            for key in params:
                noise = np.random.uniform(-noise_mag, noise_mag)
                params[key] *= (1.0 + noise)
        
        return params
    
    def sample_at_generalization_level(self, delta: float) -> Dict[str, float]:
        """
        Sample physics parameters at generalization difficulty level δ.
        
        δ = 0: nominal (all parameters = 1.0)
        δ = 0.5: training range (default)
        δ = 1, 2, 4, 8: increasing generalization difficulty
        
        Maps: training range [min, max] -> extrapolated range at δ via linear scaling
        
        Args:
            delta: Generalization difficulty level (0 to 8+)
            
        Returns:
            Dictionary of sampled parameters at the given δ level
        """
        if delta == 0:
            # Nominal - no randomization
            return self.nominal_params.copy()
        
        params = {}
        
        # Linearly scale the training range bounds by δ relative to training (δ=0.5)
        # scaling_factor = δ / 0.5
        scale_factor = delta / 0.5
        
        for key, (train_lo, train_hi) in self.train_range.items():
            # Nominal value (typically in middle or at nominal)
            nominal = self.nominal_params[key]
            
            # Scaled bounds
            delta_lo = (train_lo - nominal) * scale_factor + nominal
            delta_hi = (train_hi - nominal) * scale_factor + nominal
            
            # Ensure no negative scales
            delta_lo = max(0.01, delta_lo)
            delta_hi = max(delta_lo + 0.01, delta_hi)
            
            params[key] = np.random.uniform(delta_lo, delta_hi)
        
        # Apply correlations
        if self.correlated_params['mass_inertia']:
            base = params['mass_scale']
            params['inertia_scale'] = base * np.random.uniform(0.95, 1.05)
        
        # Apply post-sampling noise
        if self.config['apply_post_scale_noise']:
            noise_mag = self.config['noise_fraction']
            for key in params:
                noise = np.random.uniform(-noise_mag, noise_mag)
                params[key] *= (1.0 + noise)
        
        return params
    
    def sample(self, mode: SampleMode = SampleMode.TRAIN, delta: float = 0.5) -> Dict[str, float]:
        """
        Unified sampling interface.
        
        Args:
            mode: TRAIN or GENERALIZATION
            delta: For GENERALIZATION mode, the difficulty level (0-8+)
            
        Returns:
            Sampled physics parameters et
        """
        if mode == SampleMode.TRAIN:
            return self.sample_training()
        elif mode == SampleMode.GENERALIZATION:
            return self.sample_at_generalization_level(delta)
        else:
            raise ValueError(f"Unknown sample mode: {mode}")
    
    def add_mid_episode_disturbance(
        self,
        params: Dict[str, float],
        episode_progress: float
    ) -> Dict[str, float]:
        """
        Randomly apply a mid-episode disturbance (friction change).
        
        Zhang et al. (2025) Section II-A: mid-episode randomized perturbations
        to simulate unexpected environmental changes.
        
        Args:
            params: Current physics parameters
            episode_progress: Fraction of episode elapsed [0, 1]
            
        Returns:
            Modified parameters dict (may include temporary disturbance flag)
        """
        if not self.config['mid_episode_disturbance']:
            params['mid_episode_disturbance'] = False
            params['disturbance_magnitude'] = 0.0
            return params
        
        timing_lo, timing_hi = self.config['disturbance_timing']
        disturbance_active = (
            timing_lo <= episode_progress <= timing_hi
            and np.random.rand() < 0.3  # 30% chance if in timing window
        )
        
        params['mid_episode_disturbance'] = disturbance_active
        if disturbance_active:
            # Reduce grip factor mid-episode (simulated wet track or payload shift)
            mag = self.config['disturbance_magnitude']
            params['grip_factor'] *= (1.0 - mag)
        else:
            params['disturbance_magnitude'] = 0.0
        
        return params
    
    def params_to_description(self, params: Dict[str, float]) -> str:
        """
        Human-readable description of sampled parameters.
        Useful for logging and debugging.
        """
        lines = [
            f"grip_factor={params['grip_factor']:.3f}",
            f"mass_scale={params['mass_scale']:.3f}",
            f"inertia_scale={params['inertia_scale']:.3f}",
            f"motor_steering_scale={params['motor_steering_scale']:.3f}",
            f"motor_drive_scale={params['motor_drive_scale']:.3f}",
            f"delay_steering={params['delay_steering']:.4f}s",
            f"delay_drive={params['delay_drive']:.4f}s",
        ]
        return ", ".join(lines)
