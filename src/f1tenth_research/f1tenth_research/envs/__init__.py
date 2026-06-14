"""
F1Tenth Gymnasium environment and domain randomization.
Implements Zhang et al. (2025) RMA framework for autonomous racing.
"""

from .randomization import PhysicsRandomizer, SampleMode
from .f1tenth_env import F1TenthRMAEnv

__all__ = [
    'PhysicsRandomizer',
    'SampleMode',
    'F1TenthRMAEnv',
]
