"""
Neural Network Architectures for F1Tenth RMA
=============================================

Implements three networks matching Zhang et al. (2025) Section III-C:

1. Base Policy π(xt, zt) → at
   - 3-layer MLP, 256-dim hidden, ReLU
   - Input: state xt + intrinsics zt (8D)
   - Output: action at

2. Intrinsics Encoder μ(et) → zt
   - 2-layer MLP, 128-dim hidden, ReLU
   - Input: environmental parameters et
   - Output: 8D intrinsics vector zt
   - NOTE: Shares factor-encoding layer with value function (per Zhang)

3. Adaptation Module φ(history) → ẑt
   - Input: last k state-action pairs (~0.2s history, scaled to control freq)
   - Architecture: 1D CNN (channels/kernel/stride as per Zhang)
   - Output: estimated 8D intrinsics ẑt

All modules are PyTorch nn.Module subclasses for seamless integration with
PPO trainer and supervised learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Base Policy π(xt, zt) → at
    
    Zhang et al. (2025) Section III-C, Equation (3):
    π_θ(at | xt, zt)
    
    Architecture: 3-layer MLP with 256-dim hidden layers, ReLU activations.
    Input: concatenation of state xt (obs_dim) + intrinsics zt (8D)
    Output: action at (steering + throttle, 2D)
    """
    
    def __init__(self, obs_dim: int = 5, intrinsics_dim: int = 8, action_dim: int = 2):
        """
        Initialize policy network.
        
        Args:
            obs_dim: Dimension of state observations xt (default 5 for base signals)
            intrinsics_dim: Dimension of intrinsics vector zt (default 8 per Zhang)
            action_dim: Dimension of action output (default 2: steering + throttle)
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.intrinsics_dim = intrinsics_dim
        self.action_dim = action_dim
        
        input_dim = obs_dim + intrinsics_dim
        hidden_dim = 256
        
        # 3-layer MLP: input -> hidden1 -> hidden2 -> output
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Initialize weights (default is fine, but can use Xavier if desired)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Compute policy action.
        
        Args:
            obs: State observations xt, shape (batch_size, obs_dim) or (obs_dim,)
            intrinsics: Intrinsics vector zt, shape (batch_size, intrinsics_dim) or (intrinsics_dim,)
            
        Returns:
            Action logits at, shape (batch_size, action_dim) or (action_dim,)
        """
        # Concatenate obs and intrinsics
        combined = torch.cat([obs, intrinsics], dim=-1)
        # Bound output to [-1, 1] matching action_space (steering, throttle)
        return torch.tanh(self.network(combined))


class IntrinsicsEncoder(nn.Module):
    """
    Intrinsics Encoder μ(et) → zt
    
    Zhang et al. (2025) Section III-C, Equation (2):
    μ_φ(zt | et)
    
    Architecture: 2-layer MLP with 128-dim hidden layers, ReLU activations.
    Input: environmental/physics parameters et
    Output: 8D intrinsics vector zt
    
    NOTE: Per Zhang, this shares the factor-encoding layer with the value function.
    In the full PPO implementation, we'll use a shared encoder backbone.
    """
    
    def __init__(self, env_params_dim: int, intrinsics_dim: int = 8):
        """
        Initialize intrinsics encoder.
        
        Args:
            env_params_dim: Dimension of environmental parameters et
                           (depends on randomization, e.g., 7-10 for our setup)
            intrinsics_dim: Output dimension (default 8 per Zhang)
        """
        super().__init__()
        
        self.env_params_dim = env_params_dim
        self.intrinsics_dim = intrinsics_dim
        
        hidden_dim = 128
        
        # 2-layer MLP: input -> hidden -> output
        self.network = nn.Sequential(
            nn.Linear(env_params_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, intrinsics_dim),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, env_params: torch.Tensor) -> torch.Tensor:
        """
        Encode environmental parameters to intrinsics.
        
        Args:
            env_params: Environmental parameters et, shape (batch_size, env_params_dim) or (env_params_dim,)
            
        Returns:
            Intrinsics vector zt, shape (batch_size, intrinsics_dim) or (intrinsics_dim,)
        """
        return self.network(env_params)


class AdaptationModule(nn.Module):
    """
    Adaptation Module φ(history) → ẑt
    
    Zhang et al. (2025) Section III-D: learns to estimate intrinsics from
    recent state-action history without ground-truth physics parameters.
    
    Input: last k state-action pairs, where k covers ~0.2s of history
           (scaled to actual control frequency, not raw step count)
    
    Architecture: 1D CNN followed by flattening and linear projection
    - Channel depths: [32, 32, 8] as per Zhang (adapted slightly for F1Tenth freq)
    - Kernels: [5, 5] (Zhang: [32,32,5,1], [32,32,5,1])
    - Strides: [1, 1]
    
    Output: estimated 8D intrinsics ẑt
    """
    
    def __init__(
        self,
        state_action_dim: int = 7,  # obs_dim (5) + action_dim (2)
        history_window: int = 10,    # k steps of history (~0.2s at 50Hz)
        intrinsics_dim: int = 8,
    ):
        """
        Initialize adaptation module.
        
        Args:
            state_action_dim: Dimension of each state-action pair (obs + action)
            history_window: Number of past timesteps to use (k)
            intrinsics_dim: Output dimension (8D)
        """
        super().__init__()
        
        self.state_action_dim = state_action_dim
        self.history_window = history_window
        self.intrinsics_dim = intrinsics_dim
        
        # 1D CNN layers
        # Input shape: (batch_size, state_action_dim, history_window)
        # Output: flattened features -> intrinsics_dim
        
        self.cnn_layers = nn.Sequential(
            # Conv1 output: (batch, 32, history_window - kernel + 1)
            nn.Conv1d(
                in_channels=state_action_dim,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,  # Preserve length
            ),
            nn.ReLU(),
            
            # Conv2
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            
            # Conv3
            nn.Conv1d(
                in_channels=32,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
        )
        
        # Fully connected layers
        # After CNN: (batch, 8, history_window)
        fc_input_dim = 8 * history_window
        
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, intrinsics_dim),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Estimate intrinsics from state-action history.
        
        Args:
            history: State-action history, shape (batch_size, history_window, state_action_dim)
                    or (history_window, state_action_dim)
        
        Returns:
            Estimated intrinsics ẑt, shape (batch_size, intrinsics_dim) or (intrinsics_dim,)
        """
        # Handle unbatched input
        if history.dim() == 2:
            history = history.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Transpose to (batch, state_action_dim, history_window) for Conv1d
        history = history.transpose(1, 2)
        
        # Apply CNN
        cnn_out = self.cnn_layers(history)
        
        # Flatten
        flattened = cnn_out.reshape(cnn_out.shape[0], -1)
        
        # Apply FC layers
        intrinsics = self.fc_layers(flattened)
        
        if squeeze_output:
            intrinsics = intrinsics.squeeze(0)
        
        return intrinsics


class ValueNetwork(nn.Module):
    """
    Value Function Vθ(xt, zt) → scalar
    
    Estimates state-intrinsics value for PPO critic.
    Architecture: shared encoder with policy, separate head for value output.
    
    Per Zhang et al. (2025), the encoder layers (computing factor representation)
    are shared across π, μ, and V for parameter efficiency.
    """
    
    def __init__(self, obs_dim: int = 5, intrinsics_dim: int = 8):
        """
        Initialize value network.
        
        Args:
            obs_dim: Dimension of state observations
            intrinsics_dim: Dimension of intrinsics vector
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.intrinsics_dim = intrinsics_dim
        
        input_dim = obs_dim + intrinsics_dim
        hidden_dim = 256
        
        # Same structure as policy (3-layer MLP)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Single value output
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimate.
        
        Args:
            obs: State observations, shape (batch_size, obs_dim) or (obs_dim,)
            intrinsics: Intrinsics vector, shape (batch_size, intrinsics_dim) or (intrinsics_dim,)
            
        Returns:
            Value estimate, shape (batch_size,) or scalar
        """
        combined = torch.cat([obs, intrinsics], dim=-1)
        return self.network(combined).squeeze(-1)


class RMAActorCritic(nn.Module):
    """
    Combined Actor-Critic module for PPO training.
    
    Encapsulates:
    - Policy π (actor) for action selection
    - Value V (critic) for advantage estimation
    - Intrinsics encoder μ for environment adaptation
    
    Per Zhang et al., π and V share the same encoder backbone.
    """
    
    def __init__(
        self,
        obs_dim: int = 5,
        action_dim: int = 2,
        intrinsics_dim: int = 8,
        env_params_dim: int = 7,
    ):
        """
        Initialize actor-critic module.
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            intrinsics_dim: Intrinsics vector dimension
            env_params_dim: Environment parameters dimension
        """
        super().__init__()
        
        self.policy = PolicyNetwork(obs_dim, intrinsics_dim, action_dim)
        self.value = ValueNetwork(obs_dim, intrinsics_dim)
        self.encoder = IntrinsicsEncoder(env_params_dim, intrinsics_dim)
    
    def forward(
        self,
        obs: torch.Tensor,
        env_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for actor-critic.
        
        Args:
            obs: Observations
            env_params: Environmental parameters (used to compute intrinsics)
            
        Returns:
            Tuple of (action_logits, value_estimate)
        """
        intrinsics = self.encoder(env_params)
        action = self.policy(obs, intrinsics)
        value = self.value(obs, intrinsics)
        return action, value
    
    def get_intrinsics(self, env_params: torch.Tensor) -> torch.Tensor:
        """Get intrinsics for given environment parameters."""
        return self.encoder(env_params)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action and value for given observations and intrinsics.
        
        Used during rollout when intrinsics are already known.
        """
        action = self.policy(obs, intrinsics)
        value = self.value(obs, intrinsics)
        return action, value


__all__ = [
    'PolicyNetwork',
    'IntrinsicsEncoder',
    'AdaptationModule',
    'ValueNetwork',
    'RMAActorCritic',
]
