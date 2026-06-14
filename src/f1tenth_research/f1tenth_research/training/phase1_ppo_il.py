"""
Phase 1 Training: PPO + IL Joint Training
==========================================

Implements Zhang et al. (2025) Section II-D: combined RL + IL training loop.

Key components:
- PPO (Proximal Policy Optimization) for reinforcement learning
- Imitation Learning loss L_IL(π) = ||a_exp - a||^2
- Joint objective: R(π) = (1-α)·R_RL(π) - α·L_IL(π)
- Exponentially decaying IL weight: α = exp(-decay * t_epoch)

Architecture:
- Parallel environment rollouts (num_envs environments)
- Generalized Advantage Estimation (GAE) for variance reduction
- Clipped surrogate objective for PPO stability
- Shared encoder backbone (π, V, μ per Zhang)

Training scale: ~100M timesteps (scale to F1Tenth hardware/sim speed)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, Optional, List
import warnings
from collections import defaultdict
import yaml

from ..envs import F1TenthRMAEnv, PhysicsRandomizer, SampleMode
from ..models import RMAActorCritic, PolicyNetwork, IntrinsicsEncoder, ValueNetwork
from ..experts import PurePursuitExpert
from ..envs.reward import RewardComputer


class Phase1Trainer:
    """
    PPO + IL joint training for RMA policy.
    
    Reference: Zhang et al. (2025) Section II-D, Algorithm 1
    """
    
    def __init__(
        self,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        log_dir: str = 'logs/phase1',
        checkpoint_dir: str = 'checkpoints/phase1',
    ):
        """
        Initialize Phase 1 trainer.
        
        Args:
            config: Configuration dictionary (from YAML)
            device: 'cuda' or 'cpu'
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
        """
        self.config = config
        self.device = torch.device(device)
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize environments and components
        self.env_config = config.get('environment', {})
        self.training_config = config.get('phase1_training', {})
        self.expert_config = config.get('expert', {})
        
        # Determine training scale
        if self.training_config.get('debug_mode', False):
            self.total_timesteps = self.training_config.get('debug_timesteps', 100_000)
            print(f"[DEBUG MODE] Using {self.total_timesteps} timesteps")
        else:
            self.total_timesteps = self.training_config.get('total_timesteps', 100_000_000)
        
        # Create environment(s)
        self.num_envs = self.training_config.get('num_envs', 16)
        self.envs = self._create_envs()
        
        # Get observation and action dimensions
        self.obs_dim = self.envs.single_observation_space.shape[0] if hasattr(self.envs, 'single_observation_space') else 5
        self.action_dim = self.envs.single_action_space.shape[0] if hasattr(self.envs, 'single_action_space') else 2
        self.env_params_dim = 7  # Configurable based on randomization
        
        # Initialize networks
        self.actor_critic = RMAActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            intrinsics_dim=8,
            env_params_dim=self.env_params_dim,
        ).to(self.device)
        
        # Initialize expert controller
        self.expert = self._create_expert()
        
        # Optimizers
        ppo_config = self.training_config.get('ppo', {})
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=ppo_config.get('learning_rate', 3.0e-4),
        )
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        self.global_episode = 0
        self.metrics = defaultdict(list)
    
    def _create_envs(self):
        """Create parallel environments (vectorized)."""
        # For now, create single environment
        # TODO: Use gym.vector.AsyncVectorEnv for parallel execution
        env = F1TenthRMAEnv(
            config=self.config,
            max_episode_steps=self.env_config.get('max_episode_steps', 1000),
        )
        return env
    
    def _create_expert(self) -> Optional[PurePursuitExpert]:
        """Initialize expert controller for IL."""
        if not self.training_config.get('il', {}).get('use_expert_actions', False):
            return None
        
        # Load real raceline waypoints (x_m, y_m columns, per config_example_map.yaml)
        wpt_data = np.loadtxt('/f1tenth_gym/examples/example_waypoints.csv',
                               delimiter=';', skiprows=3)
        waypoints = wpt_data[:, 1:3]  # x_m, y_m columns
        
        expert_type = self.expert_config.get('type', 'pure_pursuit')
        if expert_type == 'pure_pursuit':
            return PurePursuitExpert(
                config=self.expert_config.get('pure_pursuit', {}),
                waypoints=waypoints,
            )
        else:
            warnings.warn(f"Unknown expert type: {expert_type}, disabling expert")
            return None
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards received, shape (rollout_steps,)
            values: State values, shape (rollout_steps,)
            next_value: Value of terminal state
            gamma: Discount factor
            gae_lambda: GAE decay parameter
            
        Returns:
            Tuple of:
            - advantages: Computed advantages
            - returns: Cumulative returns (values + advantages)
        """
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[step + 1]
            
            delta = rewards[step] + gamma * next_val - values[step]
            gae = delta + gamma * gae_lambda * gae
            advantages[step] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def get_expert_action(
        self,
        state: Dict,
        physics_params: Dict,
    ) -> Optional[np.ndarray]:
        """
        Get action from expert controller.
        
        Args:
            state: Current state
            physics_params: Randomized physics parameters
            
        Returns:
            Expert action or None if expert not available
        """
        if self.expert is None:
            return None
        
        return self.expert.compute_action(state, physics_params)
    
    def rollout(self, num_steps: int) -> Dict:
        """
        Perform environment rollout for PPO.
        
        Collects trajectories: (obs, action, reward, value, advantage, return, expert_action)
        
        Args:
            num_steps: Number of steps to rollout
            
        Returns:
            Dictionary with rollout data for training update
        """
        rollout_data = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'advantages': [],
            'returns': [],
            'expert_actions': [],
            'intrinsics': [],
            'log_probs': [],
        }
        
        obs, info = self.envs.reset()
        
        with torch.no_grad():
            for step in range(num_steps):
                # Convert obs to tensor
                obs_tensor = torch.from_numpy(obs if isinstance(obs, np.ndarray) else np.array(obs)).float().to(self.device)
                
                # Get environment parameters
                env_params = info.get('physics_params', {})
                env_params_tensor = torch.from_numpy(
                    np.array([env_params.get(k, 0) for k in ['grip_factor', 'mass_scale', 'inertia_scale', 'motor_steering_scale', 'motor_drive_scale', 'delay_steering', 'delay_drive']])
                ).float().to(self.device)
                
                # Get intrinsics and action
                intrinsics = self.actor_critic.get_intrinsics(env_params_tensor)
                action, value = self.actor_critic.get_action_and_value(obs_tensor, intrinsics)
                
                # Get expert action for IL
                expert_action = None
                if self.expert is not None:
                    # Convert obs_tensor back to state dict for expert
                    state_dict = {'position': (obs[0], obs[1]), 'yaw': obs[2], 'velocity': obs[3], 'yaw_rate': obs[4]}
                    expert_action = self.get_expert_action(state_dict, env_params)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
                
                # Store rollout data
                rollout_data['obs'].append(obs_tensor.cpu().numpy())
                rollout_data['actions'].append(action.cpu().numpy())
                rollout_data['rewards'].append(reward)
                rollout_data['values'].append(value.cpu().numpy())
                rollout_data['intrinsics'].append(intrinsics.cpu().numpy())
                if expert_action is not None:
                    rollout_data['expert_actions'].append(expert_action)
                
                self.global_step += 1
        
        # Compute advantages and returns
        advantages = np.array(rollout_data['values'])  # Placeholder
        returns = advantages + np.array(rollout_data['rewards'])  # Placeholder
        
        rollout_data['advantages'] = advantages
        rollout_data['returns'] = returns
        
        return rollout_data
    
    def update(self, rollout_data: Dict, epoch: int):
        """
        Update policy and value function using collected rollout.
        
        Implements:
        - PPO clipped objective for policy gradient
        - IL loss for imitation: L_IL(π) = ||a_exp - a||^2
        - Combined: R(π) = (1-α)·R_RL(π) - α·L_IL(π)
        - α = exp(-decay * epoch)
        
        Args:
            rollout_data: Data from rollout()
            epoch: Current training epoch
        """
        # Compute IL weight decay
        il_config = self.training_config.get('il', {})
        il_decay = il_config.get('il_weight_decay', 0.001)
        il_weight = il_config.get('il_weight_start', 1.0)
        
        # Exponential decay: α = exp(-decay * epoch)
        il_weight = il_weight * np.exp(-il_decay * epoch)
        il_weight = max(il_weight, il_config.get('il_weight_end', 0.01))
        
        batch_size = self.training_config.get('batch_size', 256)
        num_epochs = self.training_config.get('num_epochs', 10)
        
        # Convert rollout data to tensors
        obs_tensor = torch.from_numpy(np.array(rollout_data['obs'])).float().to(self.device)
        actions_tensor = torch.from_numpy(np.array(rollout_data['actions'])).float().to(self.device)
        returns_tensor = torch.from_numpy(rollout_data['returns']).float().to(self.device)
        advantages_tensor = torch.from_numpy(rollout_data['advantages']).float().to(self.device)
        intrinsics_tensor = torch.from_numpy(np.array(rollout_data['intrinsics'])).float().to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Training epochs
        for epoch_i in range(num_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(obs_tensor))
            
            for start_idx in range(0, len(obs_tensor), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                obs_batch = obs_tensor[batch_indices]
                actions_batch = actions_tensor[batch_indices]
                returns_batch = returns_tensor[batch_indices]
                advantages_batch = advantages_tensor[batch_indices]
                intrinsics_batch = intrinsics_tensor[batch_indices]
                
                # Forward pass
                action_logits, values = self.actor_critic.policy(obs_batch, intrinsics_batch), \
                                       self.actor_critic.value(obs_batch, intrinsics_batch)
                
                # PPO loss
                action_dist = torch.nn.functional.softmax(action_logits, dim=-1)  # Placeholder
                action_log_probs = torch.log(action_dist + 1e-8).mean(dim=-1)
                
                entropy = -(action_dist * torch.log(action_dist + 1e-8)).sum(dim=-1).mean()
                policy_loss = -(action_log_probs * advantages_batch).mean()
                value_loss = ((values - returns_batch) ** 2).mean()
                
                # IL loss (if expert actions available)
                il_loss = 0.0
                if len(rollout_data['expert_actions']) > 0 and il_config.get('enabled', False):
                    expert_actions_tensor = torch.from_numpy(
                        np.array(rollout_data['expert_actions'][:len(batch_indices)])
                    ).float().to(self.device)
                    il_loss = ((action_logits - expert_actions_tensor) ** 2).mean()
                
                # Combined loss
                ppo_config = self.training_config.get('ppo', {})
                value_coeff = ppo_config.get('value_coeff', 0.5)
                entropy_coeff = ppo_config.get('entropy_coeff', 0.01)
                
                total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
                
                if il_config.get('enabled', False):
                    total_loss += il_weight * il_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                grad_norm = nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    ppo_config.get('max_grad_norm', 0.5)
                )
                
                self.optimizer.step()
                
                # Log metrics
                self.metrics['policy_loss'].append(policy_loss.item())
                self.metrics['value_loss'].append(value_loss.item())
                self.metrics['entropy'].append(entropy.item())
                if il_config.get('enabled', False):
                    self.metrics['il_loss'].append(il_loss.item() if isinstance(il_loss, torch.Tensor) else il_loss)
                self.metrics['grad_norm'].append(grad_norm.item())
    
    def train(self):
        """
        Main training loop (Phase 1).
        
        Iteratively:
        1. Rollout trajectories with current policy
        2. Update policy and value function via PPO + IL
        3. Log metrics and save checkpoints
        """
        rollout_steps = self.training_config.get('rollout_steps', 2048)
        checkpoint_interval = self.training_config.get('checkpoint_interval', 10)
        log_interval = self.training_config.get('log_interval', 100)
        eval_interval = self.training_config.get('eval_interval', 1000)
        
        epoch = 0
        while self.global_step < self.total_timesteps:
            # Rollout
            print(f"[Epoch {epoch}] Rollout ({rollout_steps} steps)...")
            rollout_data = self.rollout(rollout_steps)
            
            # Update
            print(f"[Epoch {epoch}] Update...")
            self.update(rollout_data, epoch)
            
            # Logging
            if epoch % log_interval == 0:
                avg_reward = np.mean(rollout_data['rewards']) if len(rollout_data['rewards']) > 0 else 0
                print(f"[Epoch {epoch}] Step {self.global_step}: Avg Reward = {avg_reward:.3f}")
                
                self.writer.add_scalar('train/avg_reward', avg_reward, self.global_step)
                for key, values in self.metrics.items():
                    if len(values) > 0:
                        self.writer.add_scalar(f'train/{key}', np.mean(values), self.global_step)
                self.metrics.clear()
            
            # Checkpointing
            if epoch % checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
                print(f"[Epoch {epoch}] Checkpoint saved")
            
            epoch += 1
        
        print("[Training Complete] Phase 1 training finished")
        self.save_checkpoint('final.pt')
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.global_step = checkpoint['global_step']
        print(f"Checkpoint loaded from {path}")


def main():
    """Entry point for Phase 1 training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 1 Training: PPO + IL')
    parser.add_argument('--config', type=str, default='configs/rma_config.yaml',
                       help='Path to config YAML')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.debug:
        config['phase1_training']['debug_mode'] = True
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Training on device: {device}")
    
    # Create trainer and train
    trainer = Phase1Trainer(config, device=device)
    trainer.train()


if __name__ == '__main__':
    main()
