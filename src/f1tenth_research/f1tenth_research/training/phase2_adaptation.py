"""
Phase 2 Training: Adaptation Module φ Supervised Learning
=========================================================

Implements Zhang et al. (2025) Section III-D: training the adaptation module φ
to estimate intrinsics from state-action history without ground-truth physics.

Process:
1. Roll out Phase 1 trained policy (π + μ) across randomized environments
2. Collect (state-action history, ground-truth zt) pairs
3. Train φ via MSE loss: L = ||z - ẑ||^2 using Adam optimizer

The adaptation module φ becomes the deployment-time intrinsics estimator,
replacing μ which required ground-truth physics parameters.

Reference: Zhang et al. (2025) Algorithm 2
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque, defaultdict
from typing import Dict, Tuple, List, Optional
import yaml

from ..models import AdaptationModule, RMAActorCritic
from ..envs import F1TenthRMAEnv, PhysicsRandomizer, SampleMode


class AdaptationDataBuffer:
    """
    Circular buffer for storing state-action-intrinsics experiences.
    """
    
    def __init__(self, max_size: int = 1_000_000, window_size: int = 10):
        """
        Initialize buffer.
        
        Args:
            max_size: Maximum number of transitions to store
            window_size: Window size for adaptation history (k steps)
        """
        self.max_size = max_size
        self.window_size = window_size
        
        self.state_action_history = deque(maxlen=max_size)
        self.intrinsics = []
        self.size = 0
    
    def add(self, state_action_pair: np.ndarray, intrinsic: np.ndarray):
        """
        Add a state-action pair and corresponding intrinsic.
        
        Args:
            state_action_pair: State-action vector (state + action)
            intrinsic: Intrinsics vector (8D)
        """
        self.state_action_history.append(state_action_pair)
        self.intrinsics.append(intrinsic)
        self.size = min(self.size + 1, self.max_size)
    
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a minibatch of state-action histories and intrinsics.
        
        Returns:
            Tuple of:
            - histories: (batch_size, window_size, state_action_dim)
            - intrinsics: (batch_size, 8)
        """
        if len(self.state_action_history) < self.window_size:
            raise RuntimeError("Not enough data in buffer for window_size")
        
        # Sample starting indices
        valid_starts = max(0, len(self.state_action_history) - self.window_size)
        start_indices = np.random.randint(0, valid_starts + 1, size=batch_size)
        
        histories = []
        intrinsics_batch = []
        
        for start_idx in start_indices:
            # Get window of state-action pairs
            window = [
                self.state_action_history[start_idx + i]
                for i in range(self.window_size)
            ]
            histories.append(np.stack(window))
            intrinsics_batch.append(self.intrinsics[start_idx + self.window_size - 1])
        
        histories = torch.from_numpy(np.stack(histories)).float()
        intrinsics_batch = torch.from_numpy(np.stack(intrinsics_batch)).float()
        
        return histories, intrinsics_batch
    
    def __len__(self) -> int:
        return self.size


class Phase2Trainer:
    """
    Supervised training for adaptation module φ.
    
    Reference: Zhang et al. (2025) Section III-D
    """
    
    def __init__(
        self,
        actor_critic_checkpoint: str,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        log_dir: str = 'logs/phase2',
        checkpoint_dir: str = 'checkpoints/phase2',
    ):
        """
        Initialize Phase 2 trainer.
        
        Args:
            actor_critic_checkpoint: Path to Phase 1 trained model
            config: Configuration dictionary
            device: 'cuda' or 'cpu'
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for checkpoints
        """
        self.device = torch.device(device)
        self.config = config
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load Phase 1 trained policy
        self.actor_critic = RMAActorCritic().to(self.device)
        checkpoint = torch.load(actor_critic_checkpoint, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.actor_critic.eval()  # Set to eval mode (no grad updates)
        
        # Initialize adaptation module
        phase2_config = config.get('phase2_training', {})
        adaptation_config = config.get('networks', {}).get('adaptation', {})
        
        self.adaptation = AdaptationModule(
            state_action_dim=adaptation_config.get('state_action_dim', 7),
            history_window=adaptation_config.get('history_window', 10),
            intrinsics_dim=adaptation_config.get('intrinsics_dim', 8),
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.adaptation.parameters(),
            lr=phase2_config.get('learning_rate', 1.0e-3),
        )
        
        # Data buffer
        self.buffer = AdaptationDataBuffer(
            max_size=phase2_config.get('buffer_size', 1_000_000),
            window_size=adaptation_config.get('history_window', 10),
        )
        
        # Environment
        self.env = F1TenthRMAEnv(config=config)
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        self.metrics = defaultdict(list)
    
    def collect_data(self, num_episodes: int):
        """
        Collect state-action-intrinsics data by rolling out Phase 1 policy.
        
        Args:
            num_episodes: Number of episodes to collect
        """
        print(f"[Data Collection] Collecting {num_episodes} episodes...")
        
        with torch.no_grad():
            for episode in range(num_episodes):
                obs, info = self.env.reset(options={'training': True})
                done = False
                step = 0
                
                # Build state-action history for this episode
                history = deque(maxlen=self.buffer.window_size)
                
                while not done:
                    # Convert obs to tensor
                    obs_tensor = torch.from_numpy(obs if isinstance(obs, np.ndarray) else np.array(obs)).float().to(self.device)
                    
                    # Get ground-truth intrinsics from encoder μ
                    env_params = info.get('physics_params', {})
                    env_params_list = [
                        env_params.get(k, 0) for k in [
                            'grip_factor', 'mass_scale', 'inertia_scale',
                            'motor_steering_scale', 'motor_drive_scale',
                            'delay_steering', 'delay_drive'
                        ]
                    ]
                    env_params_tensor = torch.from_numpy(np.array(env_params_list)).float().to(self.device)
                    
                    # Get ground-truth intrinsics
                    with torch.no_grad():
                        intrinsics = self.actor_critic.get_intrinsics(env_params_tensor)
                    
                    # Get action from policy
                    with torch.no_grad():
                        mean, _ = self.actor_critic.policy(obs_tensor, intrinsics)
                        action = mean
                    
                    # Build state-action pair
                    state_action = np.concatenate([obs, action.cpu().numpy()])
                    history.append(state_action)
                    
                    # Add to buffer (only if history is full)
                    if len(history) == self.buffer.window_size:
                        self.buffer.add(state_action, intrinsics.cpu().numpy())
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
                    done = terminated or truncated
                    step += 1
                
                if (episode + 1) % max(1, num_episodes // 10) == 0:
                    print(f"  Episode {episode + 1}/{num_episodes}, Buffer size: {len(self.buffer)}")
        
        print(f"[Data Collection] Complete. Buffer has {len(self.buffer)} transitions")
    
    def train_epoch(self, num_epochs: int, batch_size: int):
        """
        Train adaptation module for multiple epochs.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for SGD
        """
        loss_fn = nn.MSELoss()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Mini-batch training
            while num_batches < len(self.buffer) // batch_size:
                try:
                    histories, intrinsics = self.buffer.get_batch(batch_size)
                except RuntimeError:
                    # Not enough data yet
                    break
                
                histories = histories.to(self.device)
                intrinsics = intrinsics.to(self.device)
                
                # Forward pass
                estimated_intrinsics = self.adaptation(histories)
                loss = loss_fn(estimated_intrinsics, intrinsics)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"[Epoch {epoch}] Loss: {avg_loss:.6f}")
                self.writer.add_scalar('train/mse_loss', avg_loss, self.global_step)
    
    def evaluate(self, num_episodes: int = 50) -> Dict[str, float]:
        """
        Evaluate adaptation module on held-out test episodes.
        
        Compares estimated intrinsics ẑ vs ground-truth z.
        
        Args:
            num_episodes: Number of test episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"[Evaluation] Evaluating on {num_episodes} test episodes...")
        
        errors = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                obs, info = self.env.reset(options={'training': True})
                done = False
                
                # Build state-action history
                history = deque(maxlen=self.buffer.window_size)
                
                while not done:
                    obs_tensor = torch.from_numpy(obs if isinstance(obs, np.ndarray) else np.array(obs)).float().to(self.device)
                    
                    # Get ground-truth intrinsics
                    env_params = info.get('physics_params', {})
                    env_params_list = [
                        env_params.get(k, 0) for k in [
                            'grip_factor', 'mass_scale', 'inertia_scale',
                            'motor_steering_scale', 'motor_drive_scale',
                            'delay_steering', 'delay_drive'
                        ]
                    ]
                    env_params_tensor = torch.from_numpy(np.array(env_params_list)).float().to(self.device)
                    
                    ground_truth_z = self.actor_critic.get_intrinsics(env_params_tensor)
                    
                    # Get action
                    mean, _ = self.actor_critic.policy(obs_tensor, ground_truth_z)
                    action = mean
                    
                    # Build history
                    state_action = np.concatenate([obs, action.cpu().numpy()])
                    history.append(state_action)
                    
                    # Estimate intrinsics if history is full
                    if len(history) == self.buffer.window_size:
                        hist_tensor = torch.from_numpy(np.stack(history)).unsqueeze(0).float().to(self.device)
                        estimated_z = self.adaptation(hist_tensor).squeeze(0)
                        
                        # Compute error
                        error = torch.norm(estimated_z - ground_truth_z).item()
                        errors.append(error)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
                    done = terminated or truncated
        
        metrics = {
            'mean_error': np.mean(errors) if len(errors) > 0 else float('inf'),
            'std_error': np.std(errors) if len(errors) > 0 else 0.0,
            'min_error': np.min(errors) if len(errors) > 0 else 0.0,
            'max_error': np.max(errors) if len(errors) > 0 else 0.0,
        }
        
        print(f"[Evaluation] Mean error: {metrics['mean_error']:.6f}")
        for key, val in metrics.items():
            self.writer.add_scalar(f'eval/{key}', val, self.global_step)
        
        return metrics
    
    def train(self):
        """
        Main Phase 2 training loop.
        
        1. Collect data from Phase 1 policy
        2. Train adaptation module φ
        3. Evaluate and save checkpoints
        """
        phase2_config = self.config.get('phase2_training', {})
        
        # Data collection
        num_eval_episodes = phase2_config.get('num_eval_episodes', 500)
        self.collect_data(num_eval_episodes)
        
        # Training
        num_epochs = phase2_config.get('num_epochs', 100)
        batch_size = phase2_config.get('batch_size', 64)
        
        print(f"[Training] Training for {num_epochs} epochs with batch size {batch_size}...")
        self.train_epoch(num_epochs, batch_size)
        
        # Evaluation
        print("[Training] Evaluating...")
        metrics = self.evaluate()
        
        # Save final model
        self.save_checkpoint('final.pt')
        
        print("[Training Complete] Phase 2 training finished")
    
    def save_checkpoint(self, filename: str):
        """Save adaptation module checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'adaptation': self.adaptation.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load adaptation module checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.adaptation.load_state_dict(checkpoint['adaptation'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.global_step = checkpoint['global_step']
        print(f"Checkpoint loaded from {path}")


def main():
    """Entry point for Phase 2 training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 2 Training: Adaptation Module')
    parser.add_argument('--phase1_checkpoint', type=str, required=True,
                       help='Path to Phase 1 trained model')
    parser.add_argument('--config', type=str, default='configs/rma_config.yaml',
                       help='Path to config YAML')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Training on device: {device}")
    
    # Create trainer and train
    trainer = Phase2Trainer(args.phase1_checkpoint, config, device=device)
    trainer.train()


if __name__ == '__main__':
    main()
