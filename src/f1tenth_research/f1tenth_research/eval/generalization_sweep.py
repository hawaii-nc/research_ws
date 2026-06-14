"""
Generalization Sweep & Evaluation
==================================

Implements Zhang et al. (2025) Section IV-C: generalization evaluation across
difficulty levels δ = 0, 0.5, 1, 2, 4, 8 with three policies:

1. RMA Model: Deployed policy (π + φ) with learned adaptation
2. Fixed-Param Baseline: Policy/controller trained only on nominal params
3. Oracle Baseline: Expert controller with ground-truth physics (upper bound)

Metrics:
- Success rate: Fraction of episodes completed without crash
- Position tracking error: Lateral deviation from centerline
- Velocity tracking error: Speed command vs actual
- Episode length: Steps before termination

Outputs: Plots comparable to Zhang et al. Figure 6
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
import yaml

from ..envs import F1TenthRMAEnv, PhysicsRandomizer, SampleMode
from ..models import RMAActorCritic, AdaptationModule
from ..experts import PurePursuitExpert


class GeneralizationEvaluator:
    """
    Evaluates policies across generalization difficulty levels.
    
    Reference: Zhang et al. (2025) Section IV-C, Figure 6
    """
    
    def __init__(
        self,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        results_dir: str = 'eval_results',
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            device: 'cuda' or 'cpu'
            results_dir: Directory to save results and plots
        """
        self.device = torch.device(device)
        self.config = config
        self.results_dir = results_dir
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize environment
        self.env = F1TenthRMAEnv(config=config)
        
        # Randomizer
        self.randomizer = PhysicsRandomizer(config.get('randomization', {}))
        
        # Initialize models (will be loaded)
        self.actor_critic = None
        self.adaptation_module = None
    
    def load_rma_model(
        self,
        actor_critic_checkpoint: str,
        adaptation_checkpoint: str,
    ):
        """
        Load trained RMA policy (π + φ).
        
        Args:
            actor_critic_checkpoint: Path to Phase 1 actor-critic
            adaptation_checkpoint: Path to Phase 2 adaptation module
        """
        # Load actor-critic
        self.actor_critic = RMAActorCritic().to(self.device)
        ckpt = torch.load(actor_critic_checkpoint, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt['actor_critic'])
        self.actor_critic.eval()
        
        # Load adaptation module
        adaptation_config = self.config.get('networks', {}).get('adaptation', {})
        self.adaptation_module = AdaptationModule(
            state_action_dim=adaptation_config.get('state_action_dim', 7),
            history_window=adaptation_config.get('history_window', 10),
            intrinsics_dim=adaptation_config.get('intrinsics_dim', 8),
        ).to(self.device)
        
        ckpt = torch.load(adaptation_checkpoint, map_location=self.device)
        self.adaptation_module.load_state_dict(ckpt['adaptation'])
        self.adaptation_module.eval()
        
        print("RMA model loaded successfully")
    
    def evaluate_rma_policy(
        self,
        delta: float,
        num_episodes: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate RMA policy (π + φ) at generalization level δ.
        
        Args:
            delta: Difficulty level (0, 0.5, 1, 2, 4, 8)
            num_episodes: Number of episodes to run
            
        Returns:
            Dictionary with success_rate, avg_position_error, etc.
        """
        assert self.actor_critic is not None, "RMA model not loaded"
        assert self.adaptation_module is not None, "Adaptation module not loaded"
        
        print(f"\n[RMA Evaluation] δ = {delta}, {num_episodes} episodes")
        
        results = {
            'success_count': 0,
            'episode_lengths': [],
            'position_errors': [],
            'velocity_errors': [],
            'episodes_data': [],
        }
        
        with torch.no_grad():
            for episode in range(num_episodes):
                obs, info = self.env.reset(
                    options={'training': False, 'delta': delta}
                )
                done = False
                step = 0
                episode_reward = 0.0
                
                # Build state-action history for adaptation
                history = []
                
                while not done and step < self.env.max_episode_steps:
                    obs_tensor = torch.from_numpy(
                        obs if isinstance(obs, np.ndarray) else np.array(obs)
                    ).float().to(self.device)
                    
                    # Estimate intrinsics using adaptation module φ
                    if len(history) >= self.config['networks']['adaptation']['history_window']:
                        hist_array = np.array(history[-self.config['networks']['adaptation']['history_window']:])
                        hist_tensor = torch.from_numpy(hist_array).unsqueeze(0).float().to(self.device)
                        estimated_z = self.adaptation_module(hist_tensor).squeeze(0)
                    else:
                        # Use zero intrinsics if history not full yet
                        estimated_z = torch.zeros(8).to(self.device)
                    
                    # Get action from policy π
                    action, _ = self.actor_critic.get_action_and_value(obs_tensor, estimated_z)
                    action_np = action.cpu().numpy()
                    
                    # Store for history
                    state_action = np.concatenate([obs, action_np])
                    history.append(state_action)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = self.env.step(action_np)
                    done = terminated or truncated
                    episode_reward += reward
                    step += 1
                
                # Record results
                success = not done or step >= self.env.max_episode_steps
                if success:
                    results['success_count'] += 1
                
                results['episode_lengths'].append(step)
                # Placeholder for position/velocity errors (would extract from info)
                results['position_errors'].append(0.0)
                results['velocity_errors'].append(0.0)
                
                if (episode + 1) % 10 == 0:
                    print(f"  Episode {episode + 1}/{num_episodes}: steps={step}, reward={episode_reward:.3f}")
        
        # Compute statistics
        success_rate = results['success_count'] / num_episodes
        avg_length = np.mean(results['episode_lengths'])
        avg_pos_error = np.mean(results['position_errors']) if len(results['position_errors']) > 0 else 0.0
        avg_vel_error = np.mean(results['velocity_errors']) if len(results['velocity_errors']) > 0 else 0.0
        
        return {
            'success_rate': success_rate,
            'avg_episode_length': avg_length,
            'avg_position_error': avg_pos_error,
            'avg_velocity_error': avg_vel_error,
            'raw_data': results,
        }
    
    def evaluate_fixed_param_baseline(
        self,
        delta: float,
        num_episodes: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate fixed-parameter baseline.
        
        Policy/controller trained only on nominal physics, not adapted to δ.
        This simulates a non-adaptive controller.
        
        Args:
            delta: Difficulty level
            num_episodes: Number of episodes
            
        Returns:
            Evaluation metrics
        """
        print(f"\n[Fixed-Param Baseline] δ = {delta}, {num_episodes} episodes")
        
        # For now, return placeholder results
        # In full implementation, would load a baseline policy trained only on δ=0
        warnings.warn("Fixed-param baseline not fully implemented - returning zeros")
        
        return {
            'success_rate': 0.0,
            'avg_episode_length': 0.0,
            'avg_position_error': 0.0,
            'avg_velocity_error': 0.0,
        }
    
    def evaluate_oracle_baseline(
        self,
        delta: float,
        num_episodes: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate oracle baseline (expert with ground-truth physics).
        
        This is the upper bound - expert controller receives true et
        and adjusts its control law accordingly.
        
        Args:
            delta: Difficulty level
            num_episodes: Number of episodes
            
        Returns:
            Evaluation metrics
        """
        print(f"\n[Oracle Baseline] δ = {delta}, {num_episodes} episodes")
        
        # TODO: Load expert controller and waypoints
        expert = None  # PurePursuitExpert(...)
        
        if expert is None:
            warnings.warn("Oracle expert not initialized - returning zeros")
            return {
                'success_rate': 0.0,
                'avg_episode_length': 0.0,
                'avg_position_error': 0.0,
                'avg_velocity_error': 0.0,
            }
        
        results = {
            'success_count': 0,
            'episode_lengths': [],
            'position_errors': [],
            'velocity_errors': [],
        }
        
        with torch.no_grad():
            for episode in range(num_episodes):
                obs, info = self.env.reset(
                    options={'training': False, 'delta': delta}
                )
                done = False
                step = 0
                
                while not done and step < self.env.max_episode_steps:
                    # Convert obs to state dict for expert
                    state_dict = {
                        'position': (obs[0], obs[1]),
                        'yaw': obs[2],
                        'velocity': obs[3],
                        'yaw_rate': obs[4],
                    }
                    
                    # Get expert action with ground-truth physics
                    physics_params = info.get('physics_params', {})
                    action = expert.compute_action(state_dict, physics_params)
                    
                    obs, _, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    step += 1
                
                success = not done or step >= self.env.max_episode_steps
                if success:
                    results['success_count'] += 1
                
                results['episode_lengths'].append(step)
                results['position_errors'].append(0.0)
                results['velocity_errors'].append(0.0)
        
        success_rate = results['success_count'] / num_episodes
        avg_length = np.mean(results['episode_lengths'])
        
        return {
            'success_rate': success_rate,
            'avg_episode_length': avg_length,
            'avg_position_error': np.mean(results['position_errors']),
            'avg_velocity_error': np.mean(results['velocity_errors']),
        }
    
    def run_sweep(self, rma_checkpoint: str, adaptation_checkpoint: str):
        """
        Run full generalization sweep across all δ levels.
        
        Args:
            rma_checkpoint: Path to RMA actor-critic model
            adaptation_checkpoint: Path to adaptation module
        """
        # Load models
        self.load_rma_model(rma_checkpoint, adaptation_checkpoint)
        
        # Get delta levels from config
        eval_config = self.config.get('evaluation', {})
        delta_levels = eval_config.get('delta_levels', [0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
        episodes_per_delta = eval_config.get('episodes_per_delta', 100)
        
        # Storage for results
        results = {
            'rma': {},
            'fixed_param': {},
            'oracle': {},
            'config': self.config,
        }
        
        # Sweep
        for delta in delta_levels:
            print(f"\n{'='*60}")
            print(f"Evaluating at δ = {delta}")
            print(f"{'='*60}")
            
            # RMA policy
            results['rma'][delta] = self.evaluate_rma_policy(delta, episodes_per_delta)
            
            # Baselines (if enabled)
            if eval_config.get('include_fixed_param_baseline', True):
                results['fixed_param'][delta] = self.evaluate_fixed_param_baseline(delta, episodes_per_delta)
            
            if eval_config.get('include_oracle_baseline', True):
                results['oracle'][delta] = self.evaluate_oracle_baseline(delta, episodes_per_delta)
        
        # Save results
        self.save_results(results)
        
        # Plot results
        self.plot_results(results)
        
        print(f"\n[Evaluation Complete] Results saved to {self.results_dir}")
    
    def save_results(self, results: Dict):
        """Save evaluation results to JSON."""
        # Extract scalar metrics for JSON serialization
        clean_results = {
            'rma': {k: {mk: v for mk, v in vals.items() if mk != 'raw_data'} 
                   for k, vals in results['rma'].items()},
            'fixed_param': results['fixed_param'],
            'oracle': results['oracle'],
        }
        
        path = os.path.join(self.results_dir, 'results.json')
        with open(path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        print(f"Results saved to {path}")
    
    def plot_results(self, results: Dict):
        """
        Plot generalization curves (Zhang et al. Figure 6 style).
        
        Args:
            results: Results dictionary from run_sweep
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('F1Tenth RMA Generalization Evaluation', fontsize=16)
        
        # Extract delta levels and success rates
        deltas = sorted(results['rma'].keys())
        
        rma_success = [results['rma'][d]['success_rate'] for d in deltas]
        rma_lengths = [results['rma'][d]['avg_episode_length'] for d in deltas]
        rma_pos_error = [results['rma'][d]['avg_position_error'] for d in deltas]
        rma_vel_error = [results['rma'][d]['avg_velocity_error'] for d in deltas]
        
        # Success rate
        ax = axes[0, 0]
        ax.plot(deltas, rma_success, 'b-o', label='RMA', linewidth=2, markersize=8)
        if results['fixed_param']:
            fixed_success = [results['fixed_param'].get(d, {}).get('success_rate', 0) for d in deltas]
            ax.plot(deltas, fixed_success, 'r--s', label='Fixed-Param', linewidth=2, markersize=6)
        if results['oracle']:
            oracle_success = [results['oracle'].get(d, {}).get('success_rate', 0) for d in deltas]
            ax.plot(deltas, oracle_success, 'g:^', label='Oracle', linewidth=2, markersize=6)
        ax.set_xlabel('Difficulty Level (δ)')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate vs Generalization Difficulty')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1.05])
        
        # Episode length
        ax = axes[0, 1]
        ax.plot(deltas, rma_lengths, 'b-o', label='RMA', linewidth=2, markersize=8)
        ax.set_xlabel('Difficulty Level (δ)')
        ax.set_ylabel('Avg Episode Length (steps)')
        ax.set_title('Episode Length vs Generalization Difficulty')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Position error
        ax = axes[1, 0]
        ax.plot(deltas, rma_pos_error, 'b-o', label='RMA', linewidth=2, markersize=8)
        ax.set_xlabel('Difficulty Level (δ)')
        ax.set_ylabel('Avg Position Error (m)')
        ax.set_title('Position Tracking Error vs Generalization Difficulty')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Velocity error
        ax = axes[1, 1]
        ax.plot(deltas, rma_vel_error, 'b-o', label='RMA', linewidth=2, markersize=8)
        ax.set_xlabel('Difficulty Level (δ)')
        ax.set_ylabel('Avg Velocity Error (m/s)')
        ax.set_title('Velocity Tracking Error vs Generalization Difficulty')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, 'generalization_curves.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
        plt.close()


def main():
    """Entry point for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generalization Evaluation')
    parser.add_argument('--rma_checkpoint', type=str, required=True,
                       help='Path to RMA actor-critic checkpoint')
    parser.add_argument('--adaptation_checkpoint', type=str, required=True,
                       help='Path to adaptation module checkpoint')
    parser.add_argument('--config', type=str, default='configs/rma_config.yaml',
                       help='Path to config YAML')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')
    parser.add_argument('--results_dir', type=str, default='eval_results',
                       help='Directory to save results')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Evaluation on device: {device}")
    
    # Run evaluation
    evaluator = GeneralizationEvaluator(config, device=device, results_dir=args.results_dir)
    evaluator.run_sweep(args.rma_checkpoint, args.adaptation_checkpoint)


if __name__ == '__main__':
    main()
