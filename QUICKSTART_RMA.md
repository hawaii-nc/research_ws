# Quick Start Guide: F1Tenth RMA Training & Deployment

## TL;DR - Get Running in 5 Minutes

### 1. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium pyyaml numpy matplotlib tensorboard
# Optional: pip install f1tenth_gym
```

### 2. Quick Test Run (Debug Mode)
```bash
cd /research_ws/src/f1tenth_research/f1tenth_research

# Phase 1: Quick test (100k steps instead of 100M)
python training/phase1_ppo_il.py \
    --config configs/rma_config.yaml \
    --device cuda \
    --debug

# Phase 2: Adaptation training
python training/phase2_adaptation.py \
    --phase1_checkpoint checkpoints/phase1/checkpoint_epoch_0.pt \
    --config configs/rma_config.yaml \
    --device cuda
```

### 3. Evaluate Generalization
```bash
python eval/generalization_sweep.py \
    --rma_checkpoint checkpoints/phase1/final.pt \
    --adaptation_checkpoint checkpoints/phase2/final.pt \
    --config configs/rma_config.yaml
```

Check `eval_results/generalization_curves.png` for results!

---

## Understanding the Output

### Phase 1 Training Logs
```
[Epoch 0] Step 0: Avg Reward = 12.345
[Epoch 10] Checkpoint saved
  reward_breakdown: {smoothness: -0.1, survival: 1.0, velocity_tracking: -0.3, ...}
  il_loss: 0.456
  total_loss: 12.789
```

**What it means:**
- Reward should increase over epochs (RL component winning)
- IL loss should decrease (imitation gets less important)
- Smoothness penalty should stay small (encourages smooth control)

### Phase 2 Evaluation
```
[Adaptation] Mean error: 0.0234 (±0.0156)
```

**What it means:**
- Lower = better. Error < 0.1 is very good.
- φ is learning to estimate intrinsics from history

### Generalization Sweep Results
```json
{
  "rma": {
    "0.0": {"success_rate": 0.98, "avg_episode_length": 987},
    "0.5": {"success_rate": 0.95, "avg_episode_length": 950},
    "1.0": {"success_rate": 0.87, "avg_episode_length": 820},
    "2.0": {"success_rate": 0.65, "avg_episode_length": 600},
    ...
  }
}
```

**What it means:**
- δ=0.0: Nominal (near-perfect)
- δ=0.5: Training range (still ~95% success)
- δ=2.0: 2× harder physics (65% success is good!)
- Plot should show graceful degradation (not cliff)

---

## Configuration Tuning Guide

### If Training is Unstable
**Symptom:** Reward bounces wildly or goes negative
**Fix:**
```yaml
phase1_training:
  ppo:
    learning_rate: 1.0e-4  # Lower LR
    entropy_coeff: 0.05    # Higher entropy (exploration)
  il:
    il_weight_decay: 0.0005  # Slower IL decay
```

### If Generalization is Poor
**Symptom:** δ=1.0 success rate drops to <30%
**Fix:**
```yaml
environment:
  randomization:
    noise_fraction: 0.30  # More noise
    mid_episode_disturbance: true
    disturbance_magnitude: 0.5
```

### If Training is Too Slow
**Symptom:** 1 epoch takes >10 minutes
**Fix:**
```yaml
phase1_training:
  num_envs: 4         # Fewer parallel envs
  rollout_steps: 512  # Shorter rollouts
  num_epochs: 5       # Fewer SGD epochs
```

---

## Common Workflows

### Full Training (100M timesteps, ~12 hours on A100)
```bash
# Edit config
sed -i 's/debug_mode: true/debug_mode: false/' configs/rma_config.yaml

# Phase 1
python training/phase1_ppo_il.py --config configs/rma_config.yaml --device cuda

# Phase 2 (after Phase 1 completes)
python training/phase2_adaptation.py \
    --phase1_checkpoint checkpoints/phase1/final.pt \
    --config configs/rma_config.yaml

# Evaluate
python eval/generalization_sweep.py \
    --rma_checkpoint checkpoints/phase1/final.pt \
    --adaptation_checkpoint checkpoints/phase2/final.pt
```

### Benchmark vs Baselines
Edit `eval/generalization_sweep.py`:
```python
eval_config = self.config.get('evaluation', {})
eval_config['include_fixed_param_baseline'] = True
eval_config['include_oracle_baseline'] = True
```

Then run evaluation to get comparison plot.

### Ablation Study: Remove IL
```yaml
phase1_training:
  il:
    enabled: false  # Train pure RL (no imitation)
```

### Deploy in Gazebo
```bash
source /opt/ros/foxy/setup.bash
cd /research_ws
colcon build --packages-select f1tenth_research

# Terminal 1: Launch sim
ros2 launch f1tenth_gym_ros gym_bridge_launch.py

# Terminal 2: Run RMA node
ros2 run f1tenth_research rma_deployment \
    --actor_critic checkpoints/phase1/final.pt \
    --adaptation checkpoints/phase2/final.pt
```

---

## Debugging: Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=logs/phase1 --port=6006
# Open http://localhost:6006
```

Check these curves:
- `train/avg_reward`: Should ↑ over time
- `train/policy_loss`: Should ↓ (stabilize)
- `train/il_loss`: Should ↓ then level off
- `train/entropy`: Should start high, ↓ as policy converges

### Print Physics Params
Edit `envs/f1tenth_env.py` to add:
```python
print(f"Physics: {self.randomizer.params_to_description(self.current_physics_params)}")
```

### Check Expert Performance
```python
# In Phase 1 trainer
expert_action = self.expert.compute_action(state, physics_params)
print(f"Expert action: {expert_action}, Policy action: {action.numpy()}")
print(f"IL loss: {((expert_action - action.numpy())**2).mean()}")
```

---

## Troubleshooting Reference

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| Out of memory | Batch size too large | Reduce `batch_size` to 128 |
| NaN loss | Learning rate too high | Reduce `learning_rate` to 1e-4 |
| Poor IL phase | Expert not generating actions | Check `expert.compute_action()` |
| Adaptation fails | Not enough history data | Increase `phase2_training.num_eval_episodes` |
| Deployment crashes | Model path wrong | Check `--actor_critic` and `--adaptation` paths |
| ROS2 errors | f1tenth_gym not installed | `pip install f1tenth_gym` |

---

## Paper Correspondence

While training, you can reference these sections of Zhang et al. (2025):

- **Domain randomization**: See paper Section II-A for quadcopter analog
- **Reward function**: Paper Section II-C equations (1)-(4)
- **Expert**: Paper Section II-B (our Pure Pursuit plays role of PD*)
- **Training loop**: Paper Algorithm 1 (Section II-D)
- **Adaptation module**: Paper Algorithm 2 (Section III-D)
- **Evaluation**: Paper Section IV-C and Figure 6

Each code file has inline comments mapping specific lines to paper sections/equations.

---

## Next Steps

After successful training:

1. **Run on real hardware:** Test deployment on physical F1Tenth (sim2real transfer)
2. **Add LiDAR:** Use full scan observations instead of odometry-only
3. **Compare to baselines:** Train fixed-param and oracle policies for comparison
4. **Hyperparameter sweep:** Grid search over reward weights
5. **Multi-track:** Train on multiple tracks simultaneously

---

## Support

- Check `README_RMA.md` for detailed documentation
- Each Python file has extensive docstrings mapping to Zhang et al.
- Config file `rma_config.yaml` has inline comments explaining each parameter
- TensorBoard logs give real-time training progress

Good luck! 🚗
