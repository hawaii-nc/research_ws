# F1Tenth RMA: Randomized Model Adaptation for Autonomous Racing

**A faithful adaptation of Zhang et al. (2025), "A Learning-Based Quadcopter Controller With Extreme Adaptation"** (IEEE Trans. Robotics) **to F1Tenth autonomous racing cars.**

This project implements the complete RMA training framework: domain randomization → joint PPO+IL training → supervised adaptation learning → generalization evaluation.

---

## Overview

**Problem:** Trained control policies fail when physics parameters change (tire grip, mass, actuator effectiveness).

**Solution:** RMA combines three key ideas:
1. **Domain Randomization** (et): Sample physics variations during training
2. **Intrinsics Encoder** (μ): Learn to predict task-relevant parameters from environment signals
3. **Adaptation Module** (φ): Learn online parameter estimation from state-action history

**Result:** Single policy (π + φ) generalizes to unseen physics (difficulty levels δ = 0.5, 1, 2, 4, 8×).

---

## Project Structure

```
f1tenth_research/
├── envs/                      # Environment & domain randomization
│   ├── randomization.py       # PhysicsRandomizer (size factor c, grip factor)
│   ├── f1tenth_env.py         # F1TenthRMAEnv (Gymnasium wrapper)
│   ├── reward.py              # 4-term composite reward (Zhang Section II-C)
│   └── __init__.py
│
├── experts/                   # Expert controllers (IL target)
│   ├── __init__.py            # PurePursuitExpert, MPCCExpert
│   │                          # Takes ground-truth physics params
│   │                          # (mirrors Zhang's PD* expert)
│   └── [expert implementations]
│
├── models/                    # Neural network architectures
│   ├── __init__.py            # PolicyNetwork π, IntrinsicsEncoder μ
│   │                          # AdaptationModule φ, ValueNetwork V
│   └── [PyTorch nn.Module implementations]
│
├── training/                  # Training loops
│   ├── phase1_ppo_il.py       # PPO + IL (Section II-D)
│   │                          # Combined RL + imitation loss
│   │                          # Exponentially decaying IL weight
│   ├── phase2_adaptation.py   # Supervised adaptation training (Section III-D)
│   │                          # Trains φ on collected (history, zt) pairs
│   └── __init__.py
│
├── eval/                      # Evaluation & generalization sweep
│   ├── generalization_sweep.py # Evaluates at δ = 0, 0.5, 1, 2, 4, 8
│   │                          # Plots vs. fixed-param & oracle baselines
│   │                          # (Zhang Figure 6 reproduction)
│   └── __init__.py
│
├── gazebo_deployment/        # ROS 2 deployment node
│   ├── rma_deployment_node.py # Subscribes to /odom, /scan
│   │                          # Publishes to /drive (Ackermann)
│   │                          # Runtime: φ(history) → ẑ, π(xt, ẑ) → at
│   └── __init__.py
│
├── configs/                  # Hyperparameters (YAML)
│   └── rma_config.yaml      # All tunable params for domain, reward,
│                            # networks, training, evaluation
│
├── [existing files]
│   ├── il_driver.py         # ROS 2 node for IL deployment (outdated)
│   ├── pure_pursuit.py      # Original Pure Pursuit (non-adaptive)
│   ├── recorder_node.py
│   ├── teleop_node.py
│   └── setup.py
```

---

## Key Components Explained

### 1. Domain Randomization (`envs/randomization.py`)

Samples physics parameters (et) from training range (δ=0.5 reference):

```python
randomizer = PhysicsRandomizer(config)

# Training range
params = randomizer.sample(mode=SampleMode.TRAIN)
# → {grip_factor∈[0.4,1.0], mass_scale∈[0.8,1.2], ...}

# Generalization level (for evaluation)
params = randomizer.sample(mode=SampleMode.GENERALIZATION, delta=2.0)
# → Scaled range at δ=2.0 (harder difficulty)
```

**Key parameters:**
- `grip_factor c`: Tire friction scaling (analogous to quadcopter thrust coefficient)
- `mass_scale`, `inertia_scale`: Correlated chassis dynamics
- `motor_steering_scale`, `motor_drive_scale`: Actuator effectiveness
- `delay_steering`, `delay_drive`: Command delays (sensors to actuators)
- Post-sampling ±20% uniform noise
- Mid-episode disturbances (friction change, payload shift)

Reference: **Zhang et al. Section II-A**, Table I (quadcopter → F1Tenth adaptation)

### 2. Reward Function (`envs/reward.py`)

Zhang et al. Section II-C: four-term composite reward

```python
R = (smooth penalty) + (survival bonus) + (velocity track error) + (yaw rate track error)
R = -||at - at-1|| + δt - ||vt - vdes|| - ||ωt - ωdes||
```

**Configurable weights:**
- `weight_smoothness`: 0.1 (penalize jerky actions)
- `weight_survival`: 1.0 (bonus for staying on track)
- `weight_velocity_tracking`: 0.5 (speed regulation)
- `weight_yaw_rate_tracking`: 0.3 (heading stability)

Reference: **Zhang et al. Section II-C**, Equations (1)-(4)

### 3. Expert Controller (`experts/__init__.py`)

**Critical:** Expert takes ground-truth physics params as input, adapts control law.

```python
expert = PurePursuitExpert(config, waypoints)
expert_action = expert.compute_action(state, physics_params)
# physics_params = {grip_factor, mass_scale, ...}
```

**Grip-aware adaptation:**
- **High grip (c=1.0)**: Aggressive steering, nominal speed
- **Low grip (c=0.4)**: Gentle steering (avoid skidding), reduced speed

Mirrors Zhang's **PD* expert** which had access to ground-truth model parameters.

Reference: **Zhang et al. Section II-B**

### 4. Neural Networks (`models/__init__.py`)

Three PyTorch networks matching Zhang exactly:

#### Policy π(xt, zt) → at
```python
policy = PolicyNetwork(obs_dim=5, intrinsics_dim=8, action_dim=2)
# 3-layer MLP: 256-dim hidden, ReLU
# Input: state xt + intrinsics zt (concatenated)
# Output: action (steering, throttle)
```
Reference: **Zhang et al. Section III-C**, Eq (3)

#### Intrinsics Encoder μ(et) → zt
```python
encoder = IntrinsicsEncoder(env_params_dim=7, intrinsics_dim=8)
# 2-layer MLP: 128-dim hidden, ReLU
# Maps physics params et → 8D intrinsics zt
# Ground-truth version (Phase 1 training)
```
Reference: **Zhang et al. Section III-C**, Eq (2)

#### Adaptation Module φ(history) → ẑt
```python
adaptation = AdaptationModule(
    state_action_dim=7,      # obs(5) + action(2)
    history_window=10,       # k steps, ~0.2s @ 50Hz
    intrinsics_dim=8
)
# 1D CNN: 3 conv layers [32,32,8] channels, kernel=5
# Input: last k state-action pairs
# Output: estimated intrinsics ẑt (no ground-truth physics needed!)
```
Reference: **Zhang et al. Section III-D**

#### Value Network V(xt, zt) → ℜ
```python
value = ValueNetwork(obs_dim=5, intrinsics_dim=8)
# Same architecture as π (shared encoder per Zhang)
# Used for PPO advantage estimation
```

### 5. Phase 1 Training: PPO + IL (`training/phase1_ppo_il.py`)

**Joint training** combining RL (PPO) with imitation learning.

```python
trainer = Phase1Trainer(config, device='cuda')
trainer.train()
```

**Key elements:**

1. **Rollout:** Collect trajectories with randomized physics
2. **IL Loss:** `L_IL(π) = ||a_exp - a||²` (imitation)
3. **RL Loss:** PPO clipped objective (policy gradient)
4. **Combined:** `R(π) = (1-α)·R_RL(π) - α·L_IL(π)`
5. **Decay:** `α = exp(-0.001 * epoch)` (IL weight exponentially decays)

**Training scale:** ~100M timesteps (scale to F1Tenth sim speed)

Reference: **Zhang et al. Section II-D**, Algorithm 1

### 6. Phase 2 Training: Adaptation Module (`training/phase2_adaptation.py`)

**Supervised learning** to train φ without ground-truth physics.

```python
trainer = Phase2Trainer('checkpoints/phase1/final.pt', config)
trainer.train()
```

**Process:**
1. Roll out Phase 1 policy (π + μ) across randomized environments
2. Collect (state-action history, ground-truth zt) pairs
3. Train φ: minimize `L = ||ẑt - zt||²` with Adam optimizer

Reference: **Zhang et al. Algorithm 2**

### 7. Evaluation & Generalization Sweep (`eval/generalization_sweep.py`)

Tests learned policy at increasing difficulty levels.

```python
evaluator = GeneralizationEvaluator(config)
evaluator.load_rma_model('phase1.pt', 'phase2.pt')
evaluator.run_sweep()
# Outputs: success rate, tracking error vs δ (like Figure 6)
```

**Δ levels:** 0 (nominal), 0.5 (training), 1, 2, 4, 8

**Baselines:**
- **RMA:** Learned policy π + φ (our method)
- **Fixed-Param:** Policy trained only on nominal (non-adaptive)
- **Oracle:** Expert with ground-truth physics (upper bound)

Reference: **Zhang et al. Section IV-C**, Figure 6

### 8. Deployment (`gazebo_deployment/rma_deployment_node.py`)

ROS 2 node for sim-to-sim validation in Gazebo.

```bash
ros2 run f1tenth_research rma_deployment \
    --actor_critic checkpoints/phase1/final.pt \
    --adaptation checkpoints/phase2/final.pt
```

**Pipeline:**
1. Subscribe to `/ego_racecar/odom` (velocity, pose, yaw)
2. Build state-action history buffer (k steps)
3. Estimate intrinsics: `ẑt = φ(history)`
4. Compute action: `at = π(xt, ẑt)`
5. Publish to `/drive` (AckermannDriveStamped)

**No ground-truth physics needed** (unlike Phase 1).

Reference: **Zhang et al. Section V**, Deployment Policy

---

## Training Workflow

### Step 1: Prepare Configuration

Edit `configs/rma_config.yaml` to tune:
- Domain randomization ranges
- Reward weights
- Network architectures
- PPO hyperparameters
- Evaluation difficulty levels

### Step 2: Phase 1 - PPO + IL

```bash
cd /research_ws/src/f1tenth_research/f1tenth_research

python training/phase1_ppo_il.py \
    --config configs/rma_config.yaml \
    --device cuda \
    --debug  # Use --debug for quick test run
```

**Outputs:**
- `checkpoints/phase1/final.pt` — Trained π + μ
- `logs/phase1/` — TensorBoard logs (reward, loss curves)

**Expected:** Policy learns to adapt to randomized physics, IL loss decays over time.

### Step 3: Phase 2 - Adaptation Module

```bash
python training/phase2_adaptation.py \
    --phase1_checkpoint checkpoints/phase1/final.pt \
    --config configs/rma_config.yaml \
    --device cuda
```

**Outputs:**
- `checkpoints/phase2/final.pt` — Trained adaptation module φ
- `logs/phase2/` — Training curves (φ estimation error)

**Expected:** φ learns to estimate task parameters from state-action history.

### Step 4: Evaluation

```bash
python eval/generalization_sweep.py \
    --rma_checkpoint checkpoints/phase1/final.pt \
    --adaptation_checkpoint checkpoints/phase2/final.pt \
    --config configs/rma_config.yaml
```

**Outputs:**
- `eval_results/results.json` — Metrics at each δ
- `eval_results/generalization_curves.png` — Success rate vs δ

**Expected:** Success rate ↓ as δ ↑, but RMA >> Fixed-Param baseline.

---

## Configuration Highlights

Key entries in `configs/rma_config.yaml`:

```yaml
environment:
  max_episode_steps: 1000
  control_frequency_hz: 50  # F1Tenth physical speed
  
  randomization:
    apply_post_scale_noise: true
    noise_fraction: 0.20
    mid_episode_disturbance: true  # Simulates payload shifts
  
  reward:
    weight_smoothness: 0.1
    weight_survival: 1.0
    weight_velocity_tracking: 0.5
    weight_yaw_rate_tracking: 0.3

phase1_training:
  total_timesteps: 100_000_000  # ~100M (scale to your hardware)
  debug_mode: false
  debug_timesteps: 100_000
  
  ppo:
    learning_rate: 3.0e-4
    gamma: 0.99
    gae_lambda: 0.95
    clip_ratio: 0.2
  
  il:
    il_weight_decay: 0.001  # α = exp(-0.001 * epoch)
    il_weight_start: 1.0

evaluation:
  delta_levels: [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
  episodes_per_delta: 100
```

---

## Mapping to Zhang et al. (2025)

Each component explicitly references the paper:

| Code Component | Zhang Section | Purpose |
|---|---|---|
| `randomization.py` | II-A | Domain randomization with size factor c |
| `reward.py` | II-C | 4-term composite reward |
| `experts/__init__.py` | II-B | Expert controller (PD*) with physics adaptation |
| `models/__init__.py` | III-C | Networks π, μ, V, φ |
| `phase1_ppo_il.py` | II-D | Joint PPO + IL training (Algorithm 1) |
| `phase2_adaptation.py` | III-D | Supervised φ training (Algorithm 2) |
| `generalization_sweep.py` | IV-C | Generalization evaluation (Figure 6) |
| `rma_deployment_node.py` | V | Deployment policy (π + φ) |

Each class docstring includes the relevant equation/section number.

---

## Dependencies

### Core
- `torch`, `torchvision` — Neural networks
- `gymnasium` — RL environment API
- `numpy`, `scipy` — Numerical computation
- `pyyaml` — Configuration parsing

### Robotics
- `rclpy`, `geometry_msgs`, `nav_msgs`, `ackermann_msgs` — ROS 2
- `f1tenth_gym` — F1Tenth simulator (optional, can run without)

### Utilities
- `matplotlib` — Plotting evaluation results
- `tensorboard` — Training visualization

**Installation:**
```bash
# Install PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other deps
pip install gymnasium pyyaml numpy matplotlib tensorboard

# Optional: f1tenth_gym for full sim integration
pip install f1tenth_gym
```

---

## Troubleshooting & Common Issues

### 1. **Missing `f1tenth_gym`**
If `f1tenth_gym` is not available, the `F1TenthRMAEnv` will run in mock mode (zeros for observations). 
- Install via: `pip install f1tenth_gym`
- Or: implement your own gym wrapper using ROS 2 topics

### 2. **CUDA Out of Memory**
Reduce batch size or num_envs:
```yaml
phase1_training:
  num_envs: 4    # Instead of 16
  batch_size: 128 # Instead of 256
```

### 3. **Training Too Slow**
- Use `debug_mode: true` to test pipeline quickly
- Reduce `total_timesteps` and `episodes_per_delta`
- Profile bottleneck (model, env, or data collection?)

### 4. **Poor Generalization**
- Increase `randomization.noise_fraction` and `mid_episode_disturbance`
- Tune IL decay rate: lower `il_weight_decay` for longer IL phase
- Increase `phase2_training.num_eval_episodes` for better adaptation data

---

## References

**Primary Paper:**
> Zhang et al. "A Learning-Based Quadcopter Controller With Extreme Adaptation". 
> IEEE Transactions on Robotics, 2025.

**Methods:**
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Domain Randomization: Tobin et al., "Domain Randomization for Transferring Deep Neural Networks..." (2017)
- Imitation Learning: Behav. cloning + DAgger ideas

---

## Future Enhancements

1. **Real Hardware:** Deploy on physical F1Tenth car (Gazebo → sim2real)
2. **LiDAR Integration:** Use full scan observations for better state representation
3. **MPCC Expert:** Implement full model predictive contouring control
4. **Uncertainty Quantification:** Bayesian networks for epistemic uncertainty
5. **Multi-Task:** Train on multiple tracks/objectives simultaneously
6. **Curriculum Learning:** Gradually increase δ during Phase 1 for stability

---

## License & Citation

Adaptation of Zhang et al. (2025) framework. Cite as:

```bibtex
@article{zhang2025learning,
  title={A Learning-Based Quadcopter Controller With Extreme Adaptation},
  author={Zhang, ..., et al.},
  journal={IEEE Transactions on Robotics},
  year={2025}
}
```

This F1Tenth adaptation is released under the same license as the original f1tenth_gym project.

---

## Contact & Issues

For questions or bugs related to this F1Tenth RMA implementation, refer to inline code comments (which all map back to Zhang et al. sections) and the paper itself for theoretical details.

---

**Last Updated:** June 2026  
**Framework Version:** PyTorch 2.0+, ROS 2 Foxy/Humble  
**Status:** Full framework implemented; ready for training and deployment validation
