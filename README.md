# research_ws — F1Tenth Research Workspace (Training)

ROS2 training workspace for my UNLV NSF REU work on **Rapid Motor Adaptation** for autonomous racing. This is where the policies in f1tenth-vader-deployment were trained and evaluated.

## What's Inside

- `src/f1tenth_research/` — Main ROS2 package: PPO training for LiDAR-based racing policies, imitation learning (`train_il.py`), sim environments, evaluation harness, Pure Pursuit baseline, teleoperation and data-recording nodes, trained Phase 1 checkpoints and TensorBoard logs.
- `training/` — Standalone imitation-learning training script.
- `validate_rma_project.sh` — Project validation helper.

## Results (summary)

The RMA policy trained here closed 44% of the gap to an oracle baseline and was validated on real hardware (81 trials, 4 friction conditions). Full results: https://github.com/hawaii-nc/NSF-REU-UNLV-Smart-Cities-2026

## Tech

Python · PyTorch (PPO, Imitation Learning) · ROS2 · Gazebo · LiDAR

Note: `install/`, `log/`, and `build/` artifacts from local colcon builds are kept for reference. For a clean run, clone fresh and rebuild with `colcon build`.
