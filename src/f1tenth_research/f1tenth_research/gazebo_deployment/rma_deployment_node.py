"""
ROS 2 Deployment Node for F1Tenth RMA
=====================================

Deployment-time policy execution in ROS 2 + Gazebo.

Subscribes to:
- /ego_racecar/odom: Odometry (velocity, pose)
- /scan: LiDAR scan (1080 beams, downsampled to 36 for policy input)

Publishes to:
- /drive: AckermannDriveStamped (steering + throttle commands)

Pipeline (Phase 1 mode, no adaptation module):
1. Build obs xt = [v, steer, v_des, steer_des, yaw_rate, lidar_36]
2. Use zero intrinsics zt (Phase 2 not yet trained)
3. Compute action: at = pi(xt, zt)
4. Map throttle [-1,1] -> velocity [0,8] m/s
5. Publish AckermannDriveStamped

Reference: Zhang et al. (2025) Section V - Deployment Policy
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import torch
import numpy as np
from collections import deque
from typing import Optional
import math
import argparse

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Quaternion

from f1tenth_research.models import RMAActorCritic, AdaptationModule


# Observation space constants -- must match training config
OBS_DIM = 41          # 5 base signals + 36 LiDAR beams
LIDAR_BEAMS = 36      # downsampled from 1080
LIDAR_MAX_RANGE = 10.0  # meters, same normalization as f1tenth_env.py
INTRINSICS_DIM = 8
ACTION_DIM = 2
WHEELBASE = 0.3302    # meters (lf + lr for F1Tenth, same as training)
MAX_VELOCITY = 8.0    # m/s, matches [-1,1] -> [0,8] mapping in training


class RMADeploymentNode(Node):
    """
    ROS 2 node for deploying trained RMA policy in Gazebo.

    Phase 1 mode: uses zero intrinsics (no adaptation module required).
    Phase 2 mode: estimates intrinsics from state-action history via phi.
    """

    def __init__(
        self,
        actor_critic_checkpoint: str,
        adaptation_checkpoint: Optional[str] = None,
        control_freq: float = 50.0,
    ):
        super().__init__('rma_deployment')

        self.control_freq = control_freq
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")

        # --- Load actor-critic (Phase 1) ---
        self.actor_critic = RMAActorCritic(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            intrinsics_dim=INTRINSICS_DIM,
            env_params_dim=7,
        ).to(self.device)
        ckpt = torch.load(actor_critic_checkpoint, map_location=self.device,
                          weights_only=False)
        self.actor_critic.load_state_dict(ckpt['actor_critic'])
        self.actor_critic.eval()
        self.get_logger().info(
            f"Actor-critic loaded from {actor_critic_checkpoint} (obs_dim={OBS_DIM})"
        )

        # --- Load adaptation module (Phase 2, optional) ---
        self.adaptation = None
        self.use_adaptation = False
        if adaptation_checkpoint is not None:
            # state_action_dim = obs_dim + action_dim
            self.adaptation = AdaptationModule(
                state_action_dim=OBS_DIM + ACTION_DIM,
                history_window=10,
                intrinsics_dim=INTRINSICS_DIM,
            ).to(self.device)
            ckpt2 = torch.load(adaptation_checkpoint, map_location=self.device,
                               weights_only=False)
            self.adaptation.load_state_dict(ckpt2['adaptation'])
            self.adaptation.eval()
            self.use_adaptation = True
            self.get_logger().info("Adaptation module loaded (Phase 2 mode)")
        else:
            self.get_logger().info(
                "No adaptation checkpoint provided -- running Phase 1 mode "
                "(zero intrinsics). Train Phase 2 phi to enable adaptation."
            )

        # --- State tracking ---
        self.current_velocity = 0.0
        self.current_steering = 0.0   # last commanded steering (proxy)
        self.current_yaw_rate = 0.0
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

        # LiDAR: 36-beam downsampled, normalized, initialized to max range
        self.lidar_obs = np.ones(LIDAR_BEAMS, dtype=np.float32)
        self.lidar_received = False

        # State-action history for adaptation module
        self.history = deque(maxlen=10)

        # --- ROS subscriptions ---
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )

        # --- Publisher ---
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        # --- Control timer ---
        self.timer = self.create_timer(1.0 / self.control_freq, self.control_loop)
        self.get_logger().info(
            f"RMA deployment node ready at {self.control_freq} Hz"
        )

    def odom_callback(self, msg: Odometry):
        """Extract velocity and yaw rate from odometry."""
        self.current_velocity = float(msg.twist.twist.linear.x)
        self.current_yaw_rate = float(msg.twist.twist.angular.z)

    def lidar_callback(self, msg: LaserScan):
        """
        Downsample 1080-beam LiDAR scan to 36 evenly-spaced beams,
        clip to LIDAR_MAX_RANGE, normalize to [0, 1].

        Matches exactly what _process_observation does in f1tenth_env.py
        so the policy receives the same format it was trained on.
        """
        ranges = np.array(msg.ranges, dtype=np.float32)
        # Replace inf/nan with max range
        ranges = np.where(np.isfinite(ranges), ranges, LIDAR_MAX_RANGE)
        # Downsample to LIDAR_BEAMS evenly-spaced indices
        indices = np.linspace(0, len(ranges) - 1, LIDAR_BEAMS).astype(int)
        downsampled = ranges[indices]
        # Normalize to [0, 1]
        self.lidar_obs = np.clip(downsampled, 0.0, LIDAR_MAX_RANGE) / LIDAR_MAX_RANGE
        self.lidar_received = True

    def _build_obs(self) -> np.ndarray:
        """
        Build 41D observation vector matching training format:
        [v_current, steering_current, v_desired, steering_desired, yaw_rate,
         lidar_0, ..., lidar_35]

        v_desired / steering_desired come from the previous action
        (same convention as f1tenth_env.py's _process_observation).
        """
        throttle_cmd = self.prev_action[1]
        v_des = (float(throttle_cmd) + 1.0) / 2.0 * MAX_VELOCITY
        steer_des = float(self.prev_action[0])

        base = np.array([
            self.current_velocity,
            self.current_steering,
            v_des,
            steer_des,
            self.current_yaw_rate,
        ], dtype=np.float32)

        return np.concatenate([base, self.lidar_obs]).astype(np.float32)

    def control_loop(self):
        """
        Main 50 Hz control loop.

        1. Build obs vector xt (41D)
        2. Estimate or zero intrinsics zt
        3. Compute deterministic action mean
        4. Map throttle [-1,1] -> velocity [0,8] m/s
        5. Publish AckermannDriveStamped
        """
        if not self.lidar_received:
            # Don't command anything until we have at least one LiDAR scan
            return

        with torch.no_grad():
            obs = self._build_obs()
            obs_tensor = torch.from_numpy(obs).float().to(self.device)

            # Intrinsics: use adaptation module if available, else zeros
            if self.use_adaptation and len(self.history) >= 10:
                history_array = np.array(list(self.history))
                history_tensor = torch.from_numpy(
                    history_array
                ).unsqueeze(0).float().to(self.device)
                intrinsics = self.adaptation(history_tensor).squeeze(0)
            else:
                # Use nominal physics params (all scales=1.0, delays=0.0)
                # rather than zeros -- the policy was never trained with
                # zero intrinsics, so zeros produce near-zero throttle.
                # Nominal zt is the encoder output for a default-physics car.
                nominal_params = torch.tensor(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    device=self.device
                )
                with torch.no_grad():
                    intrinsics = self.actor_critic.get_intrinsics(nominal_params)

            # Deterministic action mean (no sampling at deployment time)
            mean, _ = self.actor_critic.policy(obs_tensor, intrinsics)
            action_np = mean.cpu().numpy()

        # Store history for adaptation module
        state_action = np.concatenate([obs, action_np])
        self.history.append(state_action)
        self.prev_action = action_np.copy()
        self.current_steering = float(action_np[0])

        # Extract and clip commands
        steer_cmd = float(np.clip(action_np[0], -0.4189, 0.4189))
        throttle_cmd = float(np.clip(action_np[1], -1.0, 1.0))

        # Map throttle [-1,1] -> velocity [0,8] m/s (same as training)
        velocity_cmd = (throttle_cmd + 1.0) / 2.0 * MAX_VELOCITY
        # No velocity floor -- policy now outputs meaningful throttle

        # Publish
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = steer_cmd
        msg.drive.speed = velocity_cmd
        self.drive_pub.publish(msg)


def main(args=None):
    """
    Entry point for RMA deployment node.

    Phase 1 (no adaptation module -- use zero intrinsics):
        ros2 run f1tenth_research rma_deployment \\
            --actor_critic /research_ws/src/f1tenth_research/checkpoints/phase1_lidar/final.pt

    Phase 2 (with adaptation module -- full RMA):
        ros2 run f1tenth_research rma_deployment \\
            --actor_critic /research_ws/src/f1tenth_research/checkpoints/phase1_lidar/final.pt \\
            --adaptation /research_ws/src/f1tenth_research/checkpoints/phase2/final.pt
    """
    parser = argparse.ArgumentParser(description='RMA Deployment Node')
    parser.add_argument('--actor_critic', type=str, required=True,
                        help='Path to Phase 1 actor-critic checkpoint')
    parser.add_argument('--adaptation', type=str, default=None,
                        help='Path to Phase 2 adaptation module (optional)')
    parser.add_argument('--control_freq', type=float, default=50.0,
                        help='Control frequency in Hz')
    parsed, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)

    node = RMADeploymentNode(
        actor_critic_checkpoint=parsed.actor_critic,
        adaptation_checkpoint=parsed.adaptation,
        control_freq=parsed.control_freq,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
