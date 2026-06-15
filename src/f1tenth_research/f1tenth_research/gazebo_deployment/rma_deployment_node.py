"""
ROS 2 Deployment Node for F1Tenth RMA
=====================================

Deployment-time policy execution in ROS 2 + Gazebo.

Subscribes to:
- /ego_racecar/odom: Odometry (velocity, pose)
- /scan (optional): LiDAR scan

Publishes to:
- /drive: AckermannDriveStamped (steering + throttle commands)

Pipeline:
1. Collect state-action history (~0.2s window)
2. Estimate intrinsics using learned adaptation module φ
3. Compute action: at = π(xt, ẑt)
4. Send command to simulator/hardware

Reference: Zhang et al. (2025) Section V - Deployment Policy
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Optional, Dict
import warnings

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Quaternion
import math

# Import trained models
from f1tenth_research.models import RMAActorCritic, AdaptationModule
from f1tenth_research.envs import F1TenthRMAEnv


class RMADeploymentNode(Node):
    """
    ROS 2 node for deploying trained RMA policy.
    
    Integrates:
    - Learned policy π
    - Learned adaptation module φ
    - State aggregation from ROS topics
    - Command publication to simulator
    """
    
    def __init__(
        self,
        actor_critic_checkpoint: str,
        adaptation_checkpoint: str,
        control_freq: float = 50.0,  # Hz
    ):
        """
        Initialize RMA deployment node.
        
        Args:
            actor_critic_checkpoint: Path to trained π + μ model
            adaptation_checkpoint: Path to trained φ (adaptation module)
            control_freq: Control frequency (Hz)
        """
        super().__init__('rma_deployment')
        
        self.control_freq = control_freq
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")
        
        # Load models
        self.actor_critic = RMAActorCritic().to(self.device)
        ckpt = torch.load(actor_critic_checkpoint, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt['actor_critic'])
        self.actor_critic.eval()
        
        # Load adaptation module
        self.adaptation = AdaptationModule(
            state_action_dim=7,  # obs (5) + action (2)
            history_window=10,   # ~0.2s at 50Hz
            intrinsics_dim=8,
        ).to(self.device)
        
        ckpt = torch.load(adaptation_checkpoint, map_location=self.device)
        self.adaptation.load_state_dict(ckpt['adaptation'])
        self.adaptation.eval()
        
        self.get_logger().info("Models loaded successfully")
        
        # State tracking
        self.current_velocity = 0.0
        self.current_yaw = 0.0
        self.current_yaw_rate = 0.0
        self.current_position = np.array([0.0, 0.0])
        
        # State-action history for adaptation
        self.history = deque(maxlen=10)
        self.prev_action = None
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10
        )
        
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        # Publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )
        
        # Control loop timer
        period = 1.0 / self.control_freq
        self.timer = self.create_timer(period, self.control_loop)
        
        self.get_logger().info(f"RMA deployment node initialized at {self.control_freq} Hz")
    
    def odom_callback(self, msg: Odometry):
        """
        Process odometry update.
        
        Extracts position, velocity, and heading.
        """
        # Position
        self.current_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])
        
        # Velocity
        self.current_velocity = msg.twist.twist.linear.x
        
        # Heading (yaw) from quaternion
        q = msg.pose.pose.orientation
        self.current_yaw = self._quat_to_yaw(q)
        
        # Yaw rate
        self.current_yaw_rate = msg.twist.twist.angular.z
    
    def lidar_callback(self, msg: LaserScan):
        """
        Process LiDAR scan.
        
        TODO: Extract range data if using LiDAR observations.
        For now, ignored (not required for RMA).
        """
        pass
    
    def _quat_to_yaw(self, q: Quaternion) -> float:
        """Convert quaternion to yaw angle (rad)."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def _build_state_vector(self) -> np.ndarray:
        """
        Build observation vector from ROS state.
        
        State vector xt: [v_current, steering_current, v_desired, steering_desired, yaw_rate, ...]
        
        For now, using simplified version:
        [velocity, yaw, yaw_rate, position_x, position_y]
        
        Returns:
            State vector (5D)
        """
        # Placeholder implementation
        # In full version, would include:
        # - Current velocity
        # - Current steering angle (from steering feedback)
        # - Commanded velocity (from last action)
        # - Commanded steering (from last action)
        # - Yaw rate (from IMU)
        
        state = np.array([
            self.current_velocity,
            self.current_yaw,
            self.current_yaw_rate,
            self.current_position[0],
            self.current_position[1],
        ], dtype=np.float32)
        
        return state
    
    def control_loop(self):
        """
        Main control loop: estimate intrinsics and compute action.
        
        Pipeline:
        1. Build state vector from ROS data
        2. Estimate intrinsics using φ(history)
        3. Compute action using π(state, intrinsics)
        4. Publish command
        """
        with torch.no_grad():
            # Build state vector
            state = self._build_state_vector()
            state_tensor = torch.from_numpy(state).float().to(self.device)
            
            # Estimate intrinsics from history
            if len(self.history) >= self.adaptation.history_window:
                # Build history tensor
                history_array = np.array(list(self.history)[-self.adaptation.history_window:])
                history_tensor = torch.from_numpy(history_array).unsqueeze(0).float().to(self.device)
                estimated_intrinsics = self.adaptation(history_tensor).squeeze(0)
            else:
                # Use zero intrinsics if history not full
                estimated_intrinsics = torch.zeros(8).to(self.device)
            
            # Compute deterministic action mean for deployment.
            mean, _ = self.actor_critic.policy(state_tensor, estimated_intrinsics)
            action_np = mean.cpu().numpy()
            
            # Store in history for next step
            state_action = np.concatenate([state, action_np])
            self.history.append(state_action)
            
            # Extract steering and throttle
            steering_cmd = np.clip(action_np[0], -0.4189, 0.4189)
            throttle_cmd = np.clip(action_np[1], -1.0, 1.0)
            
            # Publish command
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.drive.steering_angle = float(steering_cmd)
            msg.drive.speed = float(throttle_cmd)  # Maps to velocity command
            
            self.drive_pub.publish(msg)
    
    def destroy_node(self):
        """Cleanup."""
        super().destroy_node()
        self.get_logger().info("RMA deployment node destroyed")


def main(args=None):
    """
    Entry point for RMA deployment node.
    
    Usage:
        ros2 run f1tenth_research rma_deployment \
            --actor_critic path/to/phase1_final.pt \
            --adaptation path/to/phase2_final.pt
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='RMA Deployment Node')
    parser.add_argument('--actor_critic', type=str, required=True,
                       help='Path to actor-critic checkpoint')
    parser.add_argument('--adaptation', type=str, required=True,
                       help='Path to adaptation module checkpoint')
    parser.add_argument('--control_freq', type=float, default=50.0,
                       help='Control frequency (Hz)')
    
    args, unknown = parser.parse_known_args()
    
    rclpy.init(args=unknown)
    
    node = RMADeploymentNode(
        actor_critic_checkpoint=args.actor_critic,
        adaptation_checkpoint=args.adaptation,
        control_freq=args.control_freq,
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
