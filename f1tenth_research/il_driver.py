import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import torch
import torch.nn as nn
import numpy as np
import time


class DrivingPolicy(nn.Module):
    def __init__(self, lidar_beams=36):
        super().__init__()
        input_dim = lidar_beams + 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        return self.network(x)


class ILDriver(Node):
    def __init__(self):
        super().__init__('il_driver')

        # Load trained model
        self.declare_parameter('model_path', '/research_ws/models/il_policy.pth')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.model = DrivingPolicy(lidar_beams=36)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )
        self.model.eval()
        self.get_logger().info(f'Model loaded from {model_path}')

        self.current_lidar = None
        self.current_speed = 0.0
        self.max_range = 10.0
        self.max_speed = 4.0
        self.num_beams = 36

        # Crash/stuck detection
        self.stuck_threshold_speed = 0.15      # m/s - below this counts as "not moving"
        self.stuck_threshold_time = 1.5        # seconds of not moving = crashed
        self.last_moving_time = time.time()
        self.reset_count = 0
        self.resetting = False
        self.resume_time = None
        self.reset_pause = 1.0  # seconds to wait after reset before driving resumes

        # Starting pose (from sim.yaml: sx, sy, stheta)
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0

        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10
        )

        self.get_logger().info('IL Driver loaded.')
        input(">>> Press Enter to start driving...")
        self.get_logger().info('IL Driver running')

        self.create_timer(0.05, self.drive)

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), self.max_range, ranges)
        ranges = np.where(np.isnan(ranges), self.max_range, ranges)
        indices = np.linspace(
            0, len(ranges) - 1, self.num_beams, dtype=int
        )
        self.current_lidar = ranges[indices]

    def odom_callback(self, msg):
        self.current_speed = msg.twist.twist.linear.x

    def publish_reset_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = self.start_x
        msg.pose.pose.position.y = self.start_y
        msg.pose.pose.position.z = 0.0

        # yaw -> quaternion (yaw = self.start_yaw)
        half_yaw = self.start_yaw / 2.0
        msg.pose.pose.orientation.z = np.sin(half_yaw)
        msg.pose.pose.orientation.w = np.cos(half_yaw)

        self.pose_pub.publish(msg)

    def stop_car(self):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = 0.0
        msg.drive.speed = 0.0
        self.drive_pub.publish(msg)

    def drive(self):
        if self.current_lidar is None:
            return

        now = time.time()

        # If we're in the post-reset pause, just wait
        if self.resetting:
            self.stop_car()
            if now >= self.resume_time:
                self.resetting = False
                self.last_moving_time = now
                self.get_logger().info('Resuming driving.')
            return

        # Predict action
        lidar_norm = self.current_lidar / self.max_range
        speed_norm = np.array([self.current_speed / self.max_speed])
        state = np.concatenate([lidar_norm, speed_norm])
        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            output = self.model(state_tensor)

        steering = float(output[0, 0])
        throttle = float(output[0, 1])
        steering = np.clip(steering, -0.4, 0.4)
        throttle = np.clip(throttle, 0.0, 4.0)

        # Stuck/crash detection
        if abs(self.current_speed) > self.stuck_threshold_speed:
            self.last_moving_time = now

        if (throttle > self.stuck_threshold_speed and
                now - self.last_moving_time > self.stuck_threshold_time):
            self.reset_count += 1
            self.get_logger().warn(
                f'Car appears stuck/crashed. Resetting to start (reset #{self.reset_count}).'
            )
            self.stop_car()
            self.publish_reset_pose()
            self.resetting = True
            self.resume_time = now + self.reset_pause
            return

        # Normal drive command
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering
        msg.drive.speed = throttle
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ILDriver()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
