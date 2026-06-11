import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import torch
import torch.nn as nn
import numpy as np

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
        model_path = '/research_ws/models/il_policy.pth'
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

        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        self.create_timer(0.05, self.drive)
        self.get_logger().info('IL Driver running')

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), self.max_range, ranges)
        ranges = np.where(np.isnan(ranges), self.max_range, ranges)
        indices = np.linspace(
            0, len(ranges)-1, self.num_beams, dtype=int
        )
        self.current_lidar = ranges[indices]

    def odom_callback(self, msg):
        self.current_speed = msg.twist.twist.linear.x

    def drive(self):
        if self.current_lidar is None:
            return

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
