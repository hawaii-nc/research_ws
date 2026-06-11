import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import pickle
import os
from datetime import datetime

class RecorderNode(Node):
    def __init__(self):
        super().__init__('recorder_node')

        # Storage
        self.dataset = []
        self.current_lidar = None
        self.current_speed = 0.0
        self.is_recording = False
        self.num_beams = 36

        # Subscribers
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10
        )
        # Listen to what Pure Pursuit publishes
        self.drive_sub = self.create_subscription(
            AckermannDriveStamped, '/drive', self.drive_callback, 10
        )

        # Status timer - print every 5 seconds
        self.create_timer(5.0, self.print_status)

        self.get_logger().info('Recorder ready')
        self.get_logger().info('Type r to start/stop, s to save, q to quit')

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 10.0, ranges)
        indices = np.linspace(0, len(ranges)-1, self.num_beams, dtype=int)
        self.current_lidar = ranges[indices]

    def odom_callback(self, msg):
        self.current_speed = msg.twist.twist.linear.x

    def drive_callback(self, msg):
        # Every time Pure Pursuit publishes a command,
        # record the current state + that command
        if not self.is_recording:
            return
        if self.current_lidar is None:
            return

        datapoint = {
            'lidar': self.current_lidar.copy(),
            'speed': float(self.current_speed),
            'steering': float(msg.drive.steering_angle),
            'throttle': float(msg.drive.speed)
        }
        self.dataset.append(datapoint)

    def print_status(self):
        if self.is_recording:
            self.get_logger().info(
                f'Recording... {len(self.dataset)} datapoints collected'
            )

    def save_dataset(self):
        if len(self.dataset) == 0:
            self.get_logger().warn('No data to save')
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'expert_data_{timestamp}.pkl'
        save_path = f'/research_ws/data/{filename}'

        os.makedirs('/research_ws/data', exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(self.dataset, f)

        self.get_logger().info(
            f'Saved {len(self.dataset)} datapoints to {save_path}'
        )
        return save_path


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()

    import threading

    def input_thread():
        while rclpy.ok():
            cmd = input(
                "r=start/stop recording | s=save | q=quit\n"
            )
            if cmd.strip() == 'r':
                if not node.is_recording:
                    node.is_recording = True
                    node.get_logger().info('RECORDING STARTED')
                else:
                    node.is_recording = False
                    node.get_logger().info(
                        f'RECORDING STOPPED - '
                        f'{len(node.dataset)} datapoints'
                    )
            elif cmd.strip() == 's':
                node.save_dataset()
            elif cmd.strip() == 'q':
                node.save_dataset()
                rclpy.shutdown()
                break

    thread = threading.Thread(target=input_thread, daemon=True)
    thread.start()

    rclpy.spin(node)


if __name__ == '__main__':
    main()
